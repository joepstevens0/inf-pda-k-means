#if KMEANS_MODE_MPI == 1
#include "helper_functions.h"
#include "kmeans.h"
#include <iostream>
#include <algorithm>
#include <mpi.h>
#include <stdio.h>

struct KMeansItInput {
    const size_t numPoints;
    const size_t pointSize;
    const std::vector<double> &allData;
    std::vector<Point> &centroids;
    std::vector<int>& pointCounts;
    const int numClusters;
    FileCSVWriter &centroidDebugFile;
    FileCSVWriter &clustersDebugFile;
    int numThreads;
};

struct KMeansItOutput {
    size_t numSteps;
    std::vector<int> bestClusters;
    double bestDistSquaredSum;
    std::vector<int> clusters;
};

/* Only execute by rank 0 */
void getBestCluster(int srcRank, int numPoints, std::vector<int> &outBestCluster) {
    std::vector<int> bestCluster(numPoints);

    // Broadcast sending rank
    MPI_Bcast(&srcRank, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // If best cluster not in process 0, recv if from the process that has it
    if (srcRank != 0) {
        int tag = 0;
        MPI_Status status;
        MPI_Recv(&outBestCluster, numPoints, MPI_INT, srcRank, tag, MPI_COMM_WORLD, &status);
    }
}

int kmeansMPIIteration(KMeansItOutput &out, KMeansItInput &in) {

    bool changed = true;
    out.numSteps = 0;

    // write starting step clusters and centroids to the debug files if open
    if (in.centroidDebugFile.is_open())
        in.centroidDebugFile.write(in.centroids);
    if (in.clustersDebugFile.is_open())
        in.clustersDebugFile.write(out.clusters);

    while (changed) {
        changed = false;
        double distSquaredSum = 0;

        for (size_t pointIndex = 0; pointIndex < in.numPoints; pointIndex++) {
            int newCluster;
            double dist;

            findClosestCentroidIndexAndDistance(pointIndex, in.pointSize,
                                                in.allData, in.centroids,
                                                newCluster, dist);

            distSquaredSum += dist;

            if (newCluster != out.clusters[pointIndex]) {
                out.clusters[pointIndex] = newCluster;
                changed = true;
            }
        }
        if (changed) // re-calculate the centroids based on current clustering
            moveCentroidsToAverage(in.centroids, out.clusters, in.numPoints,
                                   in.pointSize, in.allData, in.pointCounts);

        // Keep track of best clustering
        if (distSquaredSum < out.bestDistSquaredSum) {
            out.bestClusters = out.clusters;
            out.bestDistSquaredSum = distSquaredSum;
        }
        ++out.numSteps;

        // write the step to the debug files if open
        if (in.centroidDebugFile.is_open())
            in.centroidDebugFile.write(in.centroids);
        if (in.clustersDebugFile.is_open())
            in.clustersDebugFile.write(out.clusters);
    }

    return 0;
}

KmeansOut kmeansMPI(KMeansIn input, int rank, int totalUsedCores, int totalCores) {

    // Divide repetitions over all cores (of all nodes) -> each core having 1 thread running
    int repsPerNode = input.repetitions / totalUsedCores;
    int extraReps = input.repetitions % totalUsedCores;
    
    int start_index = 0;
    int end_index = 0;
    // Only give cores we're using reps
    if (rank < totalUsedCores) {
        start_index =  (repsPerNode + 1) * std::min(rank, extraReps) // cores with an extra rep
                    + repsPerNode * std::max(0, (int)rank - extraReps); // cores without an extra rep
         end_index = std::min(input.repetitions, (int)start_index + repsPerNode + (rank < extraReps? 1 : 0));
    }

    printf("Hello %d/%d, reps %d-%d\n", rank, totalUsedCores, start_index, end_index);

    KmeansOut out;

    out.stepsPerRepetition.resize(input.repetitions, 0);
    out.bestDistSquaredSum = std::numeric_limits<double>::max();
    out.bestClusters = std::vector<int>(input.numPoints, -1);
    size_t it_of_best_cluster = 0;
    std::vector<std::vector<Point>> centroids_per_repetition(input.repetitions, std::vector<Point>(input.numClusters));
    
    for (size_t r = 0; r < input.repetitions; r++) {
        chooseCentroidsAtRandomFromDataset(input.rng, input.numPoints,
                                                input.pointSize, input.allData,
                                                centroids_per_repetition[r]);
    }

    for (size_t r = start_index; r < end_index; r++) {

        std::vector<int> pointCounts;
        pointCounts.resize(input.numClusters);

        // Create the iteration parameters
        KMeansItInput itinput{input.numPoints,        input.pointSize,
                            input.allData,          centroids_per_repetition[r], pointCounts,
                            input.numClusters,      input.centroidDebugFile,
                            input.clustersDebugFile, input.numThreads};

        // create iteration output struct
        KMeansItOutput itoutput;
        itoutput.bestDistSquaredSum = std::numeric_limits<double>::max();
        // Init closest centroid index for every point: 'unknown'(-1)
        itoutput.clusters = std::vector<int>(input.numPoints, -1);
        itoutput.numSteps = 0;

        // start iteration
        kmeansMPIIteration(itoutput, itinput);

        // update num of steps for this iteration
        out.stepsPerRepetition[r] = itoutput.numSteps;

        if (itoutput.bestDistSquaredSum <= out.bestDistSquaredSum) {
            // take the best clusters from te lowest repetition
            if (itoutput.bestDistSquaredSum != out.bestDistSquaredSum || r < it_of_best_cluster){
                out.bestClusters = itoutput.clusters;
                out.bestDistSquaredSum = itoutput.bestDistSquaredSum;
                it_of_best_cluster = r;
            }
        }
    }

    if (rank == 0){
        // receive results from other processes

        // gather best distSquaredSums
        std::vector<double> distSquaredSums;
        distSquaredSums.resize(totalCores);
        MPI_Gather(&out.bestDistSquaredSum, 1, MPI_DOUBLE, distSquaredSums.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // gather iteration of best cluster
        std::vector<int> it_of_best_clusters;
        it_of_best_clusters.resize(totalCores);
        MPI_Gather(&it_of_best_cluster, 1, MPI_INT, it_of_best_clusters.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // gather best clusters
        // std::vector<int> bestClusters;
        // bestClusters.resize(totalCores*input.numPoints);
        // MPI_Gather(out.bestClusters.data(), input.numPoints, MPI_INT, bestClusters.data(), input.numPoints, MPI_INT, 0, MPI_COMM_WORLD);

        // reduce num steps per repetition
        std::vector<int> steps;
        steps.resize(out.stepsPerRepetition.size());
        MPI_Reduce(out.stepsPerRepetition.data(), steps.data(), out.stepsPerRepetition.size(), MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        out.stepsPerRepetition = steps;

        // find best cluster from all repetitions
        int bestClusterSrcRank = 0;
        for (int i = 0; i < totalCores;++i){
            if (distSquaredSums[i] <= out.bestDistSquaredSum) {
                // take the best clusters from te lowest repetition
                if (distSquaredSums[i] != out.bestDistSquaredSum || it_of_best_clusters[i] < it_of_best_cluster){
                    bestClusterSrcRank = i;
                    out.bestDistSquaredSum = distSquaredSums[i];
                    it_of_best_cluster = it_of_best_clusters[i];
                }
            }
        }
        getBestCluster(bestClusterSrcRank, input.numPoints, out.bestClusters);

    } else {
        // send results to process 0
        MPI_Gather(&out.bestDistSquaredSum, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&it_of_best_cluster, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
        // MPI_Gather(out.bestClusters.data(), input.numPoints, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Reduce(out.stepsPerRepetition.data(), nullptr, out.stepsPerRepetition.size(), MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        
        // Recv broadcast who has best cluster
        int srcRank;
        MPI_Bcast(&srcRank, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // If has best cluster send to root process (0)
        if (rank == srcRank) {
            int tag = 0;
            MPI_Send(out.bestClusters.data(), input.numPoints, MPI_INT, 0, tag, MPI_COMM_WORLD);
        }
    }
    return out;
}

#endif
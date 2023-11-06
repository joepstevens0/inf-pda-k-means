#include "helper_functions.h"
#include "kmeans.h"
#include <iostream>
#include <omp.h>

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

int kmeansOpenMPIteration(KMeansItOutput &out, KMeansItInput &in) {

    bool changed = true;
    out.numSteps = 0;

    while (changed) {
        changed = false;
        double distSquaredSum = 0;

        #pragma omp parallel for schedule(static) num_threads(in.numThreads) reduction(+:distSquaredSum)
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
    }

    return 0;
}

KmeansOut kmeansOpenMP(KMeansIn input) {
    KmeansOut out;
    out.stepsPerRepetition.resize(input.repetitions);
    out.bestDistSquaredSum = std::numeric_limits<double>::max();
    out.bestClusters = std::vector<int>(input.numPoints, -1);
    size_t it_of_best_cluster = 0;

    std::vector<std::vector<Point>> centroids_per_repetition(input.repetitions, std::vector<Point>(input.numClusters));

    for (size_t r = 0; r < input.repetitions; r++) {
        chooseCentroidsAtRandomFromDataset(input.rng, input.numPoints,
                                                input.pointSize, input.allData,
                                                centroids_per_repetition[r]);
    }

    // Do the k-means routine a number of times, each time starting from
    // different random centroids (use Rng::pickRandomIndices), and keep
    // the best result of these repetitions.
    #pragma omp parallel for schedule(dynamic) num_threads(input.numThreads)
    for (size_t r = 0; r < input.repetitions; r++) {

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
        kmeansOpenMPIteration(itoutput, itinput);

        // update num of steps for this iteration
        out.stepsPerRepetition[r] = itoutput.numSteps;

        #pragma omp critical
        if (itoutput.bestDistSquaredSum <= out.bestDistSquaredSum) {

            // take the best clusters from te lowest repetition
            if (itoutput.bestDistSquaredSum != out.bestDistSquaredSum || r < it_of_best_cluster){
                out.bestClusters = itoutput.clusters;
                out.bestDistSquaredSum = itoutput.bestDistSquaredSum;
                it_of_best_cluster = r;
            }
        }
    }

    return out;
}
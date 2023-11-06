#include "helper_functions.h"
#include "kmeans.h"
#include <iostream>
#include <string.h>
#include <thrust/reduce.h>

struct KMeansItInput {
    const size_t numPoints;
    const size_t pointSize;
    const std::vector<double> &allData;
    std::vector<double> &centroids;
    const int numClusters;
    FileCSVWriter &centroidDebugFile;
    FileCSVWriter &clustersDebugFile;
    int numThreads;
    int numBlocks;
};

struct KMeansItOutput {
    size_t numSteps;
    std::vector<int> bestClusters;
    double bestDistSquaredSum;
    std::vector<int> clusters;
};

__device__ 
void closestCentroidForPoint(const size_t pointIndex, const size_t pointSize, 
                            const double* allData, const double* centroids, const size_t numCentroids,
                            int &newCluster, double& bestDist){
    newCluster = -1;
    bestDist = 3.40282e+38; // can only get better

    const size_t p = pointIndex * pointSize;

    // Loop over all centroid points
    for (size_t i = 0; i < numCentroids; i++) {
        double dist = 0;

        // Calculate quadratic euclidean distance between data point(1) and
        // centroid point(2) [ sqrt( (x1 - x2)^2 + (y1 - y2)^2 + ...) ]
        for (size_t dim = 0; dim < pointSize; dim++)
            dist += pow(allData[p + dim] - centroids[i*pointSize + dim], 2.);

        // Change cluster index for point when distance is smaller then current
        // best
        if (dist < bestDist) {
            newCluster = i;
            bestDist = dist;
        }
    }
}

__global__
void findClosestCentroidIndexAndDistance(const size_t numPoints, const size_t pointSize, const int numClusters,
                                        const double* allData, const double* centroids, int* clusters,
                                        double* distSquaredSum, bool* changed) {
    // calc points per threads and extras
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int pointsPerThread = numPoints/(blockDim.x*gridDim.x);
    int extraPoints = numPoints%(blockDim.x*gridDim.x);

    // Calculate point range for every thread
    int start_index =  (pointsPerThread + 1) * min(t, extraPoints) // threads with an extra point
                 + pointsPerThread * max(0, (int)t - extraPoints); // threads without an extra point
    int end_index = min(numPoints,(size_t)start_index + pointsPerThread + (t < extraPoints? 1 : 0));


    // Go over point range (every thread)
    for (int pointIndex = start_index; pointIndex < end_index; pointIndex++) {
        int newCluster = -1;
        double dist;

        closestCentroidForPoint(pointIndex, pointSize, allData, centroids, numClusters, newCluster, dist);

        distSquaredSum[t] += dist;

        if (newCluster != clusters[pointIndex]) {
            clusters[pointIndex] = newCluster;
            *changed = true;
        }
    }
}

__global__
void resetChangedAndDist(bool* changed, double* distSquaredSum) {
    *changed = false;
    memset(distSquaredSum, 0, sizeof(distSquaredSum));
}

void moveCentroidsToAverage(const size_t numPoints, const size_t pointSize, const size_t numCentroids,
                            double* centroids, int* clusters, double* allData) {

    // reset all centroids to 0
    for (int i = 0; i < numCentroids*pointSize; ++i)
        centroids[i] = 0;

    // set per centroid point counters to 0
    int pointCounts[numCentroids];
    memset(pointCounts, 0, numCentroids * sizeof(int));

    // Loop over all points of dataset
    for (size_t index = 0; index < numPoints; index++) {

        const int c = clusters[index];

        // add all dimensions to the centroid
        const size_t p = index * pointSize;
        for (int dim = 0; dim < pointSize; ++dim)
            centroids[c*pointSize + dim] += allData[p + dim];
        pointCounts[c] += 1;
    }

    // average out the centroids
    for (int i = 0; i < numCentroids; ++i) {
        if (pointCounts[i] > 0)
            for (size_t dim = 0; dim < pointSize; dim++)
                centroids[i*pointSize + dim] /= pointCounts[i];
    }
}


int kmeansCUDAIteration(KMeansItOutput &out, KMeansItInput &in, 
                        double* cuAllData, double* cuCentroids, 
                        int* cuClusters, double* cuDistSquaredSum, bool* cuChanged) {

    bool changed = true;
    std::vector<double> distSquaredSum(in.numBlocks*in.numThreads);
    out.numSteps = 0;

    // Reset clusters/centroids every iteration
    cudaMemcpy(cuClusters, out.clusters.data(), out.clusters.size()*sizeof(int), cudaMemcpyHostToDevice);
    

    while (changed) {
        // Reset variables
        changed = false;
        std::fill(distSquaredSum.begin(), distSquaredSum.end(), 0);
        cudaMemcpy(cuCentroids, in.centroids.data(), in.centroids.size()*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(cuChanged, &changed, sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(cuDistSquaredSum, distSquaredSum.data(), distSquaredSum.size()*sizeof(double), cudaMemcpyHostToDevice);

        findClosestCentroidIndexAndDistance<<<in.numBlocks, in.numThreads>>>(in.numPoints, in.pointSize, in.numClusters, 
                                                                                cuAllData, cuCentroids, cuClusters,
                                                                                cuDistSquaredSum, cuChanged);

        // Copy result from GPU
        cudaMemcpy(distSquaredSum.data(), cuDistSquaredSum, distSquaredSum.size()*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&changed, cuChanged, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(out.clusters.data(), cuClusters, out.clusters.size()*sizeof(int), cudaMemcpyDeviceToHost);

        if (changed) {  // re-calculate the centroids based on current clustering
            moveCentroidsToAverage((size_t)in.numPoints, (size_t)in.pointSize, (size_t)in.numClusters,
                                            (double*)in.centroids.data(), (int*)out.clusters.data(), (double*)in.allData.data());
        }

        double dist = thrust::reduce(distSquaredSum.begin(), distSquaredSum.end());


        // Keep track of best clustering
        if (dist < out.bestDistSquaredSum) {
            out.bestClusters = out.clusters;
            out.bestDistSquaredSum = dist;
        }
        ++out.numSteps;
    }

    return 0;
}

KmeansOut kmeansCUDA(KMeansIn input) {
    
    // Init output struct obj
    KmeansOut out;
    out.stepsPerRepetition.resize(input.repetitions);
    out.bestDistSquaredSum = std::numeric_limits<double>::max();
    out.bestClusters = std::vector<int>(input.numPoints, -1);
    size_t it_of_best_cluster = 0;

    // Run seeded generator serial on CPU so reps has same start-centroids every time
    // (in case repetitions are run parallel)
    std::vector<std::vector<Point>> centroids_per_repetition(input.repetitions, std::vector<Point>(input.numClusters));
    std::vector<std::vector<double>> flat_centroids_per_repetition(input.repetitions);
    for (size_t r = 0; r < input.repetitions; r++) {
        chooseCentroidsAtRandomFromDataset(input.rng, input.numPoints,
                                            input.pointSize, input.allData,
                                            centroids_per_repetition[r]);
        // Flatten centroids array
        for (Point point: centroids_per_repetition[r])
            for (double i: point)
                flat_centroids_per_repetition[r].push_back(i);
    }

    // Init array-pointers for copying to GPU
    double* cuAllData;
    double* cuCentroids;
    int* cuClusters;
    double* cuDistSquaredSum;
    bool* cuChanged;

    // For memory alloc before repetitions start
    std::vector<double> distSquaredSum(input.numBlocks*input.numThreads); 

    // Init closest centroid index for every point: 'unknown'(-1)
    // (same for all repetitions)
    std::vector<int> startClusters(input.numPoints, -1);

    // Allocate memory for GPU (reuse for every rep)
    cudaMalloc(&cuAllData, input.allData.size()*sizeof(double));
    cudaMalloc(&cuCentroids, input.numClusters*input.pointSize*sizeof(double));
    cudaMalloc(&cuClusters, startClusters.size()*sizeof(int));
    cudaMalloc(&cuDistSquaredSum, distSquaredSum.size()*sizeof(double));
    cudaMalloc(&cuChanged, sizeof(bool));

    // Copy usable information for all repetition to GPU
    cudaMemcpy(cuAllData, input.allData.data(), input.allData.size()*sizeof(double), cudaMemcpyHostToDevice);

    // Do the k-means routine a number of times, each time starting from
    // different random centroids (use Rng::pickRandomIndices), and keep
    // the best result of these repetitions.
    for (size_t r = 0; r < input.repetitions; r++) {
        
        // Create the iteration parameters
        KMeansItInput itinput{input.numPoints,       input.pointSize,
                            input.allData,           flat_centroids_per_repetition[r],
                            input.numClusters,       input.centroidDebugFile,
                            input.clustersDebugFile, input.numThreads, input.numBlocks};

        // create iteration output struct
        KMeansItOutput itoutput;
        itoutput.bestDistSquaredSum = std::numeric_limits<double>::max();
        itoutput.clusters = startClusters;
        itoutput.numSteps = 0;

        // start iteration
        kmeansCUDAIteration(itoutput, itinput, 
                            cuAllData, cuCentroids, 
                            cuClusters, cuDistSquaredSum, cuChanged);

        // update num of steps for this iteration
        out.stepsPerRepetition[r] = itoutput.numSteps;

        // (in case repetitions are run parallel)
        if (itoutput.bestDistSquaredSum <= out.bestDistSquaredSum) {

            // take the best clusters from the lowest repetition
            if (itoutput.bestDistSquaredSum != out.bestDistSquaredSum || r < it_of_best_cluster){
                out.bestClusters = itoutput.clusters;
                out.bestDistSquaredSum = itoutput.bestDistSquaredSum;
                it_of_best_cluster = r;
            }
        }
    }

    // free cuda memory
    cudaFree(cuAllData);
    cudaFree(cuCentroids);
    cudaFree(cuClusters);
    cudaFree(cuDistSquaredSum);
    cudaFree(cuChanged);

    return out;
}
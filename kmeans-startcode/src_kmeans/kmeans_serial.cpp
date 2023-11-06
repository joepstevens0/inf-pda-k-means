#include "helper_functions.h"
#include "kmeans.h"
#include <iostream>

struct KMeansItInput {
    const size_t numPoints;
    const size_t pointSize;
    const std::vector<double> &allData;
    std::vector<Point> &centroids;
    std::vector<int>& pointCounts;
    const int numClusters;
    FileCSVWriter &centroidDebugFile;
    FileCSVWriter &clustersDebugFile;
};

struct KMeansItOutput {
    size_t numSteps;
    std::vector<int> bestClusters;
    double bestDistSquaredSum;
    std::vector<int> clusters;
};

int kmeansSerialIteration(KMeansItOutput &out, KMeansItInput &in) {

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

KmeansOut kmeansSerial(KMeansIn input) {

    // to save the number of steps each rep needed
    std::vector<int> stepsPerRepetition(input.repetitions);

    // total points per cluster
    std::vector<int> pointCounts;
    pointCounts.resize(input.numClusters);

    // Create the iteration parameters
    std::vector<Point> centroids(input.numClusters);
    KMeansItInput itinput{input.numPoints,        input.pointSize,
                          input.allData,          centroids, pointCounts,
                          input.numClusters,      input.centroidDebugFile,
                          input.clustersDebugFile};

    // create iteration output struct
    KMeansItOutput itoutput;
    itoutput.bestDistSquaredSum = std::numeric_limits<double>::max();
    itoutput.clusters = std::vector<int>(input.numPoints, -1);

    // Do the k-means routine a number of times, each time starting from
    // different random centroids (use Rng::pickRandomIndices), and keep
    // the best result of these repetitions.
    for (size_t r = 0; r < input.repetitions; r++) {

        // Pick k random centroid points from the dataset, with k the number of
        // clusters.
        chooseCentroidsAtRandomFromDataset(input.rng, input.numPoints,
                                           input.pointSize, input.allData,
                                           itinput.centroids);

        // Init closest centroid index for every point: 'unknown'(-1)
        std::fill(itoutput.clusters.begin(), itoutput.clusters.end(), -1);
        itoutput.numSteps = 0;
        kmeansSerialIteration(itoutput, itinput);

        stepsPerRepetition[r] = itoutput.numSteps;

        // Make sure debug logging is only done on first iteration ; subsequent
        // checks with is_open will indicate that no logging needs to be done
        // anymore.
        input.centroidDebugFile.close();
        input.clustersDebugFile.close();
    }

    return {itoutput.bestDistSquaredSum, itoutput.bestClusters,
            stepsPerRepetition};
}
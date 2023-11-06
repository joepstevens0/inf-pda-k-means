#include "helper_functions.h"
#include <iostream>
#include <math.h>

void chooseCentroidsAtRandomFromDataset(Rng &rng, size_t numPoints,
                                        size_t pointSize,
                                        const std::vector<double> &allData,
                                        std::vector<Point> &centroids) {
    std::vector<size_t> pointIndices(centroids.size());
    rng.pickRandomIndices(numPoints, pointIndices);

    // Loop over all random generated indices (of rows) and push the data points
    // into centroids
    for (int i = 0; i < pointIndices.size(); i++) {
        // Get x, y, etc. value for point on specific row from allData into
        // subvector.
        Point pointData(allData.begin() + pointIndices[i] * pointSize,
                        allData.begin() + pointIndices[i] * pointSize +
                            pointSize);
        centroids[i] = pointData;
    }
}

void findClosestCentroidIndexAndDistance(size_t pointIndex, size_t pointSize,
                                         const std::vector<double> &allData,
                                         const std::vector<Point> &centroids,
                                         int &newCluster, double &bestDist) {
    newCluster = -1;
    bestDist = std::numeric_limits<int>::max(); // can only get better

    // Get x, y, etc. value for point from allData into
    // subvector.
    const size_t p = pointIndex * pointSize;

    // Loop over all centroid points
    for (size_t i = 0; i < centroids.size(); i++) {
        double dist = 0;

        // Calculate quadratic euclidean distance between data point(1) and
        // centroid point(2) [ sqrt( (x1 - x2)^2 + (y1 - y2)^2 + ...) ]
        for (size_t dim = 0; dim < pointSize; dim++)
            dist += pow(allData[p + dim] - centroids[i][dim], 2);

        // Change cluster index for point when distance is smaller then current
        // best
        if (dist < bestDist) {
            newCluster = i;
            bestDist = dist;
        }
    }
}

void moveCentroidsToAverage(std::vector<Point> &centroids,
                            std::vector<int> &clusters, size_t numPoints,
                            size_t pointSize, const std::vector<double> &allData, std::vector<int>& pointCounts) {

    // reset all centroids to 0
    for (int i = 0; i < centroids.size(); ++i) {
        std::fill(centroids[i].begin(), centroids[i].end(), 0); // reset centroid to 0
    }

    // set per centroid point counters to 0
    std::fill(pointCounts.begin(), pointCounts.end(), 0);

    // Loop over all points of dataset
    for (size_t index = 0; index < numPoints; index++) {

        const int c = clusters[index];

        // add all dimensions to the centroid
        const size_t p = index * pointSize;
        for (int dim = 0; dim < pointSize; ++dim) {
            centroids[c][dim] += allData[p + dim];
        }
        pointCounts[c] += 1;
    }

    // average out the centroids
    for (int i = 0; i < centroids.size(); ++i) {
        if (pointCounts[i] > 0)
            for (size_t dim = 0; dim < pointSize; dim++)
                centroids[i][dim] /= pointCounts[i];
    }
}
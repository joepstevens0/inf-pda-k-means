#pragma once

#include "types.h"
#include <cstdlib>
#include "rng.h"

void chooseCentroidsAtRandomFromDataset(Rng& rng, size_t numPoints, size_t pointSize, const std::vector<double> &allData, std::vector<Point> &centroids);


void findClosestCentroidIndexAndDistance(size_t pointIndex, size_t pointSize, const std::vector<double> &allData, const std::vector<Point> &centroids, int &newCluster, double &bestDist);

void moveCentroidsToAverage(std::vector<Point>& centroids, std::vector<int> &clusters, size_t numPoints, size_t pointSize, const std::vector<double> &allData, std::vector<int>& pointCounts);
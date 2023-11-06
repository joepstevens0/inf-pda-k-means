#pragma once

#include "rng.h"
#include "CSVWriter.hpp"
#include "types.h"

struct KMeansArgs {
    KMeansArgs(Rng &rng, const std::string &inputFileName,
               const std::string &outputFileName, int numClusters,
               int repetitions, int numBlocks, int numThreads,
               const std::string &centroidDebugFileName = "",
               const std::string &clusterDebugFileName = "");

    Rng &rng;
    const std::string &inputFileName;
    const std::string &outputFileName;
    int numClusters;
    int repetitions;
    int numBlocks;
    int numThreads;

    // optional
    const std::string &centroidDebugFileName;
    const std::string &clusterDebugFileName;
};

int kmeans(KMeansArgs args);

struct KMeansIn{
    int repetitions;
    Rng& rng;
    int numClusters;
    int numBlocks;
    int numThreads;
    size_t numPoints;
    size_t pointSize;
    std::vector<double> allData;
    FileCSVWriter& centroidDebugFile;
    FileCSVWriter& clustersDebugFile;
};
struct KmeansOut
{
    double bestDistSquaredSum;
    std::vector<int> bestClusters;
    std::vector<int> stepsPerRepetition;
};

KmeansOut kmeansSerial(KMeansIn input);
KmeansOut kmeansOpenMP(KMeansIn input);
KmeansOut kmeansCUDA(KMeansIn input);
KmeansOut kmeansMPI(KMeansIn input, int rank, int totalUsedCores, int totalCores);
#include "kmeans.h"
#include "CSVReader.hpp"
#include "CSVWriter.hpp"
#include "helper_functions.h"
#include "timer.h"

#if KMEANS_MODE_MPI == 1
#include <mpi.h>
#endif

KMeansArgs::KMeansArgs(Rng &rng, const std::string &inputFileName,
                       const std::string &outputFileName, int numClusters,
                       int repetitions, int numBlocks, int numThreads,
                       const std::string &centroidDebugFileName,
                       const std::string &clusterDebugFileName)
    : rng{rng}, inputFileName{inputFileName}, outputFileName{outputFileName},
      numClusters{numClusters}, repetitions{repetitions}, numBlocks{numBlocks},
      numThreads{numThreads}, centroidDebugFileName{centroidDebugFileName},
      clusterDebugFileName{clusterDebugFileName} {}

// Helper function to read input file into allData, setting number of detected
// rows and columns. Feel free to use, adapt or ignore
void readData(std::ifstream &input, std::vector<double> &allData,
              size_t &numRows, size_t &numCols) {
    if (!input.is_open())
        throw std::runtime_error("Input file is not open");

    allData.resize(0);
    numRows = 0;
    numCols = -1;

    CSVReader inReader(input);
    int numColsExpected = -1;
    int line = 1;
    std::vector<double> row;

    while (inReader.read(row)) {
        if (numColsExpected == -1) {
            numColsExpected = row.size();
            if (numColsExpected <= 0)
                throw std::runtime_error("Unexpected error: 0 columns");
        } else if (numColsExpected != (int)row.size())
            throw std::runtime_error(
                "Incompatible number of colums read in line " +
                std::to_string(line) + ": expecting " +
                std::to_string(numColsExpected) + " but got " +
                std::to_string(row.size()));

        for (auto x : row)
            allData.push_back(x);

        line++;
    }

    numRows = (size_t)allData.size() / numColsExpected;
    numCols = (size_t)numColsExpected;
}

FileCSVWriter openDebugFile(const std::string &n) {
    FileCSVWriter f;

    if (n.length() != 0) {
        f.open(n);
        if (!f.is_open())
            std::cerr << "WARNING: Unable to open debug file " << n
                      << std::endl;
    }
    return f;
}

void printOutput(KMeansArgs args, KmeansOut output, Timer timer) {
    // Some example output, of course you can log your timing data anyway you
    // like.
    std::cerr << "# "
                 "Type,blocks,threads,file,seed,clusters,repetitions,"
                 "bestdistsquared,timeinseconds"
              << std::endl;
    std::cout << "sequential," << args.numBlocks << "," << args.numThreads
              << "," << args.inputFileName << "," << args.rng.getUsedSeed()
              << "," << args.numClusters << "," << args.repetitions << ","
              << output.bestDistSquaredSum << ","
              << timer.durationNanoSeconds() / 1e9 << std::endl;
}

int kmeans(KMeansArgs args) {
    // If debug filenames are specified, this opens them. The is_open method
    // can be used to check if they are actually open and should be written to.
    FileCSVWriter centroidDebugFile = openDebugFile(args.centroidDebugFileName);
    FileCSVWriter clustersDebugFile = openDebugFile(args.clusterDebugFileName);

    FileCSVWriter csvOutputFile(args.outputFileName);
    if (!csvOutputFile.is_open()) {
        std::cerr << "Unable to open output file " << args.outputFileName
                  << std::endl;
        return -1;
    }

    // load dataset
    size_t numPoints;
    size_t pointSize;
    std::vector<double> allData;

    std::ifstream inputFile;
    inputFile.open(args.inputFileName);
    readData(inputFile, allData, numPoints, pointSize);
    inputFile.close();

    // start the timer
    Timer timer;

    // call the correct kmeans algorithm
    KmeansOut output;
    #if KMEANS_MODE_OPENMP == 1
        output = kmeansOpenMP({args.repetitions, args.rng, args.numClusters,
                               args.numBlocks, args.numThreads,
                               numPoints, pointSize, allData, centroidDebugFile,
                               clustersDebugFile});
    #elif KMEANS_MODE_CUDA == 1
        output = kmeansCUDA({args.repetitions, args.rng, args.numClusters,
                            args.numBlocks, args.numThreads,
                            numPoints, pointSize, allData, centroidDebugFile,
                            clustersDebugFile});
    #elif KMEANS_MODE_MPI == 1
        int rank, totalUsedCores, totalCores, len;
        char name[MPI_MAX_PROCESSOR_NAME+1];
        int argc = 0; char **argv = nullptr;
        MPI_Init(&argc, &argv);
        totalUsedCores = args.numThreads;
        MPI_Comm_size(MPI_COMM_WORLD, &totalCores);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Get_processor_name(name, &len);

        output = kmeansMPI({args.repetitions, args.rng, args.numClusters,
                            args.numBlocks, args.numThreads,
                            numPoints, pointSize, allData, centroidDebugFile,
                            clustersDebugFile}, rank, totalUsedCores, totalCores);
    #else
        output = kmeansSerial({args.repetitions, args.rng, args.numClusters,
                               args.numBlocks, args.numThreads,
                               numPoints, pointSize, allData, centroidDebugFile,
                               clustersDebugFile});
    #endif

    #if KMEANS_MODE_MPI == 1
    if (rank == 0){
    #endif

    timer.stop();

    // print the results to std::cout
    printOutput(args, output, timer);

    // Write the number of steps per repetition, kind of a signature of the
    // work involved
    csvOutputFile.write(output.stepsPerRepetition, "# Steps: ");
    // Write best clusters to csvOutputFile, something like
    csvOutputFile.write(output.bestClusters);

    #if KMEANS_MODE_MPI == 1
    }
    MPI_Finalize();
    #endif
    return 0;
}
# -std=c++14: we're limiting ourselves to c++14, since that's what the 
#             GCC compiler on the VSC supports.
# -DNDEBUG: turns off e.g. assertion checks
# -O3: enables optimizations in the compiler

# Settings for optimized build
FLAGS=-O3 -DNDEBUG -std=c++14

# Settings for a debug build
#FLAGS=-g -std=c++14

# Set OpenMP parallel nesting true
export OMP_NESTED=False

all: kmeans_openmp

clean:
	rm -f kmeans
	rm -f kmeans_cuda
	rm -f kmeans_serial
	rm -f kmeans_openmp
	rm -f output/*

kmeans_mpi: main_startcode.cpp *.cpp src_kmeans/*.cpp util/*.cpp
	mpicxx $(FLAGS) -DKMEANS_MODE_MPI=1 -o kmeans_mpi $^ -I util

kmeans_serial: main_startcode.cpp *.cpp src_kmeans/*.cpp util/*.cpp
	$(CXX) $(FLAGS) -o kmeans_serial $^ -I util

kmeans_openmp: main_startcode.cpp *.cpp src_kmeans/*.cpp util/*.cpp
	$(CXX) $(FLAGS) -DKMEANS_MODE_OPENMP=1 -o kmeans_openmp $^ -I util -fopenmp

kmeans_cuda: main_startcode.cpp *.cpp src_kmeans/*.cpp util/*.cpp src_kmeans/*.cu
	nvcc $(FLAGS) -gencode arch=compute_37,code=sm_37 -DKMEANS_MODE_CUDA=1 -o kmeans_cuda $^ -I util


run_test_mpi: kmeans_mpi
	EXECUTABLE=./kmeans_mpi ./mpiwrapper.sh --input input/mouse_500x2.csv --output output/output.csv --k 3 --repetitions 10 --seed 1848586 --threads 4

run_test_cuda: kmeans_cuda
	./kmeans_cuda --input input/mouse_500x2.csv --output output/output.csv \
	--k 3 --repetitions 5 --seed 1848586 --threads 32 --blocks 1

run_test_openmp: kmeans_openmp
	./kmeans_openmp.exe --input input/1M_1000000x4.csv --output output/output.csv \
	--k 3 --repetitions 100 --seed 1848586 --threads 8


run_compare_mpi: kmeans_mpi kmeans_serial
	EXECUTABLE=./kmeans_mpi ./mpiwrapper.sh --input input/1M_1000000x4.csv --output output/output.csv.1 --k 3 --repetitions 100 --seed 1848586 --threads 4
	./kmeans_serial --input input/1M_1000000x4.csv --output output/output.csv.2 --k 3 --repetitions 100 --seed 1848586 --threads 4
	python3 help_scripts/compare.py SKIP SKIP --input input/1M_1000000x4.csv --output output/output.csv --k 3 --repetitions 100 --seed 1848586 --threads 4

run_compare_cuda: kmeans_cuda kmeans_serial
	mkdir -p output
	python3 help_scripts/compare.py ./kmeans_cuda ./kmeans_serial \
	--input input/mouse_500x2.csv --output output/output.csv --k 3 --repetitions 100 --seed 1848586 --threads 256 --blocks 5

run_compare_openmp: kmeans_openmp kmeans_serial
	mkdir -p output
	python3 help_scripts/compare.py ./kmeans_openmp ./kmeans_serial \
	--input input/1M_1000000x4.csv --output output/output.csv --k 3 --repetitions 150 --seed 1848586 --threads 4

run_compare_openmp_win: kmeans_openmp kmeans_serial
	mkdir -p output
	python3 help_scripts/compare.py kmeans_openmp.exe kmeans_serial.exe \
	--input input/1M_1000000x4.csv --output output/output.csv --k 3 --repetitions 100 --seed 1848586 --threads 8


run_visualize_clusters:
	python3 help_scripts/visualize_clusters.py input/1M_1000000x4.csv output/output.csv.1 0 1

run_test_mouse: kmeans
	mkdir -p output
	./kmeans --input input/mouse_500x2.csv --output output/output.csv \
	--k 3 --repetitions 10 --seed 1848586
	python3 help_scripts/visualize_clusters.py input/mouse_500x2.csv output/output.csv 0 1
run_test_mouse_compare: kmeans kmeans_serial
	mkdir -p output
	python3 help_scripts/compare.py ./kmeans ./kmeans_serial \
	--input input/mouse_500x2.csv --output output/output.csv --k 3 --repetitions 10 --seed 1848586
run_test_mouse_debug_centroid: kmeans
	mkdir -p output
	./kmeans --input input/mouse_500x2.csv --output output/output.csv \
	--k 3 --repetitions 10 --seed 1848586 \
	--centroidtrace output/centroidtrace.csv
	python3 help_scripts/visualize_centroids.py input/mouse_500x2.csv output/centroidtrace.csv 3 0 1

run_test_mouse_debug_clusters: kmeans
	mkdir -p output
	./kmeans --input input/mouse_500x2.csv --output output/output.csv \
	--k 3 --repetitions 10 --seed 1848586 \
	--trace output/clustertrace.csv
	python3 help_scripts/visualize_clusters.py input/mouse_500x2.csv output/clustertrace.csv 0 1

import subprocess
import sys

if len(sys.argv) < 2:
    print("usage:\n core_difference_multiple_nodes.py min_cores max_cores iterations_per_core mpirun kmeans_executable --input input/1M_1000000x4.csv --output output/output.csv --k 3 --repetitions 100 --seed 1848586")
    sys.exit(-1)

min_cores = int(sys.argv[1])
max_cores = int(sys.argv[2])
total_iterations = int(sys.argv[3])
cmd = sys.argv[4:]
cmd.append("--threads")

filename = "output/core_dif" + str(max_cores) + ".csv"
f = open(filename, "w")

# write labels
f.write("\"cores\",\"time (sec)\"\n")

for cores in range(min_cores, max_cores + 1):
    # write total cores
    f.write(str(cores))

    # calculate time for amount of cores <iterations_per_core> times
    for it in range(0,total_iterations):
        try:
            byteOutput = subprocess.check_output(cmd + [str(cores)])
            output = byteOutput.decode('UTF-8')
            time = float(output.split("\n")[-2].split(",")[-1])
            print("Cores:", cores, ", Iteration: ", it , ", Time:", str(time))
            f.write(",")
            f.write(str(time))
        except subprocess.CalledProcessError as e:
            print("Error in", cmd ,":\n", e.output)
    f.write("\n")
        
f.close()
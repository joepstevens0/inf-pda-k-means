import subprocess
import sys

if len(sys.argv) < 2:
    print("usage:\n thread_difference.py max_threads iterations_per_thread kmeans_executable --input input/1M_1000000x4.csv --output output/output.csv --k 3 --repetitions 100 --seed 1848586")
    sys.exit(-1)

max_threads = int(sys.argv[1])
total_iterations = int(sys.argv[2])
cmd = sys.argv[3:]
cmd.append("--threads")

f = open("output/thread_dif.csv", "w")

# write labels
f.write("\"threads\",\"time (sec)\"\n")

for threads in range(1,max_threads + 1):
    # write total threads
    f.write(str(threads))

    # calculate time for amount of threads 5 times
    for it in range(0,total_iterations):
        try:
            byteOutput = subprocess.check_output(cmd + [str(threads)])
            output = byteOutput.decode('UTF-8')
            time = float(output.split("\n")[-2].split(",")[-1])
            print("Threads:", threads, ", Iteration: ", it , ", Time:", str(time))
            f.write(",")
            f.write(str(time))
        except subprocess.CalledProcessError as e:
            print("Error in", cmd ,":\n", e.output)
    f.write("\n")
        
f.close()
import subprocess
import sys

if len(sys.argv) < 2:
    print("usage:\n thread_difference.py max_blocks iterations_per_block kmeans_executable --input input/1M_1000000x4.csv --output output/output.csv --k 3 --repetitions 100 --seed 1848586 --threads 512")
    sys.exit(-1)

max_blocks = int(sys.argv[1])
total_iterations = int(sys.argv[2])
cmd = sys.argv[3:]
cmd.append("--blocks")

f = open("output/block_dif.csv", "w")

# write labels
f.write("\"blocks\",\"time (sec)\"\n")

for blocks in range(1,max_blocks + 1):
    # write total blocks
    f.write(str(blocks))

    # calculate time for amount of blocks 'total_iterations' times
    for it in range(0,total_iterations):
        try:
            byteOutput = subprocess.check_output(cmd + [str(blocks)])
            output = byteOutput.decode('UTF-8')
            time = float(output.split("\n")[-2].split(",")[-1])
            print("Blocks:", blocks, ", Iteration: ", it , ", Time:", str(time))
            f.write(",")
            f.write(str(time))
        except subprocess.CalledProcessError as e:
            print("Error in", cmd ,":\n", e.output)
    f.write("\n")
        
f.close()
import numpy as np
from matplotlib import pyplot as plt
import csv

threads = []
allTimes = []

xlabel = ""
ylabel = ""

# READ CSV RESULTS
with open('results.csv', 'r') as csvfile:
     mat = csv.reader(csvfile, delimiter=',')

     firstline = True
     for row in mat:
       if not firstline:
         threads.append(int(row[0]))

         times = [float(i) for i in row[1:]]
         allTimes.append(times)
       else:
          firstline = False
          xlabel = row[0]
          ylabel = row[1]

# CALCULATE AVERAGE SERIAL TIME
avgSerial = 0
stdSerial = 0
serial_times = []

with open('serial_time.csv', 'r') as csvfile:
  mat = csv.reader(csvfile, delimiter=',')

  row = next(mat)
  for column in row:
    serial_times.append(float(column))

  avgSerial = np.average(serial_times, 0)
  stdSerial = np.std(serial_times, 0)

print("Average serial time:", avgSerial)

# PLOT AVERAGE TIMES PER AMOUNT OF THREADS
avgTimes = []
stdTimes = []

for times in allTimes:
  avgTime = np.average(times,0)
  stdTime = np.std(times, 0)
  avgTimes.append(avgTime)
  stdTimes.append(stdTime)

plt.errorbar(threads,avgTimes, stdTimes, label='OpenMP', marker='.', capsize=5)
plt.errorbar(threads,np.repeat(avgSerial, len(threads)), np.repeat(stdSerial, len(threads)), label='Serial', marker='.', capsize=5)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title('Time needed for differend thread amounts')
plt.legend()
plt.xticks(np.arange(1,len(threads) + 1,2))
plt.show()

# PLOT SPEEDUP
avgSpeedups = []
stdSpeedups = []

for times in allTimes:
  speedups = np.divide(avgSerial, times)
  avgSpeedup = np.average(speedups,0)
  stdSpeedup = np.std(speedups, 0)
  avgSpeedups.append(avgSpeedup)
  stdSpeedups.append(stdSpeedup)

plt.errorbar(threads,avgSpeedups, stdSpeedups, label='OpenMP', marker='.', capsize=5)
plt.errorbar(threads, threads, label='Max speedup')
plt.xlabel(xlabel)
plt.ylabel("Speedup")
plt.title('Speedup per amount of threads')
plt.legend()
plt.xticks(np.arange(1,len(threads) + 1,2))
plt.show()

# PLOT EFFICIENCY
avgEfficiencies = []
stdEfficiencies = []

for times in allTimes:
  speedups = np.divide(avgSerial, times)

  thread = threads[allTimes.index(times)]
  efficiencies = np.divide(speedups, thread)
  avgEff = np.average(efficiencies,0)
  stdEff = np.std(efficiencies, 0)
  avgEfficiencies.append(avgEff)
  stdEfficiencies.append(stdEff)

plt.errorbar(threads, avgEfficiencies, stdEfficiencies, label='OpenMP', marker='.', capsize=5)
plt.errorbar(threads, np.full(len(threads), 1), label='Max efficiency')
plt.xlabel(xlabel)
plt.ylabel("Efficiency")
plt.title('Efficiency per amount of threads')
plt.legend()
plt.xticks(np.arange(1,len(threads) + 1,2))
plt.show()

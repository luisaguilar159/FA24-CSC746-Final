"""

E. Wes Bethel, Copyright (C) 2022

October 2022

Description: This code loads a .csv file and creates a 3-variable plot

Inputs: the named file "openmp_small_data.csv"

Outputs: displays a chart with matplotlib

Dependencies: matplotlib, pandas modules

Assumptions: developed and tested using Python version 3.8.8 on macOS 11.6

"""
import pandas as pd
import matplotlib.pyplot as plt

# Initialize variables for plot
plot_title = "OpenMP-parallel Euclidean distance | Small dataset"
plot_y_label = "Speedup"
plot_x_label = "Split/Problem Size"
fname = "openmp_small_data.csv"

df = pd.read_csv(fname, comment="#")
print(df)

var_names = list(df.columns)

print("var names =", var_names)

# split the df into individual vars
# assumption: column order - 0=problem size, 1=blas time, 2=basic time

problem_sizes = df[var_names[0]].values.tolist()
serial_time = df[var_names[1]].values.tolist()
concurrency1_time = df[var_names[1]].values.tolist()
concurrency4_time = df[var_names[2]].values.tolist()
concurrency16_time = df[var_names[3]].values.tolist()
concurrency64_time = df[var_names[4]].values.tolist()

plt.title(plot_title)

xlocs = [i for i in range(len(problem_sizes))]

plt.xticks(xlocs, problem_sizes)

# here, we are plotting the raw values read from the input .csv file, which
# we interpret as being "time" that maps directly to the y-axis.
#
# what if we want to plot MFLOPS instead? How do we compute MFLOPS from
# time and problem size? You may need to add some code here to compute
# MFLOPS, then modify the plt.plot() lines below to plot MFLOPS rather than time.


# Speedup = Serial time / Parallel time
for i in range(0, len(problem_sizes)):
    concurrency1_time[i] = serial_time[i]/concurrency1_time[i]
    concurrency4_time[i] = serial_time[i]/concurrency4_time[i]
    concurrency16_time[i] = serial_time[i]/concurrency16_time[i]
    concurrency64_time[i] = serial_time[i]/concurrency64_time[i]


plt.plot(concurrency1_time, "r-o") # concurrency1_time: List[double], red dot icon
plt.plot(concurrency4_time, "g-s") # concurrency4_time: List[double], green square
plt.plot(concurrency16_time, "b-x") # concurrency16_time: List[double], blue x icon
plt.plot(concurrency64_time, "m-d") # concurrency16_time: List[double], magenta diamond

#plt.xscale("log")
#plt.yscale("log")

plt.xlabel(plot_x_label)
plt.ylabel(plot_y_label)

varNames = [var_names[1], var_names[2], var_names[3], var_names[4]]
plt.legend(varNames, loc="best")

plt.grid(axis='both')

plt.show()

# EOF
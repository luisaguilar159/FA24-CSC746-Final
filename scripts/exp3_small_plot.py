"""
E. Wes Bethel, Copyright (C) 2022
October 2022
Description: This code loads a .csv file and creates a 3-variable plot
Inputs: the named file "exp3_small_data.csv"
Outputs: displays a chart with matplotlib
Dependencies: matplotlib, pandas modules
Assumptions: developed and tested using Python version 3.8.8 on macOS 11.6
"""
import pandas as pd
import matplotlib.pyplot as plt

# Initialize variables for plot
plot_title = "Serial vs Parallel vs Vendor | Euclidean distance | Small dataset"
plot_y_label = "Elapsed time (ms)"
plot_x_label = "Split/Problem Size"
fname = "exp3_small_data.csv"

df = pd.read_csv(fname, comment="#")
print(df)

var_names = list(df.columns)

print("var names =", var_names)

# split the df into individual vars
# assumption: column order - 0=problem size, 1=serial time, 2=parallel time, 3=vendor time

problem_sizes = df[var_names[0]].values.tolist()
serial_time = df[var_names[1]].values.tolist()
parallel_time = df[var_names[2]].values.tolist()
vendor_time = df[var_names[3]].values.tolist()

plt.title(plot_title)

xlocs = [i for i in range(len(problem_sizes))]

plt.xticks(xlocs, problem_sizes)

plt.plot(serial_time, "r-o") # serial_time: List[double], red dot icon
plt.plot(parallel_time, "g-s") # parallel_time: List[double], green square
plt.plot(vendor_time, "b-x") # vendor_time: List[double], blue x icon
#plt.plot(concurrency64_time, "m-d") # concurrency16_time: List[double], magenta diamond

#plt.xscale("log")
#plt.yscale("log")

plt.xlabel(plot_x_label)
plt.ylabel(plot_y_label)

varNames = [var_names[1], var_names[2], var_names[3]]
plt.legend(varNames, loc="best")

plt.grid(axis='both')

plt.show()

# EOF
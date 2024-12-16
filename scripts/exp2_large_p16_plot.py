"""
E. Wes Bethel, Copyright (C) 2022
October 2022
Description: This code loads a .csv file and creates a 3-variable plot
Inputs: the named file "exp3_large_data.csv"
Outputs: displays a chart with matplotlib
Dependencies: matplotlib, pandas modules
Assumptions: developed and tested using Python version 3.8.8 on macOS 11.6
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Initialize variables for plot
plot_title = "Parallel Load Balancing | Large dataset | P=16"
plot_y_label = "Elapsed time (ms)"
plot_x_label = "Threads (T1 to T16)"
fname = "exp2_large_p16_data.csv"

df = pd.read_csv(fname, comment="#")
print(df)
var_names = list(df.columns)
print("var names =", var_names)

# split the df into individual vars
# assumption: column order - 0=problem size, 1=serial time, 2=parallel time, 3=vendor time

threads = df[var_names[0]].values.tolist()
runtimes = df[var_names[1]].values.tolist()
plt.bar(threads, runtimes, label=threads)

# for i in range(0, len(threads)):
#     plt.text(i, runtimes[i]+1,runtimes[i], ha = 'center',color='black', fontsize=5)

plt.title(plot_title)
plt.xlabel(plot_x_label)
plt.ylabel(plot_y_label)
plt.ylim(top = max(runtimes)+3)
#plt.grid(axis='y')
plt.show()


# EOF
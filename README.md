# Final project for CSC746-HPC

- Title: Parallelization of the K-nearest neighbors (KNN) algorithm using OpenMP
- Author: Luis Aguilar
- Term: Fall 2024

This directory contains a code harness to run a parallelized implementation of the K-nearest neighbors (KNN) algorithm.

The main code is knn.cpp, which contains the development workflow to run the KNN algorithm at varying split sizes (90%, 80%, 70%, 60%, and 50%) and over two versions of the Iris dataset (original and extended).

# Getting started



# Build instructions for Perlmutter

After logging in to Perlmutter, first set up your environment by typing this command:
```
module load cpu
```

Then, build the code. First, cd into the main source directory (vmmul-omp-harness-instructional) and then enter the following commands:
```
mkdir build  
cd build  
cmake ../  
make
```

# Running the codes on Perlmutter

After building the codes, use the salloc command to access to a Perlmutter CPU node:
```
salloc --nodes=1 --qos=interactive --time=00:15:00 --constraint=cpu --account=m3930
```

# Run the OpenMP code at 4 levels of concurrency (1, 4, 16, 64 )

From the build directory, on an interactive CPU node, you'll manually update the property to set the number of threads for OpenMP.

1. Run the KNN algorithm at P=1
```
export OMP_NUM_THREADS=1
./knn-openmp
```
2. Run the KNN algorithm at P=4
```
export OMP_NUM_THREADS=4
./knn-openmp
```
3. Run the KNN algorithm at P=16
```
export OMP_NUM_THREADS=16
./knn-openmp
```
4. Run the KNN algorithm at P=64
```
export OMP_NUM_THREADS=64
./knn-openmp
```
# Run the vendor package, mlpack in a local machine

## Prerequisites

- Install Homebrew in your local machine

## Installation

Using Homebrew, install mlpack with the following command:
```
brew install mlpack
```

Once installed, you'll be able to access the `mlpack_knn` command.

To run the KNN method, you can go to the `data/mlpack_data` directory to run the `mlpack_knn` command, since we have some data already in place.

For example, the below command executes the KNN algorithm by sending the `input/iris09.csv` data set, and storing the neighbors and their distances in `output/neighbors09.csv` and `output/distances09.csv`, respectively. Additionally, this command represent the KNN method at a split size of 90% of the original Iris data set at K=5.
```
mlpack_knn --k 5 --reference_file input/iris09.csv --neighbors_file output/neighbors09.csv --distances_file output/distances09.csv --verbose
```

The example output of the KNN method of mlpack can be found in the `data/mlpack_data/Output_mlpack_KNN_algorithm.txt` file in the `data/mlpack_data`. This file has the tests that we ran to retrieve the runtime of the vendor implementation.
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <map>
#include <chrono>
#include <cstring>
#include <omp.h>
#include "knn.hpp" // For DataPoint class

vector<DataPoint> readDataset(const string &path) 
{
    vector<DataPoint> data;
    ifstream inputFileDataset(path);
    string data_line;
    if (inputFileDataset.is_open()) {
        while (getline(inputFileDataset, data_line)) {
            stringstream input_line(data_line);
            string segment;
            DataPoint dp;
            int idx = 0;
            while (getline(input_line, segment, ',')) {
                // Add class name to DataPoint instance
                if (idx == dp.size) {
                    dp.class_name = segment;
                    break;
                }
                // Convert string to float to store in vector
                dp.fields.push_back(stof(segment));
                idx++;
            }
            data.push_back(dp);
        }
        inputFileDataset.close();
    } else {
        cerr << "Unable to open file\n";
    }
    return data;
}

void shuffleDataset(vector<DataPoint> &inputDataset)
{
    random_device rd;
    mt19937 g(rd());
    shuffle(inputDataset.begin(), inputDataset.end(), g);
}

pair<vector<DataPoint>, vector<DataPoint>> splitDataset(vector<DataPoint> inputDataset, float inputPercentage)
{
    vector<DataPoint> training_set;
    vector<DataPoint> test_set;
    float calcPercentage = inputDataset.size() * inputPercentage;
    size_t idx = 0;
    while (idx < calcPercentage)
    {
        training_set.push_back(inputDataset[idx]);
        idx++;
    }
    while (idx < inputDataset.size())
    {
        test_set.push_back(inputDataset[idx]);
        idx++;
    }
    return make_pair(training_set, test_set);
}

float calculateEuclideanDistance(const DataPoint &testItem, const DataPoint trainDataItem)
{
    float distance = 0.0;
    for (int i = 0; i < 4; i++)
    {
        float diff = trainDataItem.fields[i] - testItem.fields[i];
        distance += diff * diff;
    }
    return sqrtf(distance);
}

void updateTopKIndicesDistances(int testDpId, float testDpCalcDistance, vector<int> &topKIndices, vector<float> &topKDistances, int k)
{
    // Set the top k distances and their indices for the given test data point
    int idx = 0;
    // This loop will iterate at most K times
    // The "idx" will determine the position of the calculated "distance" in the "topKDistances" vector
    while (idx < k && topKIndices[idx] != -1 && topKDistances[idx] < testDpCalcDistance)
    {
        idx++;
    }
    if (idx < k)
    {
        int shift_idx = k - 1;
        // This loop will iterate at most K-1 times
        // This will move the values greater than the calculated "distance" to the end of the vector
        while (shift_idx > idx)
        {
            topKIndices[shift_idx] = topKIndices[shift_idx - 1];
            topKDistances[shift_idx] = topKDistances[shift_idx - 1];
            shift_idx--;
        }
        topKIndices[idx] = testDpId;
        topKDistances[idx] = testDpCalcDistance;
    }
}

pair<vector<int>, vector<float>> getTopKNeighbours(const float* lstDistances, int trainDataLength, int k)
{
    vector<float> topKDistances(k);
    vector<int> topKIndices(k);
    for (int i = 0; i < k; i++)
    {
        topKIndices[i] = -1;
    }
    // Get the distance bw given data point and train dataset
    for (int i = 0; i < trainDataLength; i++)
    {
        // Update the "topKIndices" and "topKDistances" vectors if the calculated "distance"
        // fits the top K ranking of distances
        updateTopKIndicesDistances(i, lstDistances[i], topKIndices, topKDistances, k);
    }
    return make_pair(topKIndices, topKDistances);
}

void getAllEuclideanDistances(vector<float> &lstDistances, vector<DataPoint> &testData, int testDataLength, const vector<DataPoint> &trainData, int trainDataLength, vector<double> &threads_time)
{
    #pragma omp parallel
    {
        double omp_th_start = omp_get_wtime();
        #pragma omp for
        for (int i = 0; i < testDataLength; i++)
        {
            for (int j = 0; j < trainDataLength; j++)
            {
                lstDistances[i * trainDataLength + j] = calculateEuclideanDistance(testData[i], trainData[j]);
            }
        }
        double omp_th_stop = omp_get_wtime();
        // Add elapsed time of each thread to check load balancing
        threads_time[omp_get_thread_num()] = omp_th_stop - omp_th_start;
    }
}

void run_knn_omp(vector<DataPoint> &trainSet, int trainSetLength, vector<DataPoint> &testSet, int testSetLength, int k, chrono::duration<double> &euclidean_time, vector<double> &threads_time) 
{
    vector<float> lstDistances(testSetLength*trainSetLength);
    chrono::time_point<chrono::high_resolution_clock> euclidean_start_time = chrono::high_resolution_clock::now();
    getAllEuclideanDistances(lstDistances, testSet, testSetLength, trainSet, trainSetLength, threads_time);
    chrono::time_point<chrono::high_resolution_clock> euclidean_end_time = chrono::high_resolution_clock::now();
    euclidean_time = euclidean_end_time - euclidean_start_time;
    for (int i = 0; i < testSetLength; i++) 
    {
        float* bufDistances = new float[trainSetLength];
        size_t idxDistVector = i * trainSetLength;
        memcpy(bufDistances, lstDistances.data() + idxDistVector, sizeof(float) * trainSetLength);
        testSet[i].predicted_classes = getTopKNeighbours(bufDistances, trainSetLength, k);
        delete[] bufDistances;
    }
}

string getPredictedClass(vector<int> topKIndices, vector<float> topKDistances, vector<DataPoint> trainData, int k)
{
    int index = 0;
    map<string, int> topKClassMap;
    while (index < k && topKIndices[index] != -1)
    {
        topKClassMap[trainData[topKIndices[index]].class_name]++;
        index++;
    }
    int max = -1;
    string maxClassName;
    for (auto pair : topKClassMap)
    {
        if (pair.second > max)
        {
            max = pair.second;
            maxClassName = pair.first;
        }
    }
    return maxClassName;
}

void get_knn_accuracy(vector<DataPoint> &trainSet, vector<DataPoint> &inTestSet, int k)
{
    int numCorrectPredictions = 0;
    // Compare prediction with predicted_class in Test set
    for (size_t i = 0; i < inTestSet.size(); i++)
    {
        string predictedClass = getPredictedClass(inTestSet[i].predicted_classes.first, inTestSet[i].predicted_classes.second, trainSet, k);
        if (predictedClass.compare(inTestSet[i].class_name) == 0)
        {
            numCorrectPredictions++;
        }
    }
    // Show accuracy
    cout << numCorrectPredictions << " of " << inTestSet.size() << 
    " (" << static_cast<float>(100.0) *numCorrectPredictions/static_cast<float>(inTestSet.size()) << ")" 
    << endl << endl;
}

void printDataset(vector<DataPoint> &inputData)
{
    for (size_t i = 0; i < inputData.size(); i++) {
        cout << "[" << i << "] Class: " << inputData[i].class_name << " | Fields: ";
        for (size_t j = 0; j < inputData[i].fields.size(); j++) {
            cout << inputData[i].fields[j] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char* argv[])
{
    vector<string> data_sources{"../data/iris.data", "../data/iris_extended.data"};
    //vector<int> k_sizes{5,7,9};
    vector<int> k_sizes{5};
    vector<double> split_sizes{0.9, 0.8, 0.7, 0.6, 0.5};
    //vector<double> split_sizes{0.5, 0.6};
    
    for (string data_path : data_sources)
    {
        for (int k : k_sizes)
        {
            for (double split_size : split_sizes)
            {
                printf("Working on KNN | dataset: %s | k: %d | split: %.2f\n", data_path.c_str(), k, split_size);
                // 1. Read iris dataset
                // --------------------------------
                vector<DataPoint> dataset = readDataset(data_path);
                cout << "Dataset size: " << dataset.size() << endl;
                //printDataset(dataset);
                // 2. Shuffle iris dataset
                // --------------------------------
                shuffleDataset(dataset);
                //printDataset(dataset);
                // 3. Split data into training and testing set
                // --------------------------------
                pair<vector<DataPoint>, vector<DataPoint>> splitted_set = splitDataset(dataset, split_size);
                dataset.clear();
                printf("Training set size: %ld | Test set size: %ld\n", splitted_set.first.size(), splitted_set.second.size());
                // 4. Execute KNN logic (Here we'll measure the elapsed time)
                // --------------------------------
                vector<DataPoint> trainData = splitted_set.first;
                vector<DataPoint> testData = splitted_set.second;
                splitted_set.first.clear();
                splitted_set.second.clear();
                chrono::duration<double> euclidean_elapsed_time;
                // Vector to store the elapsed time of all threads
                vector<double> threads_elapsed_time(64);
                chrono::time_point<chrono::high_resolution_clock> start_time = chrono::high_resolution_clock::now();
                run_knn_omp(trainData, trainData.size(), testData, testData.size(), k, euclidean_elapsed_time, threads_elapsed_time);
                // To print the elapsed time of all the threads
                for (int i = 0; i < 64; i++) {
                    if (threads_elapsed_time[i] != 0.000000) {
                        printf("T[%d]: %lf (secs)\n", i, threads_elapsed_time[i]);
                    }
                }
                chrono::time_point<chrono::high_resolution_clock> end_time = chrono::high_resolution_clock::now();
                chrono::duration<double> elapsed_time = end_time - start_time;
                printf("Elapsed time for KNN is: %lf (sec) or %6.4f (millisec) or %6.4f (microsec) or %6.4f (nanosec) \n", elapsed_time.count(), elapsed_time.count()*1000.0, elapsed_time.count()*1000000, elapsed_time.count()*1000000000);
                printf("Elapsed time for Euclidean calc is: %lf (sec) or %6.4f (millisec) or %6.4f (microsec) or %6.4f (nanosec) \n", euclidean_elapsed_time.count(), euclidean_elapsed_time.count()*1000.0, euclidean_elapsed_time.count()*1000000, euclidean_elapsed_time.count()*1000000000);
                // 5. Measure accuracy of input K
                // --------------------------------
                get_knn_accuracy(trainData, testData, k);
                trainData.clear();
                testData.clear();
            }
        }
    }

    return 0;
}
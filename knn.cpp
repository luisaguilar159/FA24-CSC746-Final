#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <map>
#include <chrono>
#include <omp.h>
#include "knn.hpp" // For Data class

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
            // float *fields = new float[4];
            // dp.fields = fields;
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

float calculateEuclideanDistance(const DataPoint &testItem, const DataPoint &trainDataItem)
{
    float distance = 0.0;
    //double tmp_total = 0.0;
    for (int i = 0; i < trainDataItem.size; i++)
    {
        float diff = trainDataItem.fields[i] - testItem.fields[i];
        distance += diff * diff;
        //tmp_total += (pow(trainDataItem.fields[i] - testItem.fields[i], 2));
    }
    //distance = sqrt(tmp_total);
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

pair<vector<int>, vector<float>> getTopKNeighbours(DataPoint &testDp, const vector<DataPoint> &trainData, int trainDataLength, int k) {
    // Vector to store the highest K distances
    vector<float> topKDistances(k);
    // Vector to store the indices of the highest K distances.
    vector<int> topKIndices(k);
    // Fill the array with -1
    for (int i = 0; i < k; i++)
    {
        topKIndices[i] = -1;
    }
    // Get the distance bw given data point and train dataset
    for (int i = 0; i < trainDataLength; i++)
    {
        float distance = 0.0;
        distance = calculateEuclideanDistance(testDp, trainData[i]);
        // Update the "topKIndices" and "topKDistances" vectors if the calculated "distance" fits the top K ranking of distances
        updateTopKIndicesDistances(i, distance, topKIndices, topKDistances, k);
    }
    return make_pair(topKIndices, topKDistances);
}

void run_knn(vector<DataPoint> &trainSet, int trainSetLength, vector<DataPoint> &testSet, int testSetLength, int k)
{
    // Test every Data point in the test dataset, and predict its class
    for (int i = 0; i < testSetLength; i++)
    {
        testSet[i].predicted_classes = getTopKNeighbours(testSet[i], trainSet, trainSetLength, k);
    }
}

void run_knn_omp(vector<DataPoint> &trainSet, int trainSetLength, vector<DataPoint> &testSet, int testSetLength, int k)
{
    // Test every Data point in the test dataset, and predict its class
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < testSetLength; i++)
        {
            pair<vector<int>, vector<float>> lstPredictedClasses;
            lstPredictedClasses = getTopKNeighbours(testSet[i], trainSet, trainSetLength, k);
            testSet[i].predicted_classes = lstPredictedClasses;
        }
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
    // for (const auto &item : topKClassMap)
    // {
    //     cout << "topKClassMap -> Class name: " << item.first << " | Number: " << item.second << endl;
    // }
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
    // Compare prediction with class in Test set
    // "i" is of type size_t because the size() functions returns an unsigned integer (represent only non-negative values)
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
    " (" << static_cast<float>(100.0) *numCorrectPredictions/static_cast<float>(inTestSet.size()) << ")" << endl << endl;
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
    // usage
    // CLI: ./knn -d ..data/iris.data
    // vector<string> data_sources{"iris.data", "iris_extended.data"};
    // vector<int> k_sizes{5,7,9};
    // vector<double> split_sizes{0.6, 0.7, 0.8};
    
    // for (string data_path : data_sources)
    // {
    //     for (int k : k_sizes)
    //     {
    //         for (double split_size : split_sizes)
    //         {
    //             printf("Working on KNN | dataset: %s | k: %d | split: %.1f\n", data_path.c_str(), k, split_size);
    //             // 1. Read iris dataset
    //             vector<DataPoint> dataset = readDataset(data_path);
    //             cout << "Dataset size: " << dataset.size() << endl;
    //             //printDataset(dataset);
    //             // 2. Shuffle iris dataset
    //             shuffleDataset(dataset);
    //             //printDataset(dataset);
    //             // 3. Split data into training and testing set
    //             pair<vector<DataPoint>, vector<DataPoint>> splitted_set = splitDataset(dataset, split_size);
    //             dataset.clear();
    //             printf("Training set size: %d | Test set size: %d\n", splitted_set.first.size(), splitted_set.second.size());
    //             // 4. Execute KNN logic (Here we'll measure the elapsed time)
    //             vector<DataPoint> trainData = splitted_set.first;
    //             vector<DataPoint> testData = splitted_set.second;
    //             splitted_set.first.clear();
    //             splitted_set.second.clear();
    //             chrono::time_point<chrono::high_resolution_clock> start_time = chrono::high_resolution_clock::now();
    //             run_knn_omp(trainData, trainData.size(), testData, testData.size(), k);
    //             chrono::time_point<chrono::high_resolution_clock> end_time = chrono::high_resolution_clock::now();
    //             chrono::duration<double> elapsed_time = end_time - start_time;
    //             printf("Elapsed time for KNN is: %6.4f (ms) \n", elapsed_time*1000.0);
    //             // 5. Measure accuracy of input K
    //             get_knn_accuracy(trainData, testData, k);
    //             trainData.clear();
    //             testData.clear();
    //         }
    //     }
    // }

    string dataset_path = "../data/iris.data";
    int k = 7;
    float split_size = 0.7;
    printf("Working on KNN | dataset: %s | k: %d | split: %.1f\n", dataset_path.c_str(), k, split_size);
    // 1. Read iris dataset
    // --------------------------------
    vector<DataPoint> dataset = readDataset(dataset_path);
    cout << "Dataset size: " << dataset.size() << endl;
    //printDataset(dataset);
    // 3. Shuffle iris dataset
    // --------------------------------
    cout << "Shuffling the Iris dataset..." << endl;
    shuffleDataset(dataset);
    //printDataset(dataset);
    // 4. Split data into training and testing set
    // --------------------------------
    pair<vector<DataPoint>, vector<DataPoint>> splitted_set = splitDataset(dataset, split_size);
    printf("Training set size: %ld | Test set size: %ld\n", splitted_set.first.size(), splitted_set.second.size());
    // 5. Execute KNN logic (Here we'll measure the elapsed time)
    // --------------------------------
    vector<DataPoint> trainData = splitted_set.first;
    vector<DataPoint> testData = splitted_set.second;
    chrono::time_point<chrono::high_resolution_clock> start_time = chrono::high_resolution_clock::now();
    run_knn_omp(trainData, trainData.size(), testData, testData.size(), k);
    chrono::time_point<chrono::high_resolution_clock> end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end_time - start_time;
    printf("Elapsed time for KNN is: %lf (sec) or %6.4f (millisec) or %6.4f (microsec) or %6.4f (nanosec) \n", elapsed_time.count(), elapsed_time.count()*1000.0, elapsed_time.count()*1000000, elapsed_time.count()*1000000000);
    // 6. Measure accuracy of input K
    // --------------------------------
    get_knn_accuracy(trainData, testData, k);
    return 0;
}
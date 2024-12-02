#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <map>
#include <chrono>
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

double calculateEuclideanDistance(DataPoint testItem, DataPoint trainDataItem)
{
    double distance = 0.0;
    double tmp_total = 0.0;
    for (int i = 0; i < trainDataItem.size; i++)
    {
        tmp_total += (pow(trainDataItem.fields[i] - testItem.fields[i], 2));
    }
    distance = sqrt(tmp_total);
    return distance;
}

pair<vector<int>, vector<double>> getNeighbours(DataPoint testDp, vector<DataPoint> trainData, int trainDataLength, int k) {
    // Vector to store the highest K distances
    vector<double> topKDistances(k);
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
        double distance = 0.0;
        distance = calculateEuclideanDistance(testDp, trainData[i]);
        // Set the top k distances and their indices for the given test data point
        int idx = 0;
        while (idx < k && topKIndices[idx] != -1 && topKDistances[idx] < distance)
        {
            idx++;
        }
        if (idx < k)
        {
            int shift_idx = k - 1;
            while (shift_idx > idx)
            {
                topKIndices[shift_idx] = topKIndices[shift_idx - 1];
                topKDistances[shift_idx] = topKDistances[shift_idx - 1];
                shift_idx--;
            }
            topKIndices[idx] = i;
            topKDistances[idx] = distance;
        }
    }
    return make_pair(topKIndices, topKDistances);
}

void run_knn(vector<DataPoint> &trainSet, int trainSetLength, vector<DataPoint> &testSet, int testSetLength, int k)
{
    // Test every Data point in the test dataset, and predict its class
    for (int i = 0; i < testSetLength; i++)
    {
        testSet[i].predicted_classes = getNeighbours(testSet[i], trainSet, trainSetLength, k);
    }
}

string getPredictedClass(vector<int> topKIndices, vector<double> topKDistances, vector<DataPoint> trainData, int k)
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

void get_knn_accuracy(vector<DataPoint> trainSet, vector<DataPoint> inTestSet, int k)
{
    int numCorrectPredictions = 0;
    // Compare prediction with class in Test set
    for (int i = 0; i < inTestSet.size(); i++)
    {
        string predictedClass = getPredictedClass(inTestSet[i].predicted_classes.first, inTestSet[i].predicted_classes.second, trainSet, k);
        if (predictedClass.compare(inTestSet[i].class_name) == 0)
        {
            numCorrectPredictions++;
        }
    }
    // Show accuracy
    cout << numCorrectPredictions << " of " << inTestSet.size() << 
    " (" << static_cast<double>(100.0) *numCorrectPredictions/static_cast<double>(inTestSet.size()) << ")" << endl << endl;
}

void printDataset(vector<DataPoint> &inputData)
{
    for (int i = 0; i < inputData.size(); i++) {
        cout << "[" << i << "] Class: " << inputData[i].class_name << " | Fields: ";
        for (int j = 0; j < inputData[i].fields.size(); j++) {
            cout << inputData[i].fields[j] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char* argv[])
{
    // usage
    // CLI: ./knn -d ..data/iris.data
    vector<string> data_sources{"iris.data", "iris_extended.data"};
    vector<int> k_sizes{5,7,9};
    vector<double> split_sizes{0.6, 0.7, 0.8};
    
    for (string data_path : data_sources)
    {
        for (int k : k_sizes)
        {
            for (double split_size : split_sizes)
            {
                printf("Working on KNN | dataset: %s | k: %d | split: %.1f\n", data_path.c_str(), k, split_size);
                // 1. Read iris dataset
                vector<DataPoint> dataset = readDataset(data_path);
                cout << "Dataset size: " << dataset.size() << endl;
                //printDataset(dataset);
                // 2. Shuffle iris dataset
                shuffleDataset(dataset);
                //printDataset(dataset);
                // 3. Split data into training and testing set
                pair<vector<DataPoint>, vector<DataPoint>> splitted_set = splitDataset(dataset, split_size);
                dataset.clear();
                // cout << "Training Set size: " << splitted_set.first.size() << endl;
                // cout << "Test Set size: " << splitted_set.second.size() << endl;
                printf("Training set size: %d | Test set size: %d\n", splitted_set.first.size(), splitted_set.second.size());
                // 4. Execute KNN logic (Here we'll measure the elapsed time)
                vector<DataPoint> trainData = splitted_set.first;
                vector<DataPoint> testData = splitted_set.second;
                splitted_set.first.clear();
                splitted_set.second.clear();
                chrono::time_point<chrono::high_resolution_clock> start_time = chrono::high_resolution_clock::now();
                run_knn(trainData, trainData.size(), testData, testData.size(), k);
                chrono::time_point<chrono::high_resolution_clock> end_time = chrono::high_resolution_clock::now();
                chrono::duration<double> elapsed_time = end_time - start_time;
                printf("Elapsed time for KNN is: %6.4f (ms) \n", elapsed_time*1000.0);
                // 5. Measure accuracy of input K
                get_knn_accuracy(trainData, testData, k);
                trainData.clear();
                testData.clear();
            }
        }
    }

    // string dataset_path = "iris_extended.data";
    // int k = 9;
    // double split_size = 0.6;
    // printf("Working on KNN | dataset: %s | k: %d | split: %.1f\n", dataset_path.c_str(), k, split_size);
    // // 1. Read iris dataset
    // vector<DataPoint> dataset = readDataset(dataset_path);
    // cout << "Dataset size: " << dataset.size() << endl;
    // //printDataset(dataset);
    // // 2. Normalize iris dataset
    // // 3. Shuffle iris dataset
    // cout << "Shuffling the Iris dataset..." << endl;
    // shuffleDataset(dataset);
    // //printDataset(dataset);
    // // 4. Split data into training and testing set
    // pair<vector<DataPoint>, vector<DataPoint>> splitted_set = splitDataset(dataset, split_size);
    // cout << "Training Set size: " << splitted_set.first.size() << endl;
    // cout << "Test Set size: " << splitted_set.second.size() << endl;
    // // 5. Execute KNN logic (Here we'll measure the elapsed time)
    // vector<DataPoint> trainData = splitted_set.first;
    // vector<DataPoint> testData = splitted_set.second;
    // chrono::time_point<chrono::high_resolution_clock> start_time = chrono::high_resolution_clock::now();
    // run_knn(trainData, testData, k);
    // chrono::time_point<chrono::high_resolution_clock> end_time = chrono::high_resolution_clock::now();
    // chrono::duration<double> elapsed_time = end_time - start_time;
    // printf("Elapsed time for KNN is: %6.4f (ms) \n", elapsed_time*1000.0);
    // // 6. Measure accuracy of input K
    // get_knn_accuracy(trainData, testData, k);
    return 0;
}
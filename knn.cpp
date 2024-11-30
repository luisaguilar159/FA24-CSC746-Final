#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <map>
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

vector<DataPoint> shuffleDataset(vector<DataPoint> &inputDataset)
{
    random_device rd;
    mt19937 g(rd());
    shuffle(inputDataset.begin(), inputDataset.end(), g);
    return inputDataset;
}

pair<vector<DataPoint>, vector<DataPoint>> splitDataset(vector<DataPoint> inputDataset, float inputPercentage)
{
    vector<DataPoint> training_set;
    vector<DataPoint> test_set;
    float calcPercentage = inputDataset.size() * inputPercentage;
    size_t idx = 0;
    cout << "calPercentage: " << inputDataset.size() * inputPercentage << endl;
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
    cout << "Training set: " << training_set.size() << endl;
    cout << "test set: " << test_set.size() << endl;
    return make_pair(training_set, test_set);
}

string getNeighbours(DataPoint testDp, vector<DataPoint> trainData, int k) {
    // Array to store the highest K distances
    double *topKDistances = new double[k];
    // Array to store the indices of the highest K distances
    int *topKIndices = new int[k];
    // Fill the array with -1
    fill_n(topKIndices, k, -1);
    // Get the distance bw given data point and train dataset
    for (int i = 0; i < trainData.size(); i++)
    {
        double distance = 0.0;
        double tmp_total = 0.0;
        for (int j = 0; j < trainData[i].size; j++) {
            tmp_total += (pow(trainData[i].fields[j] - testDp.fields[j], 2));
        }
        distance = sqrt(tmp_total);
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

void knn_predict(vector<DataPoint> trainSet, vector<DataPoint> testSet, int k)
{
    int numCorrectPredictions = 0;
    // Test every Data point in the test dataset, and predict its class
    for (int i = 0; i < testSet.size(); i++) 
    {
        string predClass = getNeighbours(testSet[i], trainSet, k);
        cout << "predicted Class[" << i << "]: " << predClass << endl;
        // Compare prediction with class in Test set
        if (predClass.compare(testSet[i].class_name) == 0)
        {
            numCorrectPredictions++;
        }
    }
    // Show accuracy
    cout << numCorrectPredictions << " of " << testSet.size() << 
    " (" << static_cast<double>(100.0) *numCorrectPredictions/static_cast<double>(testSet.size()) << ")" << endl;
}

int main(int argc, char* argv[])
{
    // usage
    // CLI: ./knn -d ..data/iris.data
    cout << "You sent [" << argc << "] arguments in total" << endl;
    cout << "Hello! I will run the KNN algorithm :)" << endl;
    string dataset_path = "iris.data";
    // 1. Read iris dataset
    vector<DataPoint> dataset = readDataset(dataset_path);
    cout << "Dataset size: " << dataset.size() << endl;
    for (int i = 0; i < dataset.size(); i++) {
        cout << "[" << i << "] Class: " << dataset[i].class_name << " | Fields: ";
        for (int j = 0; j < dataset[i].fields.size(); j++) {
            cout << dataset[i].fields[j] << " ";
        }
        cout << endl;
    }
    // 2. Normalize iris dataset
    // 3. Shuffle iris dataset
    cout << "Shuffle the Iris dataset" << endl;
    shuffleDataset(dataset);
    for (int i = 0; i < dataset.size(); i++) {
        cout << "[" << i << "] Class: " << dataset[i].class_name << " | Fields: ";
        for (int j = 0; j < dataset[i].fields.size(); j++) {
            cout << dataset[i].fields[j] << " ";
        }
        cout << endl;
    }
    // 4. Split data into training and testing set
    pair<vector<DataPoint>, vector<DataPoint>> splitted_set = splitDataset(dataset, 0.7);
    cout << "Training Set size: " << splitted_set.first.size() << endl;
    cout << "Test Set size: " << splitted_set.second.size() << endl;
    // 5. Execute KNN logic ()
    knn_predict(splitted_set.first, splitted_set.second, 5);
    // 6. Measure accuracy of input K
}
#include <string>
#include <vector>

using namespace std;

class DataPoint
{
    public:
    vector<float> fields;
    int size;
    string class_name;
    pair<vector<int>, vector<float>> predicted_classes;

    DataPoint(void) {
        size = 4;
        fields.resize(0);
    }

    void print(int main_idx, int field_length) {
        printf("\nDataPoint [%d] | ", main_idx);
        printf("Size: %d | ", size);
        printf("Class: %f | ", fields[4]);
        for (int i = 0; i < field_length; i++) {
            printf("%f ", fields[i]);
        }        
    }

    private:
};

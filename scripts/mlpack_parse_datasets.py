'''
This script parses the "iris.csv" and "iris_extended.csv" files into smaller instances of the same files.
The script Receives each csv and generates smaller datasets. For instance, the Iris dataset (iris.csv) of 150 rows, will be parse 
to generate 5 smaller datasets. "iris09.csv" will be 90% of the size of the original dataset. "iris08.csv" will be 80% of 
the size of the original dataset. And so on for 70%, 60%, and 50%.
This same process will happen for the "iris_extended.csv" dataset file.

All output CSV files will be stored in the "data/input/mlpack_data" directory.
These files will be used to run the vendor implementation. Command line example:

mlpack_knn --k 5 --reference_file iris05.csv --neighbors_file neighbors05.csv --distances_file distances05.csv --verbose

The above command will execute the K-nearest neighbors algorithm in mlpack using 50% of the data of the "iris.csv" dataset.


Original and extended Iris datasets: https://www.kaggle.com/datasets/samybaladram/iris-dataset-extended/data
'''

from pathlib import Path
import pandas as pd

app_dir = Path(__file__).parent.parent

print(app_dir)

iris_df = pd.read_csv(app_dir/ "data/iris.csv")
iris_extended_df = pd.read_csv(app_dir/ "data/iris_extended.csv")
# Extract only the 4 fields we want
iris_df = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
iris_extended_df = iris_extended_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
print(iris_df.head())
print(iris_df.info())
print(iris_extended_df.head())
print(iris_extended_df.info())

# Create dataframes of 90%, 80%, 70%, 60% and 50% of original dataset
split_sizes_dict = {
    "09": 0.9,
    "08": 0.8,
    "07": 0.7,
    "06": 0.6,
    "05": 0.5
}
# Create small Iris datasets based on the split sizes above
lenght_iris = len(iris_df)
for name_perc, val_perc in split_sizes_dict.items():
    tmp_perc = lenght_iris * val_perc
    tmp_iris_df = iris_df.head(int(tmp_perc))
    tmp_csv_name = "data/mlpack_data/input/iris" + name_perc + ".csv"
    tmp_iris_df.to_csv(app_dir/ tmp_csv_name, header=False, index=False)
# Iris extended dataset based on the split sizes above
length_iris_ext = len(iris_extended_df)
for name_perc, val_perc in split_sizes_dict.items():
    tmp_ext_perc = length_iris_ext * val_perc
    tmp_iris_ext_df = iris_extended_df.head(int(tmp_ext_perc))
    tmp_csv_name = "data/mlpack_data/input/iris_ext" + name_perc + ".csv"
    tmp_iris_ext_df.to_csv(app_dir/ tmp_csv_name, header=False, index=False)

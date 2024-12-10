'''
This script parses the "iris_extended.csv" file, filters it by specific columns and generates a .data file.
Compared to the widely-known Iris dataset, the extended version (from Kaggle) has 1200 rows.

The output file "data/iris_extended.data" is used by our "knn.cpp" file when running the knn algorithm.

Extended Iris dataset: https://www.kaggle.com/datasets/samybaladram/iris-dataset-extended/data
'''

from pathlib import Path
import pandas as pd
import os

app_dir = Path(__file__).parent

print(app_dir)

iris_ext_df = pd.read_csv(app_dir/ "iris_extended.csv")
# Extract only the 5 fields we want
iris_ext_df = iris_ext_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']]
print(iris_ext_df.head())
print(iris_ext_df.info())
# Create CSV file (we'll change the extension to .data)
iris_ext_df.to_csv('new_iris_ext.csv', header=False, index=False)
# Rename file to .data extension
os.rename('new_iris_ext.csv', 'iris_extended.data')

# OULAD_Analysis

## To run
  ```
  1. Please download the dataset from https://analyse.kmi.open.ac.uk/open_dataset
    The folder about the dataset should be placed in the same directory as this folder
    The result will be in the processed_data folder under this folder
  2. Open terminal and run `python3 preprocess_new.py` to get train.csv and test.csv.
    These two are concatenated from studentVle.csv, vle.csv and studentInfo.csv. Rows
    in test.csv are last actions for all students
  3. Run `python3 preprocess_new.py` to get their sequencing files
  4. Run `python3 unsup.py` to get the MSE result from the unsupervised learning model
  ```

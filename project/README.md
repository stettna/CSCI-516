# CSCI-516 Final Project
This folder contains the source code and artifacts for the my CSCI - 516 Final Project. This project uses an SVM and features generated from the Kaggle Titanic dataset to predict Titanic passenger survival.

## Data
All of the CSV data files used in my experiments, inlcuding all of the engineered features, are included in the results folder. The files are sorted by experiment for easy navigation. All files were created using the [Kaggle Titanic dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset).

## Results
All CSV prediction files used in the evaluations of this work are provided in the data folder. The files are sorted by experiment for easy navigation. For each set of prediction files, there is also a corresonding .txt containing the results of the evaluations.

## Scripts
This folder contains all code used for data processing, feature engineering, and survival prediction.

- generate_new_features.py - This script takes in a path to the data CSV file and outputs a new CSV file containing the desired set of generated features. To generate a new feature, I simply modified the appropriate block of code; however, I left all code for the features I engineered in comments. 

```
python generate_new_features.py <data.CSV>
```

- pre_process_data.py - This file takes a CSV data file path and outputs a new CSV file with the selected, processed features. It is currently designed to process and output the standard features used in this study, but line 32 can be modified to output any of the Kaggle features.

```
python pre_process_data.py <data.csv>
```

- random_guess.py - This file takes in the paths to the training and testing CSV files and outputs a CSV containing the predictions. It simply outputs random guesses for the passengers in the testing set, taking into account the distribution of the training set to create a more balanced randomization 

```
python random_guess.py <train.csv> <test.csv>
```

- run_svm.py - This script contains the SVM code. It reads in the training and testing CSV file paths, trains the model and tunes the hyperparamters on the desired data features from the training set, and then generates a prediction CSV file based on the testing data samples. It is setup to run these stpes in a loop based on the features provided on line 57.   

```
python run_svm.py <train.csv> <test.csv>
```

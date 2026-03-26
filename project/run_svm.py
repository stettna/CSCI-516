import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def prepare_data(training_path, testing_path):
    #Load data
    train_df = pd.read_csv(training_path)
    test_df = pd.read_csv(testing_path)

    #Target column
    target_column = "Survived"  

    #Split features and labels
    X_training = train_df.drop(columns=[target_column])
    X_training = X_training.drop(columns=["PassengerId"])
    labels = train_df[target_column]
    ids = test_df["PassengerId"]

    #Scale features
    scaler = StandardScaler()
    X_training = scaler.fit_transform(X_training)
    X_testing = scaler.transform(test_df.drop(columns=["PassengerId"]))

    return X_training, X_testing, labels, ids

def output_to_CSV(preds, ids):

    df = pd.DataFrame({
        "PassengerId": ids,
        "Survived": preds
    })

    df.to_csv("predictions.csv", index=False, header=True)

def main():

    if len(sys.argv) == 3:
        training_path = sys.argv[1]
        testing_path = sys.argv[2]
    else:
        print("Wrong number of arguments provided. Must provide path to the training AND testing CSV files.")
        return

    training_data, testing_data, labels, ids = prepare_data(training_path, testing_path)

    #Train SVM model
    model = SVC(kernel='rbf', C=1.0, gamma='scale')  # try 'linear' or 'poly' too
    model.fit(training_data, labels)

    #Predictions on testing data
    preds = model.predict(testing_data)

    # Output results
    output_to_CSV(preds, ids)

    print("---- Run Complete ----")
    print("# of preds =", len(preds))
    

if __name__ == "__main__":
    main()
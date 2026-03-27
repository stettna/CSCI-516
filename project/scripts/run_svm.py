import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def prepare_data(training_path, testing_path, cols=[]):

    #Load data
    train_df = pd.read_csv(training_path)
    test_df = pd.read_csv(testing_path)

    #Target/id column
    target_column = "Survived"
    id = "PassengerId"
    cols.append(id)

    #Split features and labels
    test_samples = test_df.drop(columns=cols)
    test_ids = test_df[id]

    cols.append(target_column)
    train_samples = train_df.drop(columns=cols)
    train_labels = train_df[target_column]

    #Scale features
    scaler = StandardScaler()
    train_samples = scaler.fit_transform(train_samples)
    test_samples = scaler.transform(test_samples)

    return train_samples, test_samples, train_labels, test_ids

def output_to_CSV(preds, ids, name):

    df = pd.DataFrame({
        "PassengerId": ids,
        "Survived": preds
    })

    df.to_csv("predictions_"+ name +".csv", index=False, header=True)

def main():

    if len(sys.argv) == 3:
        training_path = sys.argv[1]
        testing_path = sys.argv[2]
    else:
        print("Wrong number of arguments provided. Must provide path to the training AND testing CSV files.")
        return

    #Lists of all features in the given dataset
    features = ["Pclass_age", "Relatives_age", "Weight_age"]
    #features = ["Pclass", "Age", "Relatives", "Sex"]

    eval_name = ""

    #This loop will train, tune, and eval the model on each indivual feature 
    for i in range(0,len(features) + 1):

        #Features to drop
        if (i == len(features)):
            drop = []
            eval_name = "all_features"
        else:
            drop = features[:i] + features[i+ 1:]
            eval_name = features[i] + "_only"

        train_samples, test_samples, train_labels, ids = prepare_data(training_path, testing_path, drop)

        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }

        #Train SVM model and tune hyperparameters
        grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
        grid.fit(train_samples, train_labels)

        #Get final eval results
        best_model = grid.best_estimator_
        preds = best_model.predict(test_samples)

        #Output results
        print("----- " + eval_name + " results -----")
        output_to_CSV(preds, ids, eval_name)
        print("Best parameters:", grid.best_params_)
        print("# of preds =", len(preds))
        print("-------------------------------------")
    

if __name__ == "__main__":
    main()
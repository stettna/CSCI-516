import pandas as pd
import numpy as np
import sys

def main():
    #Load data file
    testing_path = ""
    training_path = ""

    if len(sys.argv) == 3:
        training_path = sys.argv[1]
        testing_path = sys.argv[2]
    else:
        print("Wrong number of arguments provided. Must provide path to the training AND testing CSV files.")
        return

    train_df = pd.read_csv(training_path)
    test_df = pd.read_csv(testing_path)
    n_samples = 418

    # Estimate probability from training data
    np.random.seed(516)
    p = train_df["Survived"].mean()  # probability of class 1
    random_preds = np.random.choice([0, 1],size=n_samples, p=[1-p, p])

    output = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],  # if exists
        "Survived": random_preds
    })

    #Save processed data to new CSV
    output.to_csv("predictions_random.csv", index=False, header=True)

if __name__ == "__main__":
    main()
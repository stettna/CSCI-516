import pandas as pd
import numpy as np
import sys

def main():
    #Load data file
    file_path = ""

    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    else:
        print("Wrong number of arguments provided. Must provide path to the CSV file.")
        return

    df = pd.read_csv(file_path)

    print("\nColumn names:")
    print(df.columns)

    print("\nBasic info:")
    print(df.info())

    #---------------------------------------
    #This is the halfway between max and min        
    #central_age = 38
    #influence_region = 40
    #---------------------------------------

    #---------------------------------------
    #This is the median of the dataset       
    central_age = 28 
    influence_region = 30
    #---------------------------------------

    #---------------------------------------
    #Distance Weighting
    #df["Weight_age"] = np.abs(df["Age"] - central_age)
    #---------------------------------------

    #---------------------------------------
    #Gaussian weighting
    df["Weight_age"] = 1 - np.exp(-((df["Age"] - central_age)**2) / (2 * influence_region**2))
    #---------------------------------------


    #--------- Theshold for Age---------------
    #df["Age"] = (df["Age"] <= 15).astype(int)
    #---------------------------------------

    #Select features
    cols = ["PassengerId", "Sex", "Age"]

    #If training data then we want ground truth as well
    if "Survived" in df:
        cols.insert(0,"Survived")

    #Save processed data to new CSV
    df.to_csv("engineered_features.csv", columns=cols, index=False, header=True)

if __name__ == "__main__":
    main()
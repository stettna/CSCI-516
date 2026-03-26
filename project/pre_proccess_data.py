import pandas as pd
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

    #Combine all relatives into a single value
    df["Relatives"] = df["SibSp"] + df["Parch"]

    #A few sample had missing ages, so fill these with the mean age
    df["Age"] = df["Age"].fillna(int(df["Age"].mean()))

    #Convert sexes to binary
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    #Save processed data to new CSV
    df.to_csv("processed_data.csv", columns=["Pclass", "Sex", "Age", "Relatives"], index=False, header=True)

if __name__ == "__main__":
    main()
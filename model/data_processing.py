import pandas as pd
import glob
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


folder_path = './data/verified/*.csv'
output_path = './data_out/final_data2.csv'
all_files = glob.glob(folder_path)

def data_audit(files):
    for filename in files:
        df = pd.read_csv(filename, index_col=False)
        if(len(df.columns) != 26):
                print(f"\n[!] COLUMN MISMATCH FOUND")
                print(f"File: {filename}")
                print(f"Expected 26 columns, but found {len(df.columns)}")
                print(f"Columns found: {list(df.columns)}")
                print("Audit stopped: Column count mismatch.")
                return
        
        #TODO - Add check for spaces b/t columns & data within CSV
    print("Files Clear")    


data_audit(all_files)

li = []

for filename in all_files:
     df = pd.read_csv(filename, usecols=range(26), on_bad_lines='warn', index_col=False)
     df.columns = [c.strip() for c in df.columns] # Cleans the headers
     df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # Cleans the data

     #Drop time column
     if 't+' in df.columns:
          df = df.drop(columns=['t+'])

     li.append(df)

#The "Big Bang" - Fuse them into one master dataframe
master_df = pd.concat(li, axis=0, ignore_index=True)

master_df.to_csv(output_path, index=False)

print(f"Loaded {len(all_files)} files into a single matrix of {master_df.shape[0]} rows.")
print(master_df.head(10))
print(master_df['class'].value_counts())
print(master_df['class'].value_counts(normalize=True) * 100)


def split_files(file):
     
     df = pd.read_csv(file, index_col=False)
     X = df.drop('class', axis=1)
     Y = df['class']

     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.20, random_state=42, stratify=Y)

     #Create test file
     test_df = pd.concat([y_test, x_test], axis=1)
     test_df.to_csv('./data_out/test_data_pure.csv', index=False)
     print(f"Saved {len(test_df)} rows to test_data_pure.csv")
     print(test_df['class'].value_counts())

     #Undersample normal rows
     rus = RandomUnderSampler(sampling_strategy={0: 2048}, random_state=42)
     X_train_under, y_train_under = rus.fit_resample(x_train, y_train)

     strategy = {0:2048, 1:2048, 2:2048, 3:2048, 4:2048}

     #Oversample fault rows in training file
     ros = RandomOverSampler(sampling_strategy=strategy, random_state=42)
     X_train_final, y_train_final = ros.fit_resample(X_train_under, y_train_under)
     
     #Write to training CSV
     train_resampled_df = pd.concat([y_train_final, X_train_final], axis=1)
     #train_resampled_df = train_resampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

     train_resampled_df.to_csv('./data_out/train_data_oversampled.csv', index=False)
     
     print(f"Saved {len(train_resampled_df)} rows to training_data.csv")
     print(train_resampled_df['class'].value_counts())

split_files(output_path)

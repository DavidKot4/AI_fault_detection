import pandas as pd
import glob
from imblearn.over_sampling import RandomOverSampler

folder_path = './data/verified/*.csv'
output_path = './data_out/data_final.csv'
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

#split normal & fault rows
normal_rows = master_df[master_df['class'] == 0].sample(n=1032, random_state=42)
fault_rows = master_df[master_df['class'] != 0]

#Oversample fault rows to 1000
X = fault_rows.drop(columns=['class'])
Y = fault_rows['class']

strategy = {1: 1032, 2: 1032, 3: 1032, 4: 1032}

ros = RandomOverSampler(sampling_strategy=strategy, random_state=42)
x_resample, y_resample = ros.fit_resample(X, Y)

faults_balanced = pd.DataFrame(x_resample)
faults_balanced['class'] = y_resample

#Concat with normal rows
balanced_df = pd.concat([normal_rows, faults_balanced], axis=0)
#Shuffle rows
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True) 

#write to CSV
balanced_df.to_csv(output_path, index=False)

print(f"Loaded {len(all_files)} files into a single matrix of {master_df.shape[0]} rows.")
print(balanced_df.head(10))
print(balanced_df['class'].value_counts())
print(balanced_df['class'].value_counts(normalize=True) * 100)

  
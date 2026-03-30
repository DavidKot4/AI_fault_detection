import pandas as pd
import glob

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
     
     #Desample data (keeps every 4th row)
     df = df.iloc[::4, :]
     li.append(df)

#The "Big Bang" - Fuse them into one master dataframe
master_df = pd.concat(li, axis=0, ignore_index=True)

#write to master CSV
master_df.to_csv(output_path, index=False)

print(f"Loaded {len(all_files)} files into a single matrix of {master_df.shape[0]} rows.")
print(master_df.head(10))
print(master_df['class'].value_counts())
print(master_df['class'].value_counts(normalize=True) * 100)

  
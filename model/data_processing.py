import pandas as pd
import glob
import os

folder_path = './data/verified/*.csv'
output_folder = './data_dup'
os.makedirs(output_folder, exist_ok=True)
all_files = glob.glob(folder_path)

li = []

def desample_data(tar_file, output_path):
    df = pd.read_csv(tar_file, index_col=False)
    df = df.iloc[::4, :].copy()
    
    newfile = os.path.join(output_path, os.path.basename(tar_file))
    df.to_csv(newfile, index=False)

#for filename in all_files:
    # df = pd.read_csv(filename, index_col=False)
    # li.append(df)

    #TODO - Drop time column

#The "Big Bang" - Fuse them into one master dataframe
master_df = pd.concat(li, axis=0, ignore_index=True)

#TODO - write to master CSV
print(f"Loaded {len(all_files)} files into a single matrix of {master_df.shape[0]} rows.")
print(master_df.head(10))


  
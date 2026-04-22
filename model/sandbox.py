import torch
import joblib
import pandas as pd
import time
from model import FaultMLP
from train_utils import load_train_test_set, predict_single_row, plot_feature_importance, predict_rows

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running model with: {device}")

#LOAD SCALER
scaler = joblib.load('./model/saved_models/data_scaler_v5.pkl')

#INITIALIZE & LOAD MODEL
model = FaultMLP(input_size=12, num_classes=5)
model.load_state_dict(torch.load('./model/saved_models/fault_model_v5.pth', map_location=device))
model.to(device)
model.eval()
print("Loaded model successsfully")

newData_df = pd.read_csv('./data_out/unseen_data.csv')

drop_columns = ["A_L1","A_L2","A_L3","VA_L1","VA_L2","VA_L3","W_L1","W_L2","W_L3","Q_L1","Q_L2","Q_L3"]
     
newData_df = newData_df.drop(columns=drop_columns, errors="ignore")

#save 'base' row for per-unit calc
exclude_cols = ['PF_L1', 'PF_L2', 'PF_L3', 'THD_L1', 'THD_L2', 'THD_L3']
pu_cols = [c for c in newData_df.columns if c not in exclude_cols]  

#For each column within df, calculate per unit value
newData_df[pu_cols] = newData_df[pu_cols] / newData_df[pu_cols].iloc[1]

print(newData_df.head(10))

#Load test/train data and convert to tensors
train_df = pd.read_csv('./data_out/train_data.csv')
test_df = pd.read_csv('./data_out/test_data.csv')

print(train_df.groupby('class')[['V_L1', 'V_L2', 'V_L3']].mean())

train_dataset, test_dataset = load_train_test_set(train_df, test_df)

feature_names = [
     "V_L1", "V_L2", "V_L3", "V_L12", "V_L23", "V_L31",
    "PF_L1", "PF_L2", "PF_L3", "THD_L1", "THD_L2", "THD_L3"
]

x_test_raw = test_df.drop(columns=['class']).values
x_test_scaled = scaler.transform(x_test_raw)
y_test = test_df['class'].values

plot_feature_importance(model, x_test_scaled, y_test, feature_names, device)

newData_scaled = scaler.transform(newData_df)
input_tensor = torch.tensor(newData_scaled, dtype=torch.float32).to(device)

times = []

for i, row in enumerate(input_tensor):

    pred_class, confidence = predict_single_row(model, row, times)
    print(f'ROW {i} - Prediction: {pred_class} confidence: {confidence}%')


print("Avg latency (ms):", sum(times) / len(times))
print("Std dev (ms):", torch.tensor(times).std().item())
test_rows = [
    [1.0007309138698448,1.0008944544888212,1.0011444793205198,1.0009441280551346,1.0008602514032237,1.0009824729894148,1.0257911381701224,1.0099806820144437,1.0155570891447383,1.0265408543371226,1.010884066603236,1.016719414472985,1.0303758521904014,1.0139640070409195,1.0186243221668252,1.0582695134956106,1.0141635525378514,1.276328377168535,0.9684741497039796,0.9227479696273804,0.981281579,5.02643632888794,4.914964676,3.986011505126953],
    [0.8666180416994996,0.9832139312050396,1.000610419494148,0.9162327769479578,1.0197459049186373,0.9364264860613396,4.776341617208435,6.302629841319389,2.132010118680889,4.1392637884634444,6.196833684009029,2.133311578417334,-2.2653499907360533,1.7975738668563177,1.6102322157098394,21.416700893420472,19.29936932283241,28.49905737105356,0.525301576,0.26828146,0.738196433,6.273979664,5.339056015,4.966526508],
    [1.06093187, 1.24698363, 1.34664078, 0.99955967, 1.00078962, 0.99894483, 1.05201614, 1.09660576, 1.10197174, 1.11611743, 1.36744944, 1.48396007, 0.70553334, 2.21247669, 1.04276546, 0.78987096, 1.1150405,  1.43299654, 0.18828422, 0.21421939, 0.24552712, 5.7090621, 5.32226133, 4.46400404]
]

predict_rows(test_rows, scaler, model, device)

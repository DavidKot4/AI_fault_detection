import modbus_polling as modbus_polling
import joblib
import torch
import numpy as np
import pandas as pd
import time
from pymodbus.client import ModbusTcpClient
from train_utils import predict_single_row
from model import FaultMLP

SCALER_PATH='./model/saved_models/data_scalerV3.1.pkl'
MODEL_PATH='./model/saved_models/fault_model_v3.1.pth'
DEVICE_IP="192.168.168.11"
POLL_TIME=0.125
PU_INDICIES=np.arange(18)

#load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running model with: {device}")

scaler = joblib.load(SCALER_PATH)
print("Loaded scaler successfully")

model = FaultMLP(input_size=24, num_classes=5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("Loaded model successfully")

#create modbus client
client = ModbusTcpClient(DEVICE_IP, port=502, timeout=3)
client.connect()

print(f'Connected successfully to {DEVICE_IP}')


first_row = modbus_polling.poll_device(client, DEVICE_IP)
print(first_row)
base_vals = np.array(first_row)

print(base_vals[1])
print("Saved first row for PU calculation")

#poll data
try:
    while True:

        loop_start = time.time()

        curr_row = np.array(modbus_polling.poll_device(client, DEVICE_IP))

        curr_row[PU_INDICIES] = curr_row[PU_INDICIES] / base_vals[PU_INDICIES]

        print(curr_row)

        #scale data & convert to tensor
        scaled_row = scaler.transform(curr_row.reshape(1, -1))
        tensor_row = torch.tensor(scaled_row, dtype=torch.float32).to(device)

        prediction, confidence =  predict_single_row(model, tensor_row)

        #print results (TODO - Send to frontend)
        print(f'Predicted class: {prediction} with {confidence}%')

         #precise time to sleep to ensure polling runs in every step period
        time_to_sleep = POLL_TIME - (time.time() - loop_start)
        if time_to_sleep > 0:   
            time.sleep(time_to_sleep)

except KeyboardInterrupt:
    print("Stopped by user.")

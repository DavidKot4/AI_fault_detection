import torch
import torch.nn.functional as F
import joblib
import numpy as np
from model import FaultMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running model with: {device}")

#LOAD SCALER
scaler = joblib.load('./model/saved_models/data_scaler.pkl')

#

#INITIALIZE & LOAD MODEL
model = FaultMLP(input_size=24, num_classes=5)
model.load_state_dict(torch.load('./model/saved_models/fault_model_v1.pth', map_location=device))
model.to(device)
model.eval()
print("Loaded model successsfully")



#INFERENCE LOOP
def predict_rows(sample_data):
    for i, raw_row in enumerate(sample_data):
        # Scale and Reshape
        scaled_data = scaler.transform(np.array(raw_row).reshape(1, -1))
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)

            conf, predicted = torch.max(probabilities, 1)
            percent = conf.item() * 100

        all_probs = probabilities.cpu().numpy()[0]
        print(f"Sample {i+1} Prediction: Class {predicted.item()} ({percent:.2f}% Confidence)")
        
        print(f"  Total Model Predictions: { [f'{p*100:.1f}%' for p in all_probs] }")

import torch
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from model import FaultMLP
from train_utils import load_train_test_set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running model with: {device}")

#LOAD SCALER
scaler = joblib.load('./model/saved_models/data_scalerv3.pkl')

#INITIALIZE & LOAD MODEL
model = FaultMLP(input_size=24, num_classes=5)
model.load_state_dict(torch.load('./model/saved_models/fault_model_v3.pth', map_location=device))
model.to(device)
model.eval()
print("Loaded model successsfully")

#Load test/train data and convert to tensors
train_df = pd.read_csv('./data_out/train_data_oversampled2.csv')
test_df = pd.read_csv('./data_out/test_data_pure2.csv')

print(train_df.groupby('class')[['V_L1', 'V_L2', 'V_L3']].mean())

train_dataset, test_dataset = load_train_test_set(train_df, test_df)

def plot_feature_importance(model, X_test, y_test, feature_names):
    # 1. Create a Wrapper Class that sklearn understands
    class SklearnWrapper:
        def __init__(self, model, device):
            self.model = model
            self.device = device
            self._estimator_type = "classifier"

        def predict(self, X):
            self.model.eval()
            # Convert NumPy input to Tensor
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()
        
        def score(self, X, y):
            preds = self.predict(X)
            return accuracy_score(y, preds)
        
        def fit(self, X, y):
            return self # Model is already trained

    # 2. Initialize the wrapper
    wrapped_model = SklearnWrapper(model, device)

    # 3. Calculate Permutation Importance
    # CRITICAL: Use scaled data, not raw data!
    result = permutation_importance(
        wrapped_model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
    )

    # 4. Organize for plotting
    sorted_idx = result.importances_mean.argsort()[::-1]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=result.importances_mean[sorted_idx], 
        y=[feature_names[i] for i in sorted_idx],
        hue=[feature_names[i] for i in sorted_idx], # Added to avoid warning
        legend=False,
        palette="viridis"
    )
    
    plt.title("Feature Importance: Power Grid Fault Detection", fontsize=14)
    plt.xlabel("Accuracy Drop when Shuffled", fontsize=12)
    plt.tight_layout()
    plt.show()

feature_names = [
     "V_L1", "V_L2", "V_L3", "V_L12", "V_L23", "V_L31",
    "A_L1", "A_L2", "A_L3", "VA_L1", "VA_L2", "VA_L3",
    "W_L1", "W_L2", "W_L3", "Q_L1", "Q_L2", "Q_L3",
    "PF_L1", "PF_L2", "PF_L3", "THD_L1", "THD_L2", "THD_L3"
]

x_test_raw = test_df.drop(columns=['class']).values
x_test_scaled = scaler.transform(x_test_raw)
y_test = test_df['class'].values

plot_feature_importance(model, x_test_scaled, y_test, feature_names)

#LOAD DATASETS
sample_data = [
    [99.77698517,125.8847046,115.5966415,197.4106903,197.0393219,194.4289093,0.142395854,0.110764809,0.136261955,14.20782948,13.94359493,15.75142479,10.97858047,9.82690239,13.20140362,8.171661377,8.144480705,7.677629471,0.772713423,0.704761028,0.83810854,6.132116795,5.733846188,4.95142746],
    [7.929269314,203.7510071,204.117569,201.3128204,202.3263855,200.1340332,0.452580065,0.466681719,0.497536004,3.588629246,95.08686829,101.5558395,1.915901899,-66.2585907,-93.05221558,0.092108108,65.28312683,-35.59070587,0.533881247,0.69682169,0.916266501,5.113445282,5.003662586,4.182631016],
    [94.5214386,126.3402634,115.4724884,192.2930756,199.7386322,188.9520416,1.705238819,1.650065541,0.14518331,161.1816254,208.4697113,16.76467896,61.36824417,-18.44480133,14.06447792,55.23254776,112.4812317,8.047404289,0.380739689,0.088477127,0.838935137,6.367539406,5.581431866,4.953114986]
]

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


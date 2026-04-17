import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from model import FaultMLP
from train_utils import load_train_test_set, predict_single_row

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running model with: {device}")

#LOAD SCALER
scaler = joblib.load('./model/saved_models/data_scalerV3.pkl')

#INITIALIZE & LOAD MODEL
model = FaultMLP(input_size=24, num_classes=5)
model.load_state_dict(torch.load('./model/saved_models/fault_model_v3.pth', map_location=device))
model.to(device)
model.eval()
print("Loaded model successsfully")

newData_df = pd.read_csv('./data_out/unseen_data.csv')

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

# x_test_raw = test_df.drop(columns=['class']).values
# x_test_scaled = scaler.transform(x_test_raw)
# y_test = test_df['class'].values

# plot_feature_importance(model, x_test_scaled, y_test, feature_names)


newData_scaled = scaler.transform(newData_df)
input_tensor = torch.tensor(newData_scaled, dtype=torch.float32).to(device)

# noise_factor determines the intensity (0.01 = 1% noise, 0.05 = 5% noise)
noise_factor = 0.50 
noise = torch.randn_like(input_tensor) * noise_factor
input_tensor = input_tensor + noise

for i, row in enumerate(input_tensor):
    pred_class, confidence = predict_single_row(model, row)
    print(f'ROW {i} - Prediction: {pred_class} confidence: {confidence}%')





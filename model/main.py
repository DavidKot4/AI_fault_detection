import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import FaultMLP
from train_utils import train_model, evaluate_model
import joblib

#Hyperparameters
num_epochs = 50
learning_rate = 0.001
device = torch.device(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

#Create Model
model = FaultMLP(input_size=24, num_classes=5).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Train loader
df = pd.read_csv('./data_out/data_final.csv')
X = df.drop(columns=['class']).values #features
Y = df['class'].values #classification

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.20, random_state=42, stratify=Y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

#Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 5. Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

for epoch in range(num_epochs):
    loss, accr = train_model(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch} - Loss: {loss} Accuracy: {accr}%")

torch.save(model.state_dict(), "fault_model_v1.pth")
joblib.dump(scaler, 'data_scaler.pkl')

#Validation & testing
test_acc = evaluate_model(model, test_loader, device)


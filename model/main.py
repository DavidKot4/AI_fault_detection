import torch
import torch.nn as nn
from model import FaultMLP
from train_utils import train_model

#Hyperparameters
num_epochs = 50
learning_rate = 0.001
device = torch.device(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

#Create Model
model = FaultMLP(input_size=27, num_classes=4).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#TODO - Load data into train_loader & split into train and testing

for epoch in range(num_epochs):
    loss, accr = train_model(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch} - Loss: {loss} Accuracy: {accr}%")

#TODO - Validation & testing

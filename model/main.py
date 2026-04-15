import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
from model import FaultMLP
from train_utils import train_model, evaluate_model, plot_confusion_matrix, load_train_test_set

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

#Load test/train data and convert to tensors
train_df = pd.read_csv('./data_out/train_data_oversampled2.csv')
test_df = pd.read_csv('./data_out/test_data_pure2.csv')

train_dataset, test_dataset = load_train_test_set(train_df, test_df)

#Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

#Training loop
for epoch in range(num_epochs):
    loss, accr = train_model(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch} - Loss: {loss} Accuracy: {accr}%")

torch.save(model.state_dict(), "./model/saved_models/fault_model_v3.pth")

#Run test loader
all_preds = evaluate_model(model, test_loader, device)

#save problem rows
test_df.insert(0, 'predicted_class', all_preds)
mislabeled = test_df[test_df['class'] != test_df['predicted_class']]
print(f"Found {len(mislabeled)} mislabeled rows & saved to CSV.")
mislabeled.to_csv('./data_out/mislabeled_rows.csv', index=False)

#plot confusion matrix
labels = ["Normal", "1-phase", "2-phase", "2-phaseG", "3-Phase"]
plot_confusion_matrix(model, test_loader, device, labels)


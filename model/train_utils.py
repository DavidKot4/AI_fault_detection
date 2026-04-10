import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import QuantileTransformer
import seaborn as sns
from sklearn.metrics import f1_score, balanced_accuracy_score
import joblib

def load_train_test_set(train_df, test_df):
 """
    Loads and preprocesses training and testing datasets.

    This function:
    - Splits features and labels
    - Applies QuantileTransformer normalization
    - Saves the scaler for future inference
    - Converts arrays into PyTorch TensorDatasets

    Args:
        train_df (pd.DataFrame): Training dataframe containing features and 'class' column.
        test_df (pd.DataFrame): Testing dataframe containing features and 'class' column.

    Returns:
        tuple: (train_dataset, test_dataset) as PyTorch TensorDataset objects.
"""
    x_train_raw = train_df.drop(columns=['class']).values
    y_train = train_df['class'].values

    x_test_raw = test_df.drop(columns=['class']).values
    y_test = test_df['class'].values

    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
    x_train = scaler.fit_transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)

    joblib.dump(scaler, 'data_scalerV2.2.pkl')

    # Assuming X_train_scaled is your array after the scaler
    print(f"Normal data range (scaled): {np.min(x_train[y_train==0]):.4f} to {np.max(x_train[y_train==0]):.4f}")
    print(f"1-Phase fault range (scaled): {np.min(x_train[y_train==1]):.4f} to {np.max(x_train[y_train==1]):.4f}")
    print(f"2-Phase fault range (scaled): {np.min(x_train[y_train==2]):.4f} to {np.max(x_train[y_train==2]):.4f}")
    print(f"2-PhaseG fault range (scaled): {np.min(x_train[y_train==3]):.4f} to {np.max(x_train[y_train==3]):.4f}")
    print(f"3-Phase fault range (scaled): {np.min(x_train[y_train==4]):.4f} to {np.max(x_train[y_train==4]):.4f}")

    X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    #Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return train_dataset, test_dataset

"""
    Trains the model for one epoch.

    Performs forward pass, loss computation, backpropagation,
    and optimizer updates while tracking accuracy.

    Args:
        model (torch.nn.Module): Neural network model.
        train_loader (DataLoader): Training data loader.
        criterion: Loss function (e.g., CrossEntropyLoss).
        optimizer: Optimization algorithm (e.g., Adam).
        device (torch.device): CPU or GPU device.

    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
def train_model(model, train_loader, criterion, optimizer, device):
    model.train() # Set to training mode (enables Dropout)
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        # 1. Move data to the Jetson GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # 2. Clear previous gradients (Don't let them accumulate)
        optimizer.zero_grad()

        # 3. Forward Pass: Get the MLP's guess
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 4. Backward Pass: Calculate the "blame" for the error
        loss.backward()

        # 5. Optimizer Step: Nudge weights to reduce the error
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def evaluate_model(model, test_loader, device):
     """
    Evaluates the model on the test dataset.

    Computes accuracy, balanced accuracy, and macro F1-score.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): CPU or GPU device.

    Returns:
        np.ndarray: Predicted class labels for the test set.
    """
    # 1. SET TO EVALUATION MODE
    # This turns off Dropout so the model uses its full 'brain'
    model.eval() 
    
    all_preds = []
    all_labels = []

    correct = 0
    total = 0
    
    # 2. DISABLE GRADIENT CALCULATION
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get the predicted class (the highest score)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = 100 * correct / total
    print(f'Final Test Accuracy: {accuracy:.2f}%')
    print(f"Balanced Accuracy: {balanced_accuracy_score(all_labels, all_preds):.4f}")
    print(f"Macro F1-Score: {f1_score(all_labels, all_preds, average='macro'):.4f}")
    
    return all_preds

def predict_single_row(model, row):
     """
    Predicts the class of a single input sample.

    Args:
        model (torch.nn.Module): Trained model.
        row (torch.Tensor): Input tensor of shape (1, num_features).

    Returns:
        tuple: (predicted_class, confidence_percentage)
    """
    model.eval()

    with torch.no_grad():
        output = model(row)
        
        # 5. Interpret Results
        # 'output' is a list of 5 "scores" (Logits)
        # We take the index of the highest score
        _, predicted_class = torch.max(output, 1)
        
        # Get the probability (confidence) using Softmax
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence = torch.max(probabilities).item() * 100

    return predicted_class.item(), confidence

def plot_confusion_matrix(model, test_loader, device, class_names):
     """
    Plots a confusion matrix for model predictions.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): CPU or GPU device.
        class_names (list): List of class labels for display.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # 1. Collect all predictions
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 2. Calculate the matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # 3. Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('Actual Fault Type')
    plt.xlabel('Predicted Fault Type')
    plt.title('Fault Detection Confusion Matrix')
    plt.show()

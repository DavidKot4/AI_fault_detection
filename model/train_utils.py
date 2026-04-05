import torch

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
    # 1. SET TO EVALUATION MODE
    # This turns off Dropout so the model uses its full 'brain'
    model.eval() 
    
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

    accuracy = 100 * correct / total
    print(f'Final Test Accuracy: {accuracy:.2f}%')
    return accuracy

def predict_single_row(model, row):
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

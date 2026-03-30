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
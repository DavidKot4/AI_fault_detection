import torch.nn as nn
import torch.nn.functional as F

class FaultMLP(nn.Module):
    def __init__(self, input_size=27, num_classes=4):
        super(FaultMLP, self).__init__()
        
        # Layer 1: Reads in 27 features and transform them into 64-demension space (improved accuracy)
        self.fc1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.2) #Randomly turns off 20% of neurons to prevent model from overfitting
        
        # Layer 2: Hidden (64) -> Hidden (32); selects the most important features from the 64
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        
        # Layer 3: Hidden (32) -> Output (4)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Pass through Layer 1 with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        # Pass through Layer 2 with ReLU activation
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Output layer (CrossEntropyLoss handles Softmax automatically)
        x = self.fc3(x)
        return x
    
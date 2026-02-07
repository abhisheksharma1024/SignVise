import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# 1. Custom Dataset
class GestureDataset(Dataset):
    def __init__(self, x_path, y_path):
        # Load numpy files
        X = np.load(x_path, allow_pickle=True)
        y = np.load(y_path, allow_pickle=True)

        # Flatten landmarks if nested (should already be flat: 21*3 = 63 features)
        X = np.array(X, dtype=np.float32)

        # Encode string labels (A, B, C...) into integers
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        y = np.array(y, dtype=np.int64)

        # Convert to torch tensors
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. Define Model
class GestureNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GestureNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# 3. Load Dataset
train_dataset = GestureDataset("landmarks.npy", "labels.npy")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 4. Initialize Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = train_dataset.X.shape[1]   # should be 63 (21 landmarks * 3 coords)
num_classes = len(torch.unique(train_dataset.y))  # number of gesture classes

model = GestureNet(input_size=input_size, hidden_size=128, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    acc = correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {acc:.4f}")

# 6. Save Model
torch.save(model.state_dict(), "gesture_model.pth")
print("💾 Model saved as gesture_model.pth")


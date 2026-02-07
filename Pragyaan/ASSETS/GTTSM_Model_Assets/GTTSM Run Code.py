import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# 1. Load dataset
# ---------------------------
X = np.load("landmarks.npy", allow_pickle=True)
y = np.load("labels.npy", allow_pickle=True)

print("✅ Dataset loaded:", X.shape, len(y), "samples")

# Encode labels to integers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Convert to tensors
X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ---------------------------
# 2. Define model
# ---------------------------
class GestureNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Input size = 21 landmarks × 3 coords = 63
input_size = X_tensor.shape[1]
num_classes = len(encoder.classes_)

model = GestureNet(input_size, num_classes).to("cuda")

# ---------------------------
# 3. Training setup
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# 4. Training loop
# ---------------------------
epochs = 15
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to("cuda"), batch_y.to("cuda")

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

# ---------------------------
# 5. Save model + label encoder
# ---------------------------
torch.save(model.state_dict(), "gesture_model.pth")
np.save("label_classes.npy", encoder.classes_)

print("💾 Model and label classes saved!")





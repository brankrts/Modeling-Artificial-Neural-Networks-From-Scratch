import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import random
import numpy as np

digits = load_digits()
X = digits.data / np.max(digits.data)
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x

model = NN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 300
for epoch in range(epochs):
    model.train()  
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()  
        optimizer.step() 

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item()}")

model.eval()  
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == batch_y).sum().item()
        total_samples += batch_y.size(0)

    test_accuracy = total_correct / total_samples
    print(f"Test Accuracy: {test_accuracy}")

def test_on_samples(model, x_test, y_test, test_count):
    model.eval()
    with torch.no_grad():
        for _ in range(test_count):
            sample_index = random.randint(0, len(x_test) - 1)
            sample_data = torch.tensor(x_test[sample_index, :], dtype=torch.float32).unsqueeze(0)
            sample_label = y_test[sample_index]
            output = model(sample_data)
            _, predicted = torch.max(output, 1)
            print(f"True Class: {sample_label}, Predicted Class: {predicted.item()}")

test_on_samples(model, X_test, y_test, 20)

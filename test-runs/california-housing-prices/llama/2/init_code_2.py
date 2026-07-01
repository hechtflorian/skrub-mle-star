
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load the data
train_df = pd.read_csv('./input/train.csv')

# Define the features and target
X = train_df.drop('median_house_value', axis=1)
y = train_df['median_house_value']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Neural Network Regressor model using PyTorch
class NeuralNetworkRegressor(nn.Module):
    def __init__(self):
        super(NeuralNetworkRegressor, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 50)  # input layer (8) -> hidden layer (50)
        self.fc2 = nn.Linear(50, 50)  # hidden layer (50) -> hidden layer (50)
        self.fc3 = nn.Linear(50, 1)  # hidden layer (50) -> output layer (1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset and data loader
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
model = NeuralNetworkRegressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):
    for X_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Make predictions on the validation set
model.eval()
y_pred = []
with torch.no_grad():
    for X_batch, _ in val_loader:
        outputs = model(X_batch)
        y_pred.extend(outputs.cpu().numpy())

y_pred = np.array(y_pred).flatten()

# Calculate the root mean squared error
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

# Print the final validation performance
print(f'Final Validation Performance: {rmse}')

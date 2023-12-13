import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import math 
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from gat import GAT_LSTM
import sys
sys.path.append("../")
from wMSE import WeightedMSELoss

JOIN = os.path.join

print("Using torch", torch.__version__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)

data_path = "../../data/norm_data"
train_X = torch.tensor(np.load(f"{data_path}/train_data.npy")).double()
train_Y = torch.tensor(np.load(f"{data_path}/train_labels.npy")).double()
val_X = torch.tensor(np.load(f"{data_path}/val_data.npy")).double()
val_Y = torch.tensor(np.load(f"{data_path}/val_labels.npy")).double()

batch_size = 32 
train_dataset = TensorDataset(train_X, train_Y)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_X, val_Y)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for i in range(3):
    model = GAT_LSTM().double()
    model.to(DEVICE)

    weights = torch.ones(90, 19, 3)
    weights[50:71, 2:8, :] *= 4 # weight action periods and joints
    weights = weights.view(-1, 19*3).to(DEVICE)
    criterion = WeightedMSELoss(weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 150
    training_loss_hist = []
    val_loss_hist = []
    best_model = model 
    best_val_loss = math.inf
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0
        for inputs, targets in train_dataloader:

            batch = model.make_graph(inputs)
            batch = batch.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss = total_loss / len(train_dataloader)
        training_loss_hist.append(epoch_loss)
        print(f'Training - Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}')

        # Validation
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                
                batch = model.make_graph(inputs)
                batch = batch.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(batch)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

        val_loss = total_loss / len(val_dataloader)
        val_loss_hist.append(val_loss)
        print(f'Validation - Epoch [{epoch + 1}/{num_epochs}] Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

    torch.save(best_model.state_dict(), f"gat_model.pt")
    plt.plot(training_loss_hist)
    plt.plot(val_loss_hist)
    plt.title('GAT')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("gat_loss_curve.png")

    print(f'INFO: Training complete. Best model saved. Final Val Loss {best_val_loss:.8f}')

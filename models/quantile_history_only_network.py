import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange


class LossFunction(nn.Module):
    def __init__(self, forecast_length: int):
        super().__init__()
        self.forecast_length = forecast_length

    def forward(self, labels: torch.tensor, predictions: torch.tensor) -> torch.tensor:
        pass


class QuantileLoss(LossFunction):
    def __init__(self, forecast_length: int, quantile: float):
        super().__init__(forecast_length=forecast_length)
        self.quantile = quantile

    def forward(self, labels: torch.tensor, predictions: torch.tensor) -> torch.tensor:
        losses = []
        errors = labels - predictions
        losses.append(torch.max((self.quantile - 1) * errors, self.quantile * errors))
        return torch.mean(torch.cat(losses, dim=1))


# Define the PyTorch model
class QuantileHistoryNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, quantile):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.quantile = quantile

        module_list = [nn.Linear(self.input_size, self.hidden_size[0]), nn.ReLU(),
                       nn.Linear(self.hidden_size[0], self.hidden_size[1])]
        self.base_network = nn.Sequential(*module_list)

        module_list_output = [nn.ReLU(), nn.Linear(self.hidden_size[1], self.hidden_size[2]), nn.ReLU(),
                              nn.Linear(self.hidden_size[2], self.output_size)]

        self.output_network = nn.Sequential(*module_list_output)

        self.loss_function = QuantileLoss(forecast_length=self.output_size, quantile=self.quantile)

    def loss(self, labels: torch.tensor, predictions: torch.tensor) -> torch.tensor:
        return self.loss_function(labels=labels, predictions=predictions)

    def forward(self, x):
        x = self.base_network(x)
        q = self.output_network(x)
        return q


def train_quanilte_history_model(model, X_train, y_train, num_epochs, batch_size, optimizer, save_path, save_name):
    model.train()
    train_losses = []
    for epoch in trange(num_epochs):
        # Train Model
        epoch_losses = []
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = model.loss(labels=batch_y, predictions=preds)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        train_losses.append(np.mean(epoch_losses))

    save_path = save_path + "/models/"
    isExist = os.path.exists(save_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(save_path)
    full_name = save_path + save_name
    torch.save(model.state_dict(), full_name)
    return model, train_losses


def create_quantile_history_predictions(model, X_test):
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        q = test_preds
        return q

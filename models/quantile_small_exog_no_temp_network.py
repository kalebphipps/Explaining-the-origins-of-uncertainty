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
class QuantileSmallExogenousNoTemporalNetwork(nn.Module):
    def __init__(self, history_size, feature_size, hidden_size, hidden_size_history, output_size, quantile):
        super().__init__()
        self.history_size = history_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.hidden_size_history = hidden_size_history
        self.output_size = output_size
        self.quantile = quantile

        self.history_network = nn.Sequential(nn.Linear(self.history_size, self.hidden_size_history))

        self.feature1_network = nn.Sequential(nn.Linear(self.feature_size, self.hidden_size[0]))

        self.feature2_network = nn.Sequential(nn.Linear(self.feature_size, self.hidden_size[0]))

        module_list_base = [nn.ReLU(),
                            nn.Linear(self.hidden_size[0] * 2 + self.hidden_size_history, self.hidden_size[1]),
                            nn.ReLU(),
                            nn.Linear(self.hidden_size[1], self.hidden_size[2]), nn.ReLU(),
                            nn.Linear(self.hidden_size[2], self.hidden_size[3]), nn.ReLU()]
        self.main_network = nn.Sequential(*module_list_base)

        module_list_output = [nn.ReLU(), nn.Linear(self.hidden_size[3], self.hidden_size[4]), nn.ReLU(),
                              nn.Linear(self.hidden_size[4], self.output_size)]

        self.output_network = nn.Sequential(*module_list_output)

        self.loss_function = QuantileLoss(forecast_length=self.output_size, quantile=self.quantile)

    def loss(self, labels: torch.tensor, predictions: torch.tensor) -> torch.tensor:
        return self.loss_function(labels=labels, predictions=predictions)

    def forward(self, history, feature1, feature2):
        history = self.history_network(history)
        feature1 = self.feature1_network(feature1)
        feature2 = self.feature2_network(feature2)
        x = torch.cat(
            [history, feature1, feature2], dim=1)
        x = self.main_network(x)
        q = self.output_network(x)
        return q


def train_small_exogenous_no_temp_quantile_model(model, history, target, feature1, feature2, num_epochs, batch_size,
                                                 optimizer, save_path, save_name):
    model.train()
    train_losses = []
    for epoch in trange(num_epochs):
        # Train Model
        epoch_losses = []
        for i in range(0, len(history), batch_size):
            batch_history = history[i:i + batch_size]
            batch_feature1 = feature1[i:i + batch_size]
            batch_feature2 = feature2[i:i + batch_size]
            batch_target = target[i:i + batch_size]

            optimizer.zero_grad()
            preds = model(history=batch_history,
                          feature1=batch_feature1,
                          feature2=batch_feature2)
            loss = model.loss(labels=batch_target, predictions=preds)
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


def create_small_exogenous_no_temp_quantile_predictions(model, history, feature1, feature2):
    model.eval()
    with torch.no_grad():
        test_preds = model(history=history,
                           feature1=feature1,
                           feature2=feature2)
        q = test_preds
        return q

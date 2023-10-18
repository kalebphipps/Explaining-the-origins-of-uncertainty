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
class QuantileTemporalNetwork(nn.Module):
    def __init__(self, history_size, feature_size, hidden_size, hidden_size_history, output_size, quantile):
        super().__init__()
        self.history_size = history_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.hidden_size_history = hidden_size_history
        self.output_size = output_size
        self.quantile = quantile

        self.history_network = nn.Sequential(nn.Linear(self.history_size, self.hidden_size_history))

        self.hour_cos_network = nn.Sequential(nn.Linear(self.feature_size, self.hidden_size[0]))

        self.hour_sin_network = nn.Sequential(nn.Linear(self.feature_size, self.hidden_size[0]))

        self.day_cos_network = nn.Sequential(nn.Linear(self.feature_size, self.hidden_size[0]))

        self.day_sin_network = nn.Sequential(nn.Linear(self.feature_size, self.hidden_size[0]))

        self.month_cos_network = nn.Sequential(nn.Linear(self.feature_size, self.hidden_size[0]))

        self.month_sin_network = nn.Sequential(nn.Linear(self.feature_size, self.hidden_size[0]))

        module_list_base = [nn.ReLU(),
                            nn.Linear(self.hidden_size[0] * 6 + self.hidden_size_history, self.hidden_size[1]),
                            nn.ReLU(),
                            nn.Linear(self.hidden_size[1], self.hidden_size[2]), nn.ReLU(),
                            nn.Linear(self.hidden_size[2], self.hidden_size[3]), nn.ReLU(),
                            nn.Linear(self.hidden_size[3], self.hidden_size[4])]
        self.main_network = nn.Sequential(*module_list_base)

        module_list_output = [nn.ReLU(), nn.Linear(self.hidden_size[4], self.hidden_size[5]), nn.ReLU(),
                              nn.Linear(self.hidden_size[5], self.output_size)]

        self.output_network = nn.Sequential(*module_list_output)

        self.loss_function = QuantileLoss(forecast_length=self.output_size, quantile=self.quantile)

    def loss(self, labels: torch.tensor, predictions: torch.tensor) -> torch.tensor:
        return self.loss_function(labels=labels, predictions=predictions)

    def forward(self, history, hour_cos, hour_sin, day_cos, day_sin, month_cos, month_sin):
        history = self.history_network(history)
        hour_cos = self.hour_cos_network(hour_cos)
        hour_sin = self.hour_sin_network(hour_sin)
        day_cos = self.day_cos_network(day_cos)
        day_sin = self.day_sin_network(day_sin)
        month_cos = self.month_cos_network(month_cos)
        month_sin = self.month_sin_network(month_sin)
        x = torch.cat(
            [history, hour_cos, hour_sin, day_cos, day_sin, month_cos, month_sin], dim=1)
        x = self.main_network(x)
        q = self.output_network(x)
        return q


def train_quantile_temporal_model(model, history, target, hour_cos, hour_sin, day_cos, day_sin, month_cos, month_sin,
                                  num_epochs, batch_size, optimizer, save_path, save_name):
    model.train()
    train_losses = []
    for epoch in trange(num_epochs):
        # Train Model
        epoch_losses = []
        for i in range(0, len(history), batch_size):
            batch_history = history[i:i + batch_size]
            batch_hour_cos = hour_cos[i:i + batch_size]
            batch_hour_sin = hour_sin[i:i + batch_size]
            batch_day_cos = month_cos[i:i + batch_size]
            batch_day_sin = month_sin[i:i + batch_size]
            batch_month_cos = day_cos[i:i + batch_size]
            batch_month_sin = day_sin[i:i + batch_size]
            batch_target = target[i:i + batch_size]

            optimizer.zero_grad()
            preds = model(history=batch_history,
                          hour_cos=batch_hour_cos,
                          hour_sin=batch_hour_sin,
                          day_cos=batch_day_cos,
                          day_sin=batch_day_sin,
                          month_cos=batch_month_cos,
                          month_sin=batch_month_sin)
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


def create_quantile_temporal_predictions(model, history, hour_cos, hour_sin, day_cos, day_sin, month_cos, month_sin):
    model.eval()
    with torch.no_grad():
        test_preds = model(history=history,
                           hour_cos=hour_cos,
                           hour_sin=hour_sin,
                           day_cos=day_cos,
                           day_sin=day_sin,
                           month_cos=month_cos,
                           month_sin=month_sin)
        q = test_preds
        return q

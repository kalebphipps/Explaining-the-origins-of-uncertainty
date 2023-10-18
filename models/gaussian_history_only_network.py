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


class LikelihoodLossFunction(LossFunction):
    def __init__(self, forecast_length: int):
        super().__init__(forecast_length=forecast_length)

    def forward(self, labels: torch.tensor, predictions: torch.tensor) -> torch.tensor:
        losses = []
        for i in range(self.forecast_length):
            mu = predictions[:, i, 0]
            logvar = predictions[:, i, 1]
            distribution = torch.distributions.Normal(mu, logvar.exp().sqrt())
            likelikehood = distribution.log_prob(labels[:, i])
            losses.append(-likelikehood.reshape(-1, 1))
        return torch.mean(torch.mean(torch.cat(losses, dim=1), dim=1))


# Define the PyTorch model
class GaussianHistoryNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        module_list = [nn.Linear(self.input_size, self.hidden_size[0]), nn.ReLU(),
                       nn.Linear(self.hidden_size[0], self.hidden_size[1])]
        self.base_network = nn.Sequential(*module_list)

        module_list_mu = [nn.ReLU(), nn.Linear(self.hidden_size[1], self.hidden_size[2]), nn.ReLU(),
                          nn.Linear(self.hidden_size[2], self.output_size)]
        module_list_logvar = [nn.ReLU(), nn.Linear(self.hidden_size[1], self.hidden_size[2]), nn.ReLU(),
                              nn.Linear(self.hidden_size[2], self.output_size)]

        self.mu_network = nn.Sequential(*module_list_mu)
        self.logvar_network = nn.Sequential(*module_list_logvar)

        self.loss_function = LikelihoodLossFunction(forecast_length=self.output_size)

    def loss(self, labels: torch.tensor, predictions: torch.tensor) -> torch.tensor:
        return self.loss_function(labels=labels, predictions=predictions)

    def forward(self, x):
        x = self.base_network(x)
        mu = self.mu_network(x)
        logvar = self.logvar_network(x)
        return torch.cat([mu.unsqueeze(2), logvar.unsqueeze(2)], dim=2)


def train_gaussian_history_model(model, X_train, y_train, num_epochs, batch_size, optimizer, save_path, save_name):
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


def create_gaussian_history_predictions(model, X_test):
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        mu_test = test_preds[:, :, 0]
        logvar_test = test_preds[:, :, 1]
        return mu_test, logvar_test

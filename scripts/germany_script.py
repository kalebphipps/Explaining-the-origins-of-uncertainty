import os

import pandas as pd
import tomlkit
from matplotlib import pyplot as plt
from torch import optim

from metrics.crps import calculate_crps_gaussian
from models.gaussian_history_only_network import GaussianHistoryNetwork, train_gaussian_history_model, \
    create_gaussian_history_predictions
from pipelines.preprocessing_pipeline import prep_data_history

if __name__ == '__main__':
    df = pd.read_csv("../data/de.csv", index_col="start", parse_dates=True)
    df = df.drop(columns='end')
    df.index.name = "time"
    df = df.resample('H').sum()

    with open('../configs/germany_config.toml', 'rb') as f:
        configs = tomlkit.load(f)

    for key, config in configs.items():
        # Parameters
        history_length = config['history_length']
        forecast_horizon = config['forecast_horizon']
        learning_rate = config['learning_rate']
        number_epochs = config['number_epochs']
        batch_size = config['batch_size']
        hidden_size_history_network = config['hidden_size_history_network']
        hidden_size_temporal_network = config['hidden_size_temporal_network']
        hidden_size_history_encoder_temporal_network = config['hidden_size_history_encoder_temporal_network']
        test_size = config['test_size']
        save_path = config['save_path']
        save_name_history = config['save_name_history']
        save_name_temporal = config['save_name_temporal']
        target_var = config['target_var']
        descriptor = config['descriptor']

        # Prepare data for historical model
        target_train, target_test, history_train, history_test, target_scaler, history_scaler = prep_data_history(df=df,
                                                                                                                  target_var=target_var,
                                                                                                                  history_length=history_length,
                                                                                                                  forecast_horizon=forecast_horizon,
                                                                                                                  test_size=test_size)
        # Create model only using historical values
        history_model = GaussianHistoryNetwork(input_size=history_length,
                                               hidden_size=hidden_size_history_network,
                                               output_size=forecast_horizon)

        # Define optimizer for history model
        optimizer_history = optim.Adam(history_model.parameters(), lr=learning_rate)
        history_model, train_losses = train_gaussian_history_model(model=history_model,
                                                                   X_train=history_train,
                                                                   y_train=target_train,
                                                                   num_epochs=number_epochs,
                                                                   batch_size=batch_size,
                                                                   optimizer=optimizer_history,
                                                                   save_path=save_path,
                                                                   save_name=f"{descriptor}_{save_name_history}")
        # Plot Losses
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(number_epochs), train_losses)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Training Loss")
        ax.set_title("Loss Progression for History Model")
        plt.savefig(f"{save_path}/models/{descriptor}_Historyloss.png")
        plt.close(fig)

        # Calculate CRPS for history model
        mu_test_history, logvar_test_history = create_gaussian_history_predictions(model=history_model,
                                                                                   X_test=history_test)
        crps_history = calculate_crps_gaussian(target_test, mu_test_history, logvar_test_history.exp().sqrt())

        crps_save = save_path + "crps/"
        isExist = os.path.exists(crps_save)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(crps_save)
        pd.DataFrame([crps_history]).to_csv(f"{crps_save}/{descriptor}_history.csv")

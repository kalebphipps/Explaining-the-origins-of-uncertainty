import os

import matplotlib.pyplot as plt
import pandas as pd
import tomlkit
from torch import optim

from metrics.crps import calculate_crps_gaussian
from models.gaussian_history_only_network import GaussianHistoryNetwork, train_gaussian_history_model, \
    create_gaussian_history_predictions
from models.quantile_history_only_network import QuantileHistoryNetwork, train_quanilte_history_model
from pipelines.preprocessing_pipeline import prep_data_exogenous_no_temporal
from pipelines.synthetic_data_pipeline import create_synthetic_data

if __name__ == '__main__':

    with open('../configs/synthetic_config.toml', 'rb') as f:
        configs = tomlkit.load(f)

    for key, config in configs.items():
        # Parameters
        num_points = config['num_points']
        base_amplitude = config['base_amplitude']
        base_frequency = config['base_frequency']
        base_noise_scale = config['base_noise_scale']
        base_noise_amplitude = config['base_noise_amplitude']
        num_trend_events = config['num_trend_events']
        trend_parameters = config['trend_parameters']
        num_cosine_events = config['num_cosine_events']
        cosine_frequency_params = config['cosine_frequency_params']
        cosine_amplitude_params = config['cosine_amplitude_params']
        num_increased_noise_events = config['num_increased_noise_events']
        increased_noise_params = config['increased_noise_params']
        test_size = config['test_size']
        number_of_features = config['number_of_features']
        hidden_size_history = config['hidden_size_history']
        learning_rate_history = config['learning_rate_history']
        learning_rate_exog = config['learning_rate_exog']
        hidden_size_exog = config['hidden_size_exog']
        hidden_size_exog_hi = config['hidden_size_exog_hi']
        num_epochs = config['num_epochs']
        batch_size = config['batch_size']
        quantiles = config['quantiles']
        save_path = config['save_path']
        save_path_data = f"{save_path}data/"
        save_path_crps = f"{save_path}crps/"
        save_history_name = config['save_history_name']
        save_exog_name = config['save_exog_name']
        descriptor = config['descriptor']
        save_history_name = f"{save_history_name}_{descriptor}.pt"
        save_exog_name = f"{save_exog_name}_{descriptor}.pt"
        noise_type = config['noise_type']
        forecast_horizon = 20
        history_length = 40

        isExist = os.path.exists(save_path_crps)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path_crps)
        isExist = os.path.exists(save_path_data)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path_data)

        synthetic_series, trend_time_series, cosine_time_series, increased_noise_time_series = create_synthetic_data(
            size=num_points,
            base_amplitude=base_amplitude,
            base_frequency=base_frequency,
            base_noise_scale=base_noise_scale,
            base_noise_amplitude=base_noise_amplitude,
            number_of_trend_events=num_trend_events,
            trend_parameters=trend_parameters,
            number_of_cosine_events=num_cosine_events,
            cosine_frequency_parameters=cosine_frequency_params,
            cosine_amplitude_parameters=cosine_amplitude_params,
            number_of_increased_noise_events=num_increased_noise_events,
            increased_noise_parameters=increased_noise_params,
            noise_type=noise_type)

        df = pd.DataFrame(columns=["Target", "Trend", "Cosine", "Noise"])
        df['Target'] = synthetic_series
        df['Trend'] = trend_time_series
        df['Cosine'] = cosine_time_series
        df['Noise'] = increased_noise_time_series

        save_data_name = f"{save_path_data}{descriptor}_data.csv"
        df.to_csv(save_data_name)

        target_train, target_test, history_train, history_test, feature1_train, feature1_test, feature2_train, \
            feature2_test, feature3_train, feature3_test, target_scaler, history_scaler, feature1_scaler, \
            feature2_scaler, feature3_scaler = prep_data_exogenous_no_temporal(
            df=df,
            target_var='Target',
            feature1_var='Trend',
            feature2_var='Cosine',
            feature3_var='Noise',
            history_length=history_length,
            forecast_horizon=forecast_horizon,
            test_size=test_size)

        model_history = GaussianHistoryNetwork(input_size=history_length,
                                               hidden_size=hidden_size_history,
                                               output_size=forecast_horizon)
        optimizer_history = optim.Adam(model_history.parameters(), lr=learning_rate_history)
        model_history, losses_history = train_gaussian_history_model(model=model_history,
                                                                     X_train=history_train,
                                                                     y_train=target_train,
                                                                     num_epochs=num_epochs,
                                                                     batch_size=batch_size,
                                                                     optimizer=optimizer_history,
                                                                     save_path=save_path,
                                                                     save_name=f"Gaussian_{save_history_name}")

        # Plot Losses
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(num_epochs), losses_history)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Training Loss")
        ax.set_title("Loss Progression for History Model")
        plt.savefig(f"{save_path}/models/GaussianHistory_loss_{descriptor}.png")
        plt.close(fig)

        # Calculate CRPS for history model
        mu_test_history, logvar_test_history = create_gaussian_history_predictions(model=model_history,
                                                                                   X_test=history_test)
        crps_history = calculate_crps_gaussian(y_test=target_test,
                                               mu_test=mu_test_history,
                                               logvar_test=logvar_test_history)

        pd.DataFrame([crps_history]).to_csv(f"{save_path_crps}GaussianHistory_{descriptor}.csv")

        quantile_model_dict = dict()
        quantile_predictions_dict = dict()
        for q in quantiles:
            quantile_model_history = QuantileHistoryNetwork(input_size=history_length,
                                                            hidden_size=hidden_size_history,
                                                            output_size=forecast_horizon,
                                                            quantile=q)
            quantile_optimizer_history = optim.Adam(quantile_model_history.parameters(), lr=learning_rate_history)
            quantile_model_dict[q], quantile_losses_history = train_quanilte_history_model(model=quantile_model_history,
                                                                                           X_train=history_train,
                                                                                           y_train=target_train,
                                                                                           num_epochs=num_epochs,
                                                                                           batch_size=batch_size,
                                                                                           optimizer=quantile_optimizer_history,
                                                                                           save_path=save_path,
                                                                                           save_name=f"Quantile{q}_{save_history_name}")
            # Plot Losses
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(num_epochs), quantile_losses_history)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Training Loss")
            ax.set_title("Loss Progression for History Model")
            plt.savefig(f"{save_path}/models/QuantileHistory{q}_loss_{descriptor}.png")
            plt.close(fig)

        print(f"Config {key} Finished")

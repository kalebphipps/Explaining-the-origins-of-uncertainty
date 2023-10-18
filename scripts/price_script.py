import os

import matplotlib.pyplot as plt
import pandas as pd
import tomlkit
from torch import optim

from metrics.crps import calculate_crps_gaussian
from models.gaussian_exog_no_temp_network import GaussianExogenousNoTemporalNetwork, \
    train_exogenous_no_temp_gaussian_model, create_exogenous_no_temp_gaussian_predictions
from models.gaussian_small_exog_no_temp_network import GaussianSmallExogenousNoTemporalNetwork, \
    train_small_exogenous_no_temp_gaussian_model, create_small_exogenous_no_temp_gaussian_predictions
from pipelines.preprocessing_pipeline import prep_data_exogenous, prep_data_small_exogenous

if __name__ == '__main__':

    df = pd.read_csv("../data/price.csv", index_col='timestamp')
    df.index = pd.date_range(start='01-01-2011 00:00:00', freq='H', periods=len(df))
    df = df[['Forecasted Total Load', 'Forecasted Zonal Load', 'Zonal Price']]
    df.index.name = 'time'

    with open('../configs/price_config.toml', 'rb') as f:
        configs = tomlkit.load(f)

    for key, config in configs.items():
        # Parameters
        test_size = config['test_size']
        number_of_features = config['number_of_features']
        hidden_size_exogenous = config['hidden_size_exogenous']
        hidden_size_exogenous_history = config['hidden_size_exogenous_history']
        learning_rate = config['learning_rate']
        hidden_size_no_temporal = config['hidden_size_no_temporal']
        hidden_size_no_temporal_history = config['hidden_size_no_temporal_history']
        num_epochs = config['num_epochs']
        batch_size = config['batch_size']
        save_path = config['save_path']
        forecast_horizon = config['forecast_horizon']
        history_length = config['history_length']
        save_path_crps = f"{save_path}crps/"
        save_exogenous_name = config['save_exogenous_name']
        save_no_temporal_name = config['save_no_temporal_name']
        descriptor = config['descriptor']
        target_var = config['target_var']
        feature1_var = config['feature1_var']
        feature2_var = config['feature2_var']
        feature3_var = config['feature3_var']

        isExist = os.path.exists(save_path_crps)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path_crps)

        if number_of_features == 3:
            target_train, target_test, history_train, history_test, hour_cos_train, hour_cos_test, hour_sin_train, \
                hour_sin_test, day_cos_train, day_cos_test, day_sin_train, day_sin_test, month_cos_train, \
                month_cos_test, month_sin_train, month_sin_test, feature1_train, feature1_test, feature2_train, \
                feature2_test, feature3_train, feature3_test, target_scaler, history_scaler, feature1_scaler, \
                feature2_scaler, feature3_scaler = prep_data_exogenous(df=df,
                                                                       target_var=target_var,
                                                                       feature1_var=feature1_var,
                                                                       feature2_var=feature2_var,
                                                                       feature3_var=feature3_var,
                                                                       history_length=history_length,
                                                                       forecast_horizon=forecast_horizon,
                                                                       test_size=test_size)

            model_no_temporal = GaussianExogenousNoTemporalNetwork(history_size=history_length,
                                                                   feature_size=forecast_horizon,
                                                                   hidden_size=hidden_size_no_temporal,
                                                                   hidden_size_history=hidden_size_no_temporal_history,
                                                                   output_size=forecast_horizon)
            optimizer_no_temporal = optim.Adam(model_no_temporal.parameters(), lr=learning_rate)

            model_no_temporal, losses_no_temporal = train_exogenous_no_temp_gaussian_model(model=model_no_temporal,
                                                                                           history=history_train,
                                                                                           target=target_train,
                                                                                           feature1=feature1_train,
                                                                                           feature2=feature2_train,
                                                                                           feature3=feature3_train,
                                                                                           num_epochs=num_epochs,
                                                                                           batch_size=batch_size,
                                                                                           optimizer=optimizer_no_temporal,
                                                                                           save_path=save_path,
                                                                                           save_name=f"{descriptor}_{save_no_temporal_name}")
            # Plot Losses
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(num_epochs), losses_no_temporal)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Training Loss")
            ax.set_title("Loss Progression for Exogenous No Temporal Model")
            plt.savefig(f"{save_path}/models/GaussianNoTemporal_loss_{descriptor}.png")
            plt.close(fig)

            mu_test_no_temporal, logvar_test_no_temporal = create_exogenous_no_temp_gaussian_predictions(
                model=model_no_temporal,
                history=history_test,
                feature1=feature1_test,
                feature2=feature2_test,
                feature3=feature3_test)
            crps_no_temporal = calculate_crps_gaussian(y_test=target_test,
                                                       mu_test=mu_test_no_temporal,
                                                       logvar_test=logvar_test_no_temporal)
            pd.DataFrame([crps_no_temporal]).to_csv(f"{save_path_crps}GaussianNoTemporal_{descriptor}.csv")


        elif number_of_features == 2:
            target_train, target_test, history_train, history_test, hour_cos_train, hour_cos_test, hour_sin_train, \
                hour_sin_test, day_cos_train, day_cos_test, day_sin_train, day_sin_test, month_cos_train, \
                month_cos_test, month_sin_train, month_sin_test, feature1_train, feature1_test, feature2_train, \
                feature2_test, target_scaler, history_scaler, feature1_scaler, \
                feature2_scaler = prep_data_small_exogenous(df=df,
                                                            target_var=target_var,
                                                            feature1_var=feature1_var,
                                                            feature2_var=feature2_var,
                                                            history_length=history_length,
                                                            forecast_horizon=forecast_horizon,
                                                            test_size=test_size)

            model_no_temporal = GaussianSmallExogenousNoTemporalNetwork(history_size=history_length,
                                                                        feature_size=forecast_horizon,
                                                                        hidden_size=hidden_size_no_temporal,
                                                                        hidden_size_history=hidden_size_no_temporal_history,
                                                                        output_size=forecast_horizon)
            optimizer_no_temporal = optim.Adam(model_no_temporal.parameters(), lr=learning_rate)

            model_no_temporal, losses_no_temporal = train_small_exogenous_no_temp_gaussian_model(
                model=model_no_temporal,
                history=history_train,
                target=target_train,
                feature1=feature1_train,
                feature2=feature2_train,
                num_epochs=num_epochs,
                batch_size=batch_size,
                optimizer=optimizer_no_temporal,
                save_path=save_path,
                save_name=f"{descriptor}_{save_no_temporal_name}")
            # Plot Losses
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(num_epochs), losses_no_temporal)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Training Loss")
            ax.set_title("Loss Progression for Exogenous No Temporal Model")
            plt.savefig(f"{save_path}/models/GaussianNoTemporal_loss_{descriptor}.png")
            plt.close(fig)

            mu_test_no_temporal, logvar_test_no_temporal = create_small_exogenous_no_temp_gaussian_predictions(
                model=model_no_temporal,
                history=history_test,
                feature1=feature1_test,
                feature2=feature2_test)
            crps_no_temporal = calculate_crps_gaussian(y_test=target_test,
                                                       mu_test=mu_test_no_temporal,
                                                       logvar_test=logvar_test_no_temporal)
            pd.DataFrame([crps_no_temporal]).to_csv(f"{save_path_crps}GaussianNoTemporal_{descriptor}.csv")

        print(f"Config {key} Finished")

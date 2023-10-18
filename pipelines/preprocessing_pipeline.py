import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_test_train_split(df, test_size):
    return train_test_split(df, test_size=test_size, shuffle=False, random_state=7)


def include_encodings(df):
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24).values
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24).values
    df["day_cos"] = np.cos(2 * np.pi * df.index.day / 7).values
    df["day_sin"] = np.sin(2 * np.pi * df.index.day / 7).values
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12).values
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12).values
    return df


# %%
def prep_data_exogenous(df, target_var, feature1_var, feature2_var, feature3_var, history_length, forecast_horizon,
                        test_size):
    if df.isnull().values.any():
        df = df.interpolate()
    df = include_encodings(df)
    history = []
    target = []
    hour_cos = []
    hour_sin = []
    day_cos = []
    day_sin = []
    month_cos = []
    month_sin = []
    feature1 = []
    feature2 = []
    feature3 = []
    for i in range(len(df) - history_length - forecast_horizon + 1):
        history.append(df[target_var][i:i + history_length])
        target.append(df[target_var][i + history_length:i + history_length + forecast_horizon])
        feature1.append(df[feature1_var][i + history_length:i + history_length + forecast_horizon])
        feature2.append(df[feature2_var][i + history_length:i + history_length + forecast_horizon])
        feature3.append(df[feature3_var][i + history_length:i + history_length + forecast_horizon])
        hour_cos.append(df["hour_cos"][i + history_length:i + history_length + forecast_horizon])
        hour_sin.append(df["hour_sin"][i + history_length:i + history_length + forecast_horizon])
        day_cos.append(df["day_cos"][i + history_length:i + history_length + forecast_horizon])
        day_sin.append(df["day_sin"][i + history_length:i + history_length + forecast_horizon])
        month_cos.append(df["month_cos"][i + history_length:i + history_length + forecast_horizon])
        month_sin.append(df["month_sin"][i + history_length:i + history_length + forecast_horizon])

    # Convert to Tensors
    history = torch.FloatTensor(np.array(history))
    target = torch.FloatTensor(np.array(target))
    feature1 = torch.FloatTensor(np.array(feature1))
    feature2 = torch.FloatTensor(np.array(feature2))
    feature3 = torch.FloatTensor(np.array(feature3))
    hour_cos = torch.FloatTensor(np.array(hour_cos))
    hour_sin = torch.FloatTensor(np.array(hour_sin))
    day_cos = torch.FloatTensor(np.array(day_cos))
    day_sin = torch.FloatTensor(np.array(day_sin))
    month_cos = torch.FloatTensor(np.array(month_cos))
    month_sin = torch.FloatTensor(np.array(month_sin))

    # Create train and test splits
    history_train, history_test = create_test_train_split(history, test_size)
    target_train, target_test = create_test_train_split(target, test_size)
    feature1_train, feature1_test = create_test_train_split(feature1, test_size)
    feature2_train, feature2_test = create_test_train_split(feature2, test_size)
    feature3_train, feature3_test = create_test_train_split(feature3, test_size)
    hour_cos_train, hour_cos_test = create_test_train_split(hour_cos, test_size)
    hour_sin_train, hour_sin_test = create_test_train_split(hour_sin, test_size)
    day_cos_train, day_cos_test = create_test_train_split(day_cos, test_size)
    day_sin_train, day_sin_test = create_test_train_split(day_sin, test_size)
    month_cos_train, month_cos_test = create_test_train_split(month_cos, test_size)
    month_sin_train, month_sin_test = create_test_train_split(month_sin, test_size)

    # Normalise training data
    history_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature1_scaler = StandardScaler()
    feature2_scaler = StandardScaler()
    feature3_scaler = StandardScaler()

    history_train = torch.FloatTensor(history_scaler.fit_transform(history_train))
    history_test = torch.FloatTensor(history_scaler.transform(history_test))
    target_train = torch.FloatTensor(target_scaler.fit_transform(target_train))
    target_test = torch.FloatTensor(target_scaler.transform(target_test))
    feature1_train = torch.FloatTensor(feature1_scaler.fit_transform(feature1_train))
    feature1_test = torch.FloatTensor(feature1_scaler.transform(feature1_test))
    feature2_train = torch.FloatTensor(feature2_scaler.fit_transform(feature2_train))
    feature2_test = torch.FloatTensor(feature2_scaler.transform(feature2_test))
    feature3_train = torch.FloatTensor(feature3_scaler.fit_transform(feature3_train))
    feature3_test = torch.FloatTensor(feature3_scaler.transform(feature3_test))

    return target_train, target_test, history_train, history_test, hour_cos_train, hour_cos_test, hour_sin_train, hour_sin_test, day_cos_train, day_cos_test, day_sin_train, day_sin_test, month_cos_train, month_cos_test, month_sin_train, month_sin_test, feature1_train, feature1_test, feature2_train, feature2_test, feature3_train, feature3_test, target_scaler, history_scaler, feature1_scaler, feature2_scaler, feature3_scaler


def prep_data_exogenous_no_temporal(df, target_var, feature1_var, feature2_var, feature3_var, history_length,
                                    forecast_horizon,
                                    test_size):
    if df.isnull().values.any():
        df = df.interpolate()
    history = []
    target = []
    feature1 = []
    feature2 = []
    feature3 = []
    for i in range(len(df) - history_length - forecast_horizon + 1):
        history.append(df[target_var][i:i + history_length])
        target.append(df[target_var][i + history_length:i + history_length + forecast_horizon])
        feature1.append(df[feature1_var][i + history_length:i + history_length + forecast_horizon])
        feature2.append(df[feature2_var][i + history_length:i + history_length + forecast_horizon])
        feature3.append(df[feature3_var][i + history_length:i + history_length + forecast_horizon])

    # Convert to Tensors
    history = torch.FloatTensor(np.array(history))
    target = torch.FloatTensor(np.array(target))
    feature1 = torch.FloatTensor(np.array(feature1))
    feature2 = torch.FloatTensor(np.array(feature2))
    feature3 = torch.FloatTensor(np.array(feature3))

    # Create train and test splits
    history_train, history_test = create_test_train_split(history, test_size)
    target_train, target_test = create_test_train_split(target, test_size)
    feature1_train, feature1_test = create_test_train_split(feature1, test_size)
    feature2_train, feature2_test = create_test_train_split(feature2, test_size)
    feature3_train, feature3_test = create_test_train_split(feature3, test_size)

    # Normalise training data
    history_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature1_scaler = StandardScaler()
    feature2_scaler = StandardScaler()
    feature3_scaler = StandardScaler()

    history_train = torch.FloatTensor(history_scaler.fit_transform(history_train))
    history_test = torch.FloatTensor(history_scaler.transform(history_test))
    target_train = torch.FloatTensor(target_scaler.fit_transform(target_train))
    target_test = torch.FloatTensor(target_scaler.transform(target_test))
    feature1_train = torch.FloatTensor(feature1_scaler.fit_transform(feature1_train))
    feature1_test = torch.FloatTensor(feature1_scaler.transform(feature1_test))
    feature2_train = torch.FloatTensor(feature2_scaler.fit_transform(feature2_train))
    feature2_test = torch.FloatTensor(feature2_scaler.transform(feature2_test))
    feature3_train = torch.FloatTensor(feature3_scaler.fit_transform(feature3_train))
    feature3_test = torch.FloatTensor(feature3_scaler.transform(feature3_test))

    return target_train, target_test, history_train, history_test, feature1_train, feature1_test, feature2_train, feature2_test, feature3_train, feature3_test, target_scaler, history_scaler, feature1_scaler, feature2_scaler, feature3_scaler


# %%
def prep_data_small_exogenous(df, target_var, feature1_var, feature2_var, history_length, forecast_horizon,
                              test_size):
    if df.isnull().values.any():
        df = df.interpolate()
    df = include_encodings(df)
    history = []
    target = []
    hour_cos = []
    hour_sin = []
    day_cos = []
    day_sin = []
    month_cos = []
    month_sin = []
    feature1 = []
    feature2 = []
    for i in range(len(df) - history_length - forecast_horizon + 1):
        history.append(df[target_var][i:i + history_length])
        target.append(df[target_var][i + history_length:i + history_length + forecast_horizon])
        feature1.append(df[feature1_var][i + history_length:i + history_length + forecast_horizon])
        feature2.append(df[feature2_var][i + history_length:i + history_length + forecast_horizon])
        hour_cos.append(df["hour_cos"][i + history_length:i + history_length + forecast_horizon])
        hour_sin.append(df["hour_sin"][i + history_length:i + history_length + forecast_horizon])
        day_cos.append(df["day_cos"][i + history_length:i + history_length + forecast_horizon])
        day_sin.append(df["day_sin"][i + history_length:i + history_length + forecast_horizon])
        month_cos.append(df["month_cos"][i + history_length:i + history_length + forecast_horizon])
        month_sin.append(df["month_sin"][i + history_length:i + history_length + forecast_horizon])

    # Convert to Tensors
    history = torch.FloatTensor(np.array(history))
    target = torch.FloatTensor(np.array(target))
    feature1 = torch.FloatTensor(np.array(feature1))
    feature2 = torch.FloatTensor(np.array(feature2))
    hour_cos = torch.FloatTensor(np.array(hour_cos))
    hour_sin = torch.FloatTensor(np.array(hour_sin))
    day_cos = torch.FloatTensor(np.array(day_cos))
    day_sin = torch.FloatTensor(np.array(day_sin))
    month_cos = torch.FloatTensor(np.array(month_cos))
    month_sin = torch.FloatTensor(np.array(month_sin))

    # Create train and test splits
    history_train, history_test = create_test_train_split(history, test_size)
    target_train, target_test = create_test_train_split(target, test_size)
    feature1_train, feature1_test = create_test_train_split(feature1, test_size)
    feature2_train, feature2_test = create_test_train_split(feature2, test_size)
    hour_cos_train, hour_cos_test = create_test_train_split(hour_cos, test_size)
    hour_sin_train, hour_sin_test = create_test_train_split(hour_sin, test_size)
    day_cos_train, day_cos_test = create_test_train_split(day_cos, test_size)
    day_sin_train, day_sin_test = create_test_train_split(day_sin, test_size)
    month_cos_train, month_cos_test = create_test_train_split(month_cos, test_size)
    month_sin_train, month_sin_test = create_test_train_split(month_sin, test_size)

    # Normalise training data
    history_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature1_scaler = StandardScaler()
    feature2_scaler = StandardScaler()

    history_train = torch.FloatTensor(history_scaler.fit_transform(history_train))
    history_test = torch.FloatTensor(history_scaler.transform(history_test))
    target_train = torch.FloatTensor(target_scaler.fit_transform(target_train))
    target_test = torch.FloatTensor(target_scaler.transform(target_test))
    feature1_train = torch.FloatTensor(feature1_scaler.fit_transform(feature1_train))
    feature1_test = torch.FloatTensor(feature1_scaler.transform(feature1_test))
    feature2_train = torch.FloatTensor(feature2_scaler.fit_transform(feature2_train))
    feature2_test = torch.FloatTensor(feature2_scaler.transform(feature2_test))

    return target_train, target_test, history_train, history_test, hour_cos_train, hour_cos_test, hour_sin_train, hour_sin_test, day_cos_train, day_cos_test, day_sin_train, day_sin_test, month_cos_train, month_cos_test, month_sin_train, month_sin_test, feature1_train, feature1_test, feature2_train, feature2_test, target_scaler, history_scaler, feature1_scaler, feature2_scaler


def prep_data_small_exogenous_no_temporal(df, target_var, feature1_var, feature2_var, history_length, forecast_horizon,
                                          test_size):
    if df.isnull().values.any():
        df = df.interpolate()
    history = []
    target = []
    feature1 = []
    feature2 = []
    for i in range(len(df) - history_length - forecast_horizon + 1):
        history.append(df[target_var][i:i + history_length])
        target.append(df[target_var][i + history_length:i + history_length + forecast_horizon])
        feature1.append(df[feature1_var][i + history_length:i + history_length + forecast_horizon])
        feature2.append(df[feature2_var][i + history_length:i + history_length + forecast_horizon])

    # Convert to Tensors
    history = torch.FloatTensor(np.array(history))
    target = torch.FloatTensor(np.array(target))
    feature1 = torch.FloatTensor(np.array(feature1))
    feature2 = torch.FloatTensor(np.array(feature2))

    # Create train and test splits
    history_train, history_test = create_test_train_split(history, test_size)
    target_train, target_test = create_test_train_split(target, test_size)
    feature1_train, feature1_test = create_test_train_split(feature1, test_size)
    feature2_train, feature2_test = create_test_train_split(feature2, test_size)

    # Normalise training data
    history_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature1_scaler = StandardScaler()
    feature2_scaler = StandardScaler()

    history_train = torch.FloatTensor(history_scaler.fit_transform(history_train))
    history_test = torch.FloatTensor(history_scaler.transform(history_test))
    target_train = torch.FloatTensor(target_scaler.fit_transform(target_train))
    target_test = torch.FloatTensor(target_scaler.transform(target_test))
    feature1_train = torch.FloatTensor(feature1_scaler.fit_transform(feature1_train))
    feature1_test = torch.FloatTensor(feature1_scaler.transform(feature1_test))
    feature2_train = torch.FloatTensor(feature2_scaler.fit_transform(feature2_train))
    feature2_test = torch.FloatTensor(feature2_scaler.transform(feature2_test))

    return target_train, target_test, history_train, history_test, feature1_train, feature1_test, feature2_train, feature2_test, target_scaler, history_scaler, feature1_scaler, feature2_scaler


def prep_data_temporal(df, target_var, history_length, forecast_horizon,
                       test_size):
    if df.isnull().values.any():
        df = df.interpolate()
    df = include_encodings(df)
    history = []
    target = []
    hour_cos = []
    hour_sin = []
    day_cos = []
    day_sin = []
    month_cos = []
    month_sin = []
    for i in range(len(df) - history_length - forecast_horizon + 1):
        history.append(df[target_var][i:i + history_length])
        target.append(df[target_var][i + history_length:i + history_length + forecast_horizon])
        hour_cos.append(df["hour_cos"][i + history_length:i + history_length + forecast_horizon])
        hour_sin.append(df["hour_sin"][i + history_length:i + history_length + forecast_horizon])
        day_cos.append(df["day_cos"][i + history_length:i + history_length + forecast_horizon])
        day_sin.append(df["day_sin"][i + history_length:i + history_length + forecast_horizon])
        month_cos.append(df["month_cos"][i + history_length:i + history_length + forecast_horizon])
        month_sin.append(df["month_sin"][i + history_length:i + history_length + forecast_horizon])

    # Convert to Tensors
    history = torch.FloatTensor(np.array(history))
    target = torch.FloatTensor(np.array(target))
    hour_cos = torch.FloatTensor(np.array(hour_cos))
    hour_sin = torch.FloatTensor(np.array(hour_sin))
    day_cos = torch.FloatTensor(np.array(day_cos))
    day_sin = torch.FloatTensor(np.array(day_sin))
    month_cos = torch.FloatTensor(np.array(month_cos))
    month_sin = torch.FloatTensor(np.array(month_sin))

    # Create train and test splits
    history_train, history_test = create_test_train_split(history, test_size)
    target_train, target_test = create_test_train_split(target, test_size)
    hour_cos_train, hour_cos_test = create_test_train_split(hour_cos, test_size)
    hour_sin_train, hour_sin_test = create_test_train_split(hour_sin, test_size)
    day_cos_train, day_cos_test = create_test_train_split(day_cos, test_size)
    day_sin_train, day_sin_test = create_test_train_split(day_sin, test_size)
    month_cos_train, month_cos_test = create_test_train_split(month_cos, test_size)
    month_sin_train, month_sin_test = create_test_train_split(month_sin, test_size)

    # Normalise training data
    history_scaler = StandardScaler()
    target_scaler = StandardScaler()

    history_train = torch.FloatTensor(history_scaler.fit_transform(history_train))
    history_test = torch.FloatTensor(history_scaler.transform(history_test))
    target_train = torch.FloatTensor(target_scaler.fit_transform(target_train))
    target_test = torch.FloatTensor(target_scaler.transform(target_test))

    return target_train, target_test, history_train, history_test, hour_cos_train, hour_cos_test, hour_sin_train, hour_sin_test, day_cos_train, day_cos_test, day_sin_train, day_sin_test, month_cos_train, month_cos_test, month_sin_train, month_sin_test, target_scaler, history_scaler


def prep_data_history(df, target_var, history_length, forecast_horizon,
                      test_size):
    if df.isnull().values.any():
        df = df.interpolate()
    history = []
    target = []
    for i in range(len(df) - history_length - forecast_horizon + 1):
        history.append(df[target_var][i:i + history_length])
        target.append(df[target_var][i + history_length:i + history_length + forecast_horizon])

    # Convert to Tensors
    history = torch.FloatTensor(np.array(history))
    target = torch.FloatTensor(np.array(target))

    # Create train and test splits
    history_train, history_test = create_test_train_split(history, test_size)
    target_train, target_test = create_test_train_split(target, test_size)

    # Normalise training data
    history_scaler = StandardScaler()
    target_scaler = StandardScaler()

    history_train = torch.FloatTensor(history_scaler.fit_transform(history_train))
    history_test = torch.FloatTensor(history_scaler.transform(history_test))
    target_train = torch.FloatTensor(target_scaler.fit_transform(target_train))
    target_test = torch.FloatTensor(target_scaler.transform(target_test))

    return target_train, target_test, history_train, history_test, target_scaler, history_scaler

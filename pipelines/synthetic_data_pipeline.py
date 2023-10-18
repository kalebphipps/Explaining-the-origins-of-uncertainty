import numpy as np


def create_synthetic_data(size, base_amplitude, base_frequency, base_noise_scale, base_noise_amplitude,
                          number_of_trend_events, trend_parameters, number_of_cosine_events,
                          cosine_frequency_parameters, cosine_amplitude_parameters, number_of_increased_noise_events,
                          increased_noise_parameters, noise_type, problem_lower=None, problem_upper=None):
    time = np.linspace(0, int(size), int(size))

    if problem_lower is None:
        problem_lower = int(int(size) / 50)
    if problem_upper is None:
        problem_upper = int(int(size) / 10)

    # Generate the synthetic time series
    if noise_type == "additive":
        synthetic_series = (base_amplitude * np.sin(base_frequency * 2 * np.pi * time) +
                            base_noise_amplitude * np.random.normal(loc=0,
                                                                    scale=base_noise_scale,
                                                                    size=size))
    else:
        synthetic_series = (base_amplitude * np.sin(base_frequency * 2 * np.pi * time) *
                            0.1 * np.random.normal(loc=0,
                                                   scale=base_noise_scale,
                                                   size=size))

    # Introduce occasional trend
    trend_event_indices = np.random.choice(np.arange(int(size)), size=number_of_trend_events, replace=False)
    trend_time_series = np.zeros(len(time))
    for idx in trend_event_indices:
        problem_length = int(np.random.uniform(low=problem_lower, high=problem_upper))
        trend_slope = np.random.uniform(low=trend_parameters[0], high=trend_parameters[1])
        if idx + problem_length < len(time):
            multipliers = np.arange(start=0, stop=problem_length, step=1)
            trend_time_series[idx:idx + problem_length] = (trend_slope * multipliers)
        else:
            remainder = len(time) - idx
            multipliers = np.arange(start=0, stop=remainder, step=1)
            trend_time_series[idx:] = (trend_slope * multipliers)

    # Introduce occasional cosine events
    cosine_event_indices = np.random.choice(np.arange(int(size)), size=number_of_cosine_events, replace=False)
    cosine_time_series = np.zeros(len(time))
    for idx in cosine_event_indices:
        problem_length = int(np.random.uniform(low=problem_lower, high=problem_upper))
        cosine_freq = np.random.uniform(low=cosine_frequency_parameters[0], high=cosine_frequency_parameters[1])
        cosine_amplitude = np.random.uniform(low=cosine_amplitude_parameters[0], high=cosine_amplitude_parameters[1])
        if idx + problem_length < len(time):
            cosine_time_series[idx:idx + problem_length] = cosine_amplitude * np.cos(
                cosine_freq * 2 * np.pi * time[idx:idx + problem_length])
        else:
            cosine_time_series[idx:] = cosine_amplitude * np.cos(cosine_freq * time[idx:])

    # Introduce occasional increased noise events
    increased_noise_event_indices = np.random.choice(np.arange(int(size)), size=number_of_increased_noise_events,
                                                     replace=False)
    increased_noise_time_series = np.zeros(len(time))
    for idx in increased_noise_event_indices:
        problem_length = int(np.random.uniform(low=problem_lower, high=problem_upper))
        random_scale = np.random.uniform(low=increased_noise_parameters[0], high=increased_noise_parameters[1])
        if idx + problem_length < len(time):
            increased_noise_time_series[idx:idx + problem_length] = np.random.normal(loc=0, scale=random_scale,
                                                                                     size=problem_length)
        else:
            remainder = len(time) - idx
            increased_noise_time_series[idx:] = np.random.normal(loc=0, scale=random_scale, size=remainder)

    synthetic_series = synthetic_series + trend_time_series + cosine_time_series + increased_noise_time_series

    return synthetic_series, trend_time_series, cosine_time_series, increased_noise_time_series


def create_synthetic_data_with_big_noise(size, base_amplitude, base_frequency, base_noise_scale, base_noise_amplitude,
                                         number_of_big_noise_events, big_noise_amplitude):
    time = np.linspace(0, int(size), int(size))

    # Generate the synthetic time series
    synthetic_series = (base_amplitude * np.sin(base_frequency * 2 * np.pi * time) +
                        base_noise_amplitude * np.random.normal(loc=0,
                                                                scale=base_noise_scale,
                                                                size=size)
                        )

    # Add extra noise at certain points
    big_noise_ts = np.zeros(len(time))
    noise_event_indices = np.random.choice(np.arange(int(size)), size=number_of_big_noise_events, replace=False)
    big_noise_ts[noise_event_indices] = big_noise_amplitude * np.random.normal(0, 1, len(noise_event_indices))
    synthetic_series = synthetic_series + big_noise_ts

    return synthetic_series, big_noise_ts

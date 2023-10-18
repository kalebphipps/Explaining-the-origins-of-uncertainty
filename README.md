# Explaining-the-origins-of-uncertainty

This repository contains code to replicate the results from Chapter 6 of the Dissertation "Quantifying and Interpreting
Uncertainty in Time Series Forecasting" by Kaleb Phipps

## Repository Structure

This repository is structured in a few key folders:

- `configs`: This folder contains the configs for the models trained for each of the data sets.
- `data`: This folder contains the data used for the analyses in the chapter.
- `explanations_analysis`: This folder contains multiple Jupyter Notebooks which create the explanations shown in Chapter 6.
- `metrics`: This folder contains the evaluation metrics implemented in our code.
- `models`: This folder contains the torch based neural network models used to generate probabilistic forecasts.
- `pipelines`: This folder contains the pipelines which are often used for preprocessing or synthetic data creation.
- `scripts`: This folder contains the scripts which need to be run to generate probabilistic forecasts.


## Installation

Before the proposed approach can be, you need to
prepare a Python environment.

### 1. Setup Python Environment

Perform the following steps:

- Set up a virtual environment of Python 3.10 using e.g. venv (`python3.10 -m venv venv`) or Anaconda (`conda create -n env_name python=3.10`).
- Possibly install pip via `conda install pip`.
- Install the dependencies with `pip install -r requirements.txt`.

### 2. Download Data (optional)

We provide the open source data to replicate our results in the folder __data__. If you want to apply our approach to further data you will need to download this yourself.


## Execution
If you are interested in running code, you should navigate to the appropriate script in the `scripts` folder and run
the respective script from there. Running a script will generate trained models and probabilistic forecasts in a `Results`
folder. To generate the explanations navigate to `explanations_analysis` and run the appropriate notebook located there.

If you are interested in applying our method to your own data, you will need to create a new script. You can use the
existing scripts in the `scripts` folder as orientation for any pipeline you create.


## Funding

This project is supported by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI and by the
Helmholtz Association under the Program “Energy System Design”.

## License

This code is licensed under the [MIT License](LICENSE).


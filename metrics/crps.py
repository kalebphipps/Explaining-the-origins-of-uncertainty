import numpy as np
import properscoring as ps


def calculate_crps_gaussian(y_test, mu_test, logvar_test):
    return np.mean(ps.crps_gaussian(x=y_test.detach().numpy(),
                                    mu=mu_test.detach().numpy(),
                                    sig=logvar_test.exp().sqrt().detach().numpy()))

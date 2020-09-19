import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
#%% Define distributions
n_trials = 10000
mu_ui, sigma_ui = 850, 0.65
mu_ti, sigma_ti = 1050, 0.5
ui_trials = st.lognorm.rvs(sigma_ui, scale=mu_ui, size=n_trials)
ti_trials = st.lognorm.rvs(sigma_ti, scale=mu_ti, size=n_trials)
sum_trials = ui_trials + ti_trials
quantiles = np.quantile(sum_trials, [0.1, 0.5, 0.9])
plt.hist(sum_trials, bins=50)
plt.show()
print(quantiles)

import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

percentiles = [0.1, 0.5, 0.9]
cumulatives = [300, 963, 1538]

dist = st.lognorm

mu = 850
sigma = 0.65
n_trials = 1000

values = dist.rvs(sigma, scale=mu, size=n_trials)

min = dist.ppf(0.01, sigma, scale=mu)
max = dist.ppf(0.99, sigma, scale=mu)
n_points = 100
x = np.linspace(min, max, n_points)

#%%
# Plot the fitted lognormal distribution
fig_0, ax_0 = plt.subplots(1, 1)
ax_0.plot(x, dist.cdf(x, sigma, scale=mu), 'b-', lw=1, alpha=0.6, label='lognorm cdf')
ax_0.scatter(cumulatives, percentiles)
plt.show()
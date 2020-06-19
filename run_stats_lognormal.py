import scipy.stats as st
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Percentile = namedtuple('Percentile', 'p value')

p1 = Percentile(0.5, 500)
p2 = Percentile(0.9, 1500)
p_find = 0.1

z1 = st.norm.ppf(p1.p)
z2 = st.norm.ppf(p2.p)

A = np.array([[1, z1], [1, z2]])
B = np.array(np.log([p1.value, p2.value]))
result = np.linalg.solve(A, B)
mu = result[0]
sigma = result[1]

print(f'mu: {mu}')
print(f'sigma: {sigma}')

# create a lognormal distribution using the previous results

M = np.exp(mu)
s = sigma
value = st.lognorm.ppf(p_find, s, scale=M)
print(value)

# Plot the fitted lognormal distribution
fig_0, ax_0 = plt.subplots(1, 1)

min = st.lognorm.ppf(0.01, s, scale=M)
max = st.lognorm.ppf(0.99, s, scale=M)
n_points = 100
x = np.linspace(min, max, n_points)

ax_0.plot(x, st.lognorm.cdf(x, s, scale=M), 'b-', lw=1, alpha=0.6, label='lognorm cdf')
x_scatter = [p1.value, p2.value, value]
y_scatter = [p1.p, p2.p, p_find]
colors = ['b', 'b', 'r']
ax_0.scatter(x_scatter, y_scatter, c=colors)
ax_0.set_ylabel('CDF')
ax_0.set_title(f'Log Normal Distribution - Find {p_find} quantile given {(p1.p, p2.p)}')
# Plot the pdf
ax_1 = ax_0.twinx()
ax_1.plot(x, st.lognorm.pdf(x, s, scale=M), 'b-', lw=1, alpha=0.6, label='lognorm pdf')
ax_1.set_ylabel('PDF')
plt.show()
# Create lognormal pdf for each activity
n_activities = 5
data = {}
n_trials = 5000
for n in range(n_activities):
    data[n] = st.lognorm.rvs(s, scale=M, size=n_trials)

df_data = pd.DataFrame(data)
df_data['sum'] = df_data.sum(axis=1)

fig_1, ax_2 = plt.subplots(1, 1)
sns.distplot(df_data['sum'], ax=ax_2)
plt.show()

quantiles = df_data['sum'].quantile([0.1, 0.5, 0.9])
type_well_cum = np.array([value, p1.value, p2.value])
#quantiles['tw_cum'] = type_well_cum
#quantiles['ratios'] = quantiles
cum_per_well = quantiles / n_activities
ratios = cum_per_well / type_well_cum
print(quantiles)
print(ratios)
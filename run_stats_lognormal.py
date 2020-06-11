import scipy.stats as st
from collections import namedtuple
import numpy as np

Percentile = namedtuple('Percentile', 'p value')

p1 = Percentile(50, 500000)
p2 = Percentile(90, 1500000)

z1 = st.norm.ppf(p1.p / 100)
z2 = st.norm.ppf(p2.p / 100)

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
value = st.lognorm.ppf(0.1, s, scale=M)
print(value)


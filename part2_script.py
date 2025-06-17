import numpy as np
import matplotlib.pyplot as plt
from my_functions import *
import scipy.stats as stats
from scipy.linalg import expm

n = 1_000
Q = np.array([
    [-0.0085,  0.0050,  0.0025, 0.0,     0.0010],
    [ 0.0,    -0.0140,  0.0050, 0.0040,  0.0050],
    [ 0.0,     0.0,    -0.0080, 0.0030,  0.0050],
    [ 0.0,     0.0,     0.0,   -0.0090,  0.0090],
    [ 0.0,     0.0,     0.0,    0.0,     0.0   ]
])

class woman:
	def __init__(self, test_func):
		self.test_func = test_func
		self.state = 0
		self.time_of_death = 0
		self.indicator = 0

	def update(self, Q):
		rate = -Q[self.state, self.state]
		advance_time = stats.expon.rvs(scale=1/rate) # time before changing state
		self.time_of_death += advance_time
		probs = - Q[self.state, self.state+1:] / Q[self.state, self.state]
		states = np.arange(1, len(probs) + 1)
		self.state += np.random.choice(states, p=probs) # update state
		self.indicator = self.test_func(self)



def simulate(n, Q, test_func=lambda x: 0):
	system = [woman(test_func) for _ in range(n)]

	while any(w.state < 4 for w in system):
		for w in system:
			if w.state == 4: # death state
				continue

			w.update(Q)

	return system

def test_distant(w:woman):
	if w.state == 2 or w.state == 3:
		if w.time_of_death < 30.5:
			return 1
		elif w.indicator == 0:
			return 2
		else:
			return w.indicator
	else:
		return w.indicator


#------- Task 7 --------
np.random.seed(42)
data = simulate(n, Q, test_func=test_distant)
lifetimes = [w.time_of_death for w in data]

#make_discrete_hist(lifetimes, "Simulation of lifetimes", "Months", "Number of patients")

confidence_interval(lifetimes)

def count_prop():
	data = simulate(10*n, test_func=test_distant)
	indicators = np.array([w.indicator for w in data])
	#print(indicators)
	indicators = np.reshape(indicators,(10,n))
	counts = [np.sum([values == 2])/n for values in indicators]
	print(counts)
	confidence_interval(counts)

# count_prop()
# ----------------------

#------- Task 8 --------
def task_8():
	p0 = np.array([1,0,0,0])  
	Q_s = Q[0:-1,0:-1]
	length = np.shape(Q_s)[0] # the length of the vector of ones
	Ft = lambda t: 1-p0 @ expm(Q_s*t) @ np.ones(n)
	def Ft_vec(t_vals):
		return np.array([1 - p0 @ expm(Q_s * t) @ np.ones(length) for t in t_vals])

	tries = 1000
	all_ps = np.zeros(tries) # array for the K-S test p values
	for i in range(tries):
		lifetimes = [w.time_of_death for w in simulate(tries, Q)]
		_,p = stats.kstest(lifetimes, Ft_vec)
		all_ps[i] = p

	make_discrete_hist(all_ps, "distribution of K-S p-values")
# ----------------------

#------- Task 9 --------
new_Q = np.array([
    [-0.00475,  0.0025,  0.00125, 0.0,     0.001],
    [ 0.0,    -0.007,   0.0,     0.002,   0.005],
    [ 0.0,     0.0,    -0.0080,  0.003,   0.005],
    [ 0.0,     0.0,     0.0,    -0.0090,  0.0090],
    [ 0.0,     0.0,     0.0,     0.0,     0.0   ]
])

new_data = simulate(n, new_Q)

new_lifetimes = [w.time_of_death for w in new_data]

x1, s1 = empirical_survival_function(lifetimes)
x2, s2 = empirical_survival_function(new_lifetimes)

# Plot
plt.figure(figsize=(8, 5))
plt.step(x1, s1, where='post', label='Previous model')
plt.step(x2, s2, where='post', label='After treatment')

plt.title('Empirical Survival Functions')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("survival.png")
plt.savefig("survival.pdf")
#plt.show()
# ----------------------

#------- Task 10 --------
# this one is mostly from chatgpt

from lifelines.statistics import logrank_test
events = np.ones_like(lifetimes, dtype=int)
# Perform the logrank test
result = logrank_test(lifetimes, new_lifetimes, event_observed_A=events, event_observed_B=events)
print("Result of logrank test:")
print(f"Test statistic: {result.test_statistic:.4f}")
print(f"P-value: {result.p_value:.4e}")

# ----------------------

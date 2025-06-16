import numpy as np
from scipy.stats import expon

Q = np.array([[-0.0085,0.005,0.0025,0,0.001],[0,-0.014,0.005,0.004,0.005],[0,0,-0.008,0.003,0.005],[0,0,0,-0.009,0.009],[0,0,0,0,0]])

class CTMC:
    def __init__(self,n : int,Q):
        self.lifetime = np.zeros(n,dtype="float32")
        self.alive = np.ones(n,dtype="bool")
        self.Q = Q
        self.currentState = np.zeros(n,dtype="int16")
        self.N = np.shape(Q)[0]-1
        self.distant_count = 0
        self.statebefore = self.currentState.copy()
        
    def eval_exp(self,rate : float) -> float:
        return expon.rvs(scale=1/rate)
    
    def sim(self):
        while any(self.alive):
            self.statebefore = self.currentState
            current_states = self.currentState[self.alive]
            probs = -self.Q[current_states,:]/self.Q[current_states,current_states,None]
            time_to_leave = self.eval_exp(-self.Q[current_states,current_states])
            self.lifetime[self.alive] += time_to_leave
            next = np.array([np.random.choice(np.arange(current_states[i]+1,self.N+1),p=np.abs(probs[i,current_states[i]+1:])) for i in range(len(current_states))])
            valid_transitions = np.isin(self.currentState[self.alive],[0,1]) & np.isin(next,[2,3])
            self.currentState[self.alive] = next
            self.distant_count += sum(1 for i in self.lifetime[self.alive][valid_transitions] if i >= 30.5)
            self.alive[self.alive] = next != self.N
            
            
            
n = 1000
women = CTMC(n,Q)
women.sim()
women.alive
women.lifetime

from scipy.stats import t,chi2
#Summary statistics:
alpha = 0.05
mu = np.mean(women.lifetime)
var = np.var(women.lifetime)
sig = np.std(women.lifetime)
print(mu,var,sig)    
k = n
t_crit = t.ppf(1 - alpha / 2, df=k - 1)
margin_mu = t_crit * sig / np.sqrt(k)
mu_l_confs = mu - margin_mu
mu_u_confs = mu + margin_mu

# CI for variance
chi2_low = chi2.ppf(alpha / 2, df=k - 1)
chi2_high = chi2.ppf(1 - alpha / 2, df=k - 1)
var_l_confs = (k - 1) * var / chi2_high
var_u_confs = (k - 1) * var / chi2_low
#Mean confidence interval:
#Todo : 100 repetitions
print(mu_l_confs, mu, mu_u_confs)
#Variance confidence intervals
print(var_l_confs, var, var_u_confs)

#proportion of women which end in either state 3 or 4
print(women.distant_count/1000)


import matplotlib.pyplot as plt


            
from scipy.linalg import expm  
p0 = np.array([1,0,0,0])  
Q_s = Q[0:-1,0:-1]
n_F = np.shape(Q_s)[0]
Ft = lambda t: 1-p0 @ expm(Q_s*t) @ np.ones(n)
def Ft_vec(t_vals):
    return np.array([1 - p0 @ expm(Q_s * t) @ np.ones(n_F) for t in t_vals])

plt.hist(women.lifetime,bins=100,density=False,color="green",edgecolor="black")
plt.show()

#plt.hist(women.lifetime,density=True,alpha=0.7,bins=100,edgecolor="black",color="green",label="MC_sim")
x = np.linspace(0, max(women.lifetime), 10000)
pmf_vals = [Ft(x) for x in x]
plt.plot(x, pmf_vals, 'r-', lw=2, label='True distribution')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('Women cancer sim')
plt.legend()
plt.grid(True)
plt.show()       



from scipy.stats import kstest

D, p_value = kstest(women.lifetime, Ft_vec)
tries = 1000
n = 1000
all_ps = np.zeros(tries)
for i in range(tries):
    women = CTMC(n,Q)
    women.sim()
    _,p = kstest(women.lifetime, Ft_vec)
    all_ps[i] = p
    print(i)

plt.hist(all_ps,bins=100,density=False,color="green",edgecolor="black")
plt.xlabel('Value')
plt.ylabel('Count')
plt.title(f'P-values for {n} runs')
plt.grid(True)
plt.show() 



#Kaplan meier:


Q_k = np.array([[-0.00475,0.0025,0.00125,0,0.001],[0,-0.007,0.0,0.002,0.005],[0,0,-0.008,0.003,0.005],[0,0,0,-0.009,0.009],[0,0,0,0,0]])

def surivival_f(lifetimes,N,t):
    return np.array([np.sum((lifetimes > ti)/N for ti in t)])

def kaplan_meier(L, t_vals):
    n = len(L)
    return np.array([(n - np.sum(L < ti)) / n for ti in t_vals])

women = CTMC(n,Q)
women.sim()

women_T = CTMC(n,Q_k)
women_T.sim()


t = np.arange(int(max([max(women.lifetime),max(women_T.lifetime)])))

KM = kaplan_meier(women.lifetime,t)
KM_T = kaplan_meier(women_T.lifetime,t)

plt.stairs(KM,color="green",label="Kaplan-meier estimate Without treatment")
plt.stairs(KM_T,color="red",label="Kaplan-meier estimate With treatment")
plt.xlabel('Month (t)')
plt.ylabel('Proportion of women alive at time t')
plt.title(f'Kaplan meier estimate')
plt.grid(True)
plt.legend()
plt.show()

np.abs(np.diag(Q))
np.abs(np.diag(Q_k))

from scipy.stats import logrank

logrank(KM,KM_T)
#There is a statistic significant difference which we see in the p-value of 2.07e-10 hence the survival functions are not drawn from the same
#Distribution

#Task 11
#???




# Part
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from scipy.special import factorial, comb
from tqdm import tqdm

P = np.array([
    [0.9915, 0.005,  0.0025, 0,     0.001],
    [0,      0.986,  0.005,  0.004, 0.005],
    [0,      0,      0.992,  0.003, 0.005],
    [0,      0,      0,      0.991, 0.009],
    [0,      0,      0,      0,     1    ]
])

#%% Task 1-3
def task1to3(P):
    N_states = len(P)
    N_sim = 1000
    states = np.arange(N_states)
    time_until_death = np.zeros(N_sim)
    death_state = N_states-1
    p_state2_hit = 0
    t = 120 # time at which to perform a check of distribution
    freq_at_t = np.zeros(N_states)
    for i in range(N_sim):
        state = 0
        time = 0
        state2_hit = False
        state_at_t = N_states-1 # always ends up in the last state
        while state != death_state:
            time += 1
            state = np.random.choice(states, p=P[state])
            
            if state == 1:
                state2_hit = True
            if time == t:
                state_at_t = state
            
        freq_at_t[state_at_t] += 1
        p_state2_hit += int(state2_hit)
        time_until_death[i] = time
    
    
    print('\n=== TASK 1\n')
    plt.figure(figsize=(10, 6))
    plt.hist(time_until_death, bins=30, edgecolor='black', alpha=0.75)
    plt.title("Histogram of lifetime distribution after surgery", fontsize=14)
    plt.xlabel("Months", fontsize=12)
    plt.ylabel("Number of patients", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("lifetime_histogram.pdf")
    plt.show()
    print(np.mean(time_until_death),np.std(time_until_death))
    p_state2_hit /= N_sim
    print(f'p_state2_hit={p_state2_hit}')
    
    print('\n=== TASK 2\n')
    dist_at_t = freq_at_t/N_sim
    p_t = np.linalg.matrix_power(P, n=t)[0] # true distribution
    print(f'{t=}')
    print(f'{dist_at_t=} (empirical)')
    print(f'{p_t=} (analytical)')
    
    chi2_stat,chi2_pval = sps.chisquare(f_obs=freq_at_t, f_exp=N_sim*p_t)
    print(f'\nchisquare test:')
    print(f'{chi2_pval=:.3f}')
    
    print('\n=== TASK 3\n')
    t_max = time_until_death.max()
    pi = np.array([1,0,0,0])
    P_s = P[:N_states-1,:N_states-1]
    p_s = P[:N_states-1,-1]
    
    p_T = np.zeros(int(t_max))
    term = pi
    for i in range(1,len(p_T)):
        term = term @ P_s
        p_T[i] = term @ p_s
        
    E_T = np.sum(pi @ np.linalg.inv(np.eye(len(P_s)) - P_s))
    
    plt.hist(time_until_death, bins=40, edgecolor='black', alpha=0.75, density=True, label='Empirical PMF')
    plt.plot(p_T, label='Analytical PMF')
    plt.xlabel('Months')
    plt.grid(True)
    plt.title('Task 3: Lifetime distribution')
    plt.legend()
    plt.savefig("lifetime_vs_analytic.pdf")
    plt.show()
    
    ### kstest
    cdf_T = np.cumsum(p_T)

    # Define analytical CDF function
    x_vals = np.arange(len(cdf_T))
    def analytical_cdf(x):
        return np.interp(x, x_vals, cdf_T, left=0.0, right=1.0)

    # Perform one-sample K-S test
    ks_stat, ks_p = sps.kstest(rvs=time_until_death, cdf=analytical_cdf)

    print(f"KS test statistic: {ks_stat:.4f}")
    print(f"KS test p-value: {ks_p:.4f}")
    ###
    
    print(f'Empirical lifetime mean: {time_until_death.mean()}')
    print(f'Analytical lifetime mean: {E_T}')

def task2_pval(P):
    N_states = len(P)
    N_sim = 1000
    states = np.arange(N_states)
    t = 120 # time at which to perform a check of distribution
    
    def sim():
        time_until_death = np.zeros(N_sim)
        death_state = N_states-1
        freq_at_t = np.zeros(N_states)
        for i in range(N_sim):
            state = 0
            time = 0
            state_at_t = N_states-1 # always ends up in the last state
            while state != death_state:
                time += 1
                state = np.random.choice(states, p=P[state])
                
                if time == t:
                    state_at_t = state
                
            freq_at_t[state_at_t] += 1
            time_until_death[i] = time
        return time_until_death,freq_at_t
    
    p_t = np.linalg.matrix_power(P, n=t)[0] # true distribution
    
    rep = 20
    pvals = np.zeros(rep)
    for i in tqdm(range(rep)):
        time_until_death,freq_at_t = sim()
        chi2_stat,chi2_pval = sps.chisquare(f_obs=freq_at_t, f_exp=N_sim*p_t)
        pvals[i] = chi2_pval
    
    plt.hist(pvals)
    plt.title('Task 2: Histogram over chi2 p-values')
    plt.show()
    
    
#%% Task 4
def task4(P):
    """
    only those who reach the two conditions will be counted
    reappeared: state > 0
    """
    print('\n=== TASK 4\n')
    
    N_states = len(P)
    N_desired = 1000
    states = np.arange(N_states)
    time_until_death = np.zeros(N_desired)
    death_state = N_states-1
    N_current = 0
    with tqdm(total=N_desired, desc="Task 4") as pbar:
        while N_current < N_desired:
            state = 0
            time = 0
            while True:
                time += 1
                state = np.random.choice(states, p=P[state])
                
                if time == 12:
                    event_reappeared = 0 < state and state < death_state
                    if not event_reappeared:
                        break
                if state == death_state:
                    break
            
            event_survived = time > 12
            if not (event_survived and event_reappeared):
                continue
            
            time_until_death[N_current] = time
            N_current += 1

            pbar.update(1)
    
    print(f'Expected lifetime: {time_until_death.mean()}')
        

#%% Task 5
def confidence_interval(data, confidence=0.95, use_t=True):
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    sem = sps.sem(data)  # Standard error of the mean
    if use_t:
        # Use Student's t-distribution
        margin = sem * sps.t.ppf((1 + confidence) / 2., n - 1)
    else:
        # Use normal distribution
        margin = sem * sps.norm.ppf((1 + confidence) / 2.)
    return mean - margin, mean + margin

def task5(P):
    print('\n=== TASK 5\n')
    
    N_states = len(P)
    N_sim = 200
    death_state = N_states-1
    
    def sim():
        states = np.arange(N_states)
        time_until_death = np.zeros(N_sim)
        die_early = np.zeros(N_sim,dtype=bool)
        for i in range(N_sim):
            state = 0
            time = 0
            event_die_early = True
            while state != death_state:
                if time == 351:
                    event_die_early = False
                time += 1
                state = np.random.choice(states, p=P[state])
                
            die_early[i] = event_die_early
            time_until_death[i] = time
        
        frac = die_early.mean()
        avg_lifetime = time_until_death.mean()
        return frac, avg_lifetime
    
    # rep = 100
    rep = 20
    x = np.zeros(rep)
    y = np.zeros(rep)
    for i in tqdm(range(rep)):
        x[i],y[i] = sim()
    
    # calculate true mean lifetime
    pi = np.array([1,0,0,0])
    P_s = P[:N_states-1,:N_states-1]
    p_s = P[:N_states-1,-1]
        
    E_T = np.sum(pi @ np.linalg.inv(np.eye(len(P_s)) - P_s))
    
    #
    c = - np.cov(x,y)[0,1] / np.var(y)
    mu_y = E_T
    z = x + c*(y-mu_y)
    
    ci_x = confidence_interval(x)
    ci_z = confidence_interval(z)
    print(f'Mean fraction: {x.mean()}')
    print(f'Crude CI95: [{ci_x[0]:.3f},{ci_x[1]:.3f}]')
    print(f'Crude CI95: [{ci_z[0]:.3f},{ci_z[1]:.3f}]')
    
    width_ci_x = ci_x[1]-ci_x[0]
    width_ci_z = ci_z[1]-ci_z[0]
    
    print(f'width_ci_z/width_ci_x={width_ci_z/width_ci_x:.3f}')
    
    
### TASK 6

# The Markov property might only be an approximation since
# if one has been in one state for long, the probability
# might be different

# Transition probabilities are constant over time, which might
# not be a good approximation

# Discrete time steps, but time is continuous


if __name__ == '__main__':
    # task1to3(P)
    # task2_pval(P)
    # task4(P)
    # task5(P)
    pass
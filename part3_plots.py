import numpy as np
#from part2 import CTMC
#Q = np.array([[-0.0085,0.005,0.0025,0,0.001],[0,-0.014,0.005,0.004,0.005],[0,0,-0.008,0.003,0.005],[0,0,0,-0.009,0.009],[0,0,0,0,0]])
n = 1000
Q = np.array([[-0.00849296, 0.00453524, 0.0027772,0,0.00118052],[0,-0.01305882,0.00479312,0.00376603,0.00449967],[0,0,-0.00846714,0.00304364,0.00542351],[0,0,0,-0.00933217,0.00933217],[0,0,0,0,0]])
from scipy.stats import expon
class CTMC_part3:
    def __init__(self,n : int,Q, step):
        self.lifetime = np.zeros(n,dtype="float32")
        self.alive = np.ones(n,dtype="bool")
        self.Q = Q
        self.currentState = np.zeros(n,dtype="int16")
        self.N = np.shape(Q)[0]-1
        self.distant_count = 0
        self.statebefore = self.currentState.copy()
        self.n = n
        self.times = [[] for _ in range(n)]
        self.timeseries = [[] for _ in range(n)]
        self.stepsize = step
        self.realtimeseries = [[] for _ in range(n)]
        
    def eval_exp(self,rate : float) -> float:
        return expon.rvs(scale=1/rate)
    def update_ts(self):
        idx_alive = np.where(self.alive)[0]
        for i, idx in enumerate(idx_alive):
            self.timeseries[idx].append(self.currentState[idx])
            self.times[idx].append(self.lifetime[idx])
    def sim(self):
        self.update_ts()
        while any(self.alive):
            current_states = self.currentState[self.alive]
            probs = -self.Q[current_states,:]/self.Q[current_states,current_states,None]
            time_to_leave = self.eval_exp(-self.Q[current_states,current_states])
            self.lifetime[self.alive] += time_to_leave
            next = np.array([np.random.choice(np.arange(current_states[i]+1,self.N+1),p=probs[i, current_states[i]+1:] / np.sum(probs[i, current_states[i]+1:])) for i in range(len(current_states))])
            self.currentState[self.alive] = next
            self.update_ts()
            self.alive[self.alive] = next != self.N
        self.create_48_ts()
        
    def create_48_ts(self):
        sampled_ts = [[] for _ in range(n)]
        for i in range(n):
            st = self.times[i]
            ts = self.timeseries[i]
            if not ts:
                continue
            t_vals = [x for x in st]
            s_vals = [x for x in ts]
            next_sample_time = 0
            idx = 0
            while next_sample_time <= t_vals[-1]:
                while idx + 1 < len(t_vals) and t_vals[idx + 1] <= next_sample_time:
                    idx += 1
                sampled_ts[i].append(s_vals[idx])
                next_sample_time += self.stepsize
            sampled_ts[i].append(4)
            self.realtimeseries[i] = self.timeseries[i].copy()
            self.timeseries[i] = sampled_ts[i]
    def initialize_Q0(self):
        delta = 48
        n = self.N+1
        C  = np.zeros(n)
        Nhat = np.zeros((n,n))
        for ts in self.timeseries:
            for a,b in zip(ts, ts[1:]):
                C[a] += 1
                Nhat[a,b] += 1
        Phat = (Nhat.T / C).T 
        Q0 = np.zeros((n,n))
        for i in range(n):
            if Phat[i,i] < 1 and not np.isnan(Phat[i,i]):
                qi = -np.log(max(Phat[i,i], 1e-10)) / delta
                for j in range(n):
                    if j != i:
                        Q0[i,j] = qi * (Phat[i,j] / max(1-Phat[i,i], 1e-10))
            Q0[i,i] = -np.sum(Q0[i,i+1:])
            
        return Q0

    def MCEMA(self,K,tol):
        max_in_iter = 100_000
        Q0 = self.initialize_Q0()
        Q = Q0.copy()
        for k in range(K):
            Nij = np.zeros((self.N+1, self.N+1))
            Si = np.zeros(self.N+1)
            
            for i in range(self.n):
                for j1,j2 in enumerate(range(self.stepsize,len(self.timeseries[i])*self.stepsize,self.stepsize)):
                    end_time = j2
                    current_time = j2-self.stepsize
                    current_state = self.timeseries[i][j1]
                    LS_ij = np.zeros_like(Nij)
                    LS_i = np.zeros_like(Si)    
                    cond = False
                    at = 0
                    while not cond:
                        rate = -Q[current_state,current_state]
                        if rate <= 0:
                            current_time = end_time
                        else:
                            probs = Q[current_state,:]/rate
                            time_to_leave = self.eval_exp(rate)
                            
                            if current_time + time_to_leave >= end_time:
                                LS_i[current_state] += end_time-current_time
                                current_time = end_time
                            else:
                                current_time += time_to_leave
                                LS_i[current_state] += time_to_leave
                                next = np.random.choice(np.arange(current_state+1,self.N+1),p=probs[current_state+1:])
                                LS_ij[current_state, next] += 1
                                current_state = next

                        if current_time >= end_time:
                            if current_state == self.timeseries[i][j1+1]:
                                Nij += LS_ij
                                Si += LS_i
                                cond = True
                            else:
                                current_time = j2 - self.stepsize
                                current_state = self.timeseries[i][j1]
                                LS_ij.fill(0)
                                LS_i.fill(0)
                        at += 1
                #print(i)
            print(Q)
            Q_k1 = np.zeros_like(Q)
            for i in range(self.N+1):
                if Si[i] > 0:
                    for j in range(self.N+1):
                        if i != j:  
                            Q_k1[i,j] = Nij[i,j] / Si[i]
                else:
                    Q_k1[i,:] = 0
                Q_k1[i,i] = -np.sum(Q_k1[i,:])
            
            diff = np.max(np.abs(Q-Q_k1))
            Q = Q_k1
            best_diff = np.max(np.abs(Q-self.Q))
            print(f"Iteration: {k} largest difference between old and new: {diff}, largest difference between new and best {best_diff}")
            if diff < tol:
                print("Termination criteria reached")
                return Q
        
  
            
step = 48
women = CTMC_part3(n,Q,step)
women.sim()
women.alive
women.lifetime
women.timeseries

tol = 0.0003
#tol = 1e-4
#1e-4
#K = 1000
#Q = women.MCEMA(K,tol)


women.timeseries

import matplotlib.pyplot as plt
    
    

max_len = max(len(ts) for ts in women.timeseries)
data = np.array([ts + [ts[-1]] * (max_len - len(ts)) for ts in women.timeseries])

props2 = np.array([np.bincount(data[:, i], minlength=5) for i in range(max_len)])

states = [f"State {s}" for s in range(5)]
time_steps = props2.shape[0]
x = np.arange(time_steps)  
width = 0.8  

fig, ax = plt.subplots(layout='constrained')

bottom = np.zeros(props2.shape[0])
colors = ['#4a90e2', '#50e3c2', '#7ed321',"#f5a623",'#D0021b']
for state_idx, (state_name, color) in enumerate(zip(states, colors)):
    counts = props2[:, state_idx]
    rects = ax.bar(x, counts, width, bottom=bottom, label=state_name, color=color)
    bottom += counts

ax.set_ylabel('Count in state')
ax.set_title('State Occupancy')
times = x * 48
ax.set_xticks(x, labels=times)
ax.set_xlabel('Time')
ax.legend(loc='upper left', ncols=2)
plt.xticks(rotation=45)
ax.set_ylim(0, bottom.max() + 5)
plt.savefig("State_occ.pdf")    
plt.show()



n_series = len(women.realtimeseries)
n_steps = max_len
time_disc = np.arange(max_len) * 48
time_grid = time_disc 

real_data = np.zeros((n_series, n_steps), dtype=int)
for i, (states, times) in enumerate(zip(women.realtimeseries, women.times)):
    for j, t in enumerate(time_grid):
        if t < times[0]:
            real_data[i, j] = states[0]
        else:
            idx = np.searchsorted(times, t, side='right') - 1
            real_data[i, j] = states[idx]

props_real = np.array([
    np.bincount(real_data[:, i], minlength=5) / n_series
    for i in range(n_steps)
])

fig, axes = plt.subplots(ncols=1, figsize=(12, 5), constrained_layout=True)

labels = [f"State {s}" for s in range(5)]
colors = ['#4a90e2', '#50e3c2', '#7ed321',"#f5a623",'#D0021b']

axes.stackplot(time_grid, props_real.T, labels=labels, colors=colors)
axes.set_title("Real timeseries distribution Q^k")
axes.set_xlabel("Time")
axes.legend(loc='upper right')
plt.savefig("Realtimedist.pdf")
plt.show()

total_figs = 9
fig, ax = plt.subplots(nrows=3,ncols=3, figsize=(12, 5),layout='constrained')
for i, (ts, real_ts, real_t) in enumerate(zip(women.timeseries[0:9], women.realtimeseries[0:9], women.times[0:9])):
    row = i // 3
    col = i % 3
    
    discrete_t = np.arange(len(ts)) * 48
    ax[row, col].step(discrete_t, ts, where='post', label=f"Observed series {i+1}", linestyle='--', linewidth=1)

    ax[row, col].step(real_t, real_ts, where='post', label=f"Real series {i+1}", linewidth=2)
    ax[row, col].set_xlabel("Time")
    ax[row, col].set_ylabel("State")
    ax[row, col].legend()

ax[0, 1].set_title("Stair plot comparison of real vs obs. timeseries")
plt.savefig("real_obs.pdf")
plt.show()

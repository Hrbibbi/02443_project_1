import numpy as np
#from part2 import CTMC
Q = np.array([[-0.0085,0.005,0.0025,0,0.001],[0,-0.014,0.005,0.004,0.005],[0,0,-0.008,0.003,0.005],[0,0,0,-0.009,0.009],[0,0,0,0,0]])
n = 1000
from scipy.stats import expon

class CTMC_part3:
    def __init__(self,n : int,Q):
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
            next = np.array([np.random.choice(np.arange(current_states[i]+1,self.N+1),p=np.abs(probs[i,current_states[i]+1:])) for i in range(len(current_states))])
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
                next_sample_time += 48
            sampled_ts[i].append(4)
            self.timeseries[i] = sampled_ts[i]
    def MCEMA(self,K):
        #Initial Q
        Q0 = np.triu(np.random.rand(self.N+1, self.N+1)/2, k=1)  
        for i in range(self.N+1):
            Q0[i, i] = -np.sum(Q0[i])
        for k in range(K):
            state = np.zeros(self.n)
            complete_timeseries = [[] for _ in range(self.n)]
            for i in range(self.n):
                for j1,j2 in enumerate(48,len(self.timeseries[i])*48,48):
                    time = j2 - 48
                    current_state = self.timeseries[(j2 // 48)-1]
                    current_lifetime = self.times[(j2 // 48)-1]
                    temp_series = []
                    if j1 == 0:
                        temp_series[i].append(current_state)
                    cond = True
                    while cond:
                        probs = -Q0[current_state,:]/Q0[current_state,current_state]
                        time_to_leave = self.eval_exp(-Q0[current_state,current_state])
                        current_lifetime += time_to_leave
                        next = np.array([np.random.choice(np.arange(current_state+1,self.N+1),p=np.abs(probs[i,current_state+1:])) for i in range(len(current_state))])
                        current_state = next
                        temp_series[i].append(current_state)
                        if time <= j2:
                            if current_state == self.timeseries[i][j1]:
                                cond = True
                                complete_timeseries[i].append(temp_series)
                            else:
                                temp_series = []
                                time = j2 - 48
                                current_state = self.timeseries[(j2 // 48)-1]
                                current_lifetime = self.times[(j2 // 48)-1]
                                    
                        
                        
                    
                    
                
            
            

women = CTMC_part3(n,Q)
women.sim()
women.alive
women.lifetime
women.timeseries

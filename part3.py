import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
from scipy.special import factorial, comb
from tqdm import tqdm

Q = np.array([
    [-0.0085,  0.005,   0.0025, 0,      0.001],
    [ 0,      -0.014,   0.005,  0.004,  0.005],
    [ 0,       0,      -0.008,  0.003,  0.005],
    [ 0,       0,       0,     -0.009,  0.009],
    [ 0,       0,       0,      0,      0    ]
])

#%% Task 12
def task12(Q):
    N_states = len(Q)
    N_sim = 1000
    states = np.arange(N_states)
    time_until_death = np.zeros(N_sim)
    death_state = N_states-1
    
    Y_list = []
    for i in range(N_sim):
        state = 0
        time = 0
        Y = []
        time_discrete = 0
        while state != death_state:
            time_prev = time
            t_exp = sps.expon.rvs(scale=1/(-Q[state,state]))
            time += t_exp
            
            while time_discrete < time:
                time_discrete += 48
                Y.append(state)
            
            state_prev = state
            state = np.random.choice(states[state+1:], p=-Q[state,state+1:]/Q[state,state])
        
        Y.append(death_state)
        time_until_death[i] = time
        Y_list.append(np.array(Y))
    
    return Y_list

#%% Task 13
def task13(Q):
    N_states = len(Q)
    states = np.arange(N_states)
    death_state = N_states-1
    
    def estimate_Q(N_ij, S_i):
        """from equation (2)"""
        Q0 = np.zeros((N_states,N_states))
        for i in range(N_states):
            for j in range(i+1,N_states):
                Q0[i,j] = N_ij[i,j] / S_i[i]
        
        np.fill_diagonal(Q0, -Q0.sum(axis=1))
        Q0[-1,-1] = 0
        return Q0
        
    def Q0_heuristic(Y_list):
        """
        we use the partial information from the Y observations
        to generate estimates of N_ij and S_i
        """
        N_ij = np.zeros((N_states,N_states))
        S_i = np.zeros(N_states)
        for Y in Y_list:
            S_i[0] += 48
            for i in range(1,len(Y)):
                S_i[Y[i]] += 48
                if Y[i] != Y[i-1]:
                    N_ij[Y[i-1],Y[i]] += 1
        
        Q0 = estimate_Q(N_ij, S_i)
        return Q0    
    
    def simulate_trajectories(Q,Y_list):
        """
        simulate possible complete trajectories
        by rejecting mismatches at each step
        
        each trajectory is a list of pairs [state,time spent in state]
        """
        trajectories = []
        for Y in Y_list:
            state = 0
            time = 0
            trajectory = []
            for step in range(1,len(Y)):
                accepted = False
                state = Y[step-1]
                while not accepted:
                    trajectory_step = []
                    time = 48*step
                    state_cand = state
                    while time < 48*(step+1):
                        if state_cand == death_state:
                            if step == len(Y)-1:
                                accepted = True
                                trajectory += trajectory_step
                                time = np.inf
                            break
                        t_exp = sps.expon.rvs(scale=1/(-Q[state,state]))
                        time += t_exp
                        trajectory_step.append([state_cand,t_exp])
                        event_state_repeat = time-t_exp == 48*step and t_exp > 48
                        if event_state_repeat:
                            continue
                        s = state_cand
                        state_cand = np.random.choice(states[s+1:], p=-Q[s,s+1:]/Q[s,s])
                    else:
                        if trajectory_step[-1][0] != Y[step]:
                            accepted = False
                        else:
                            accepted = True
                            trajectory_step[-1][1] -= time - 48*(step+1)
                            trajectory += trajectory_step
            
            trajectories.append(trajectory)
            
        return trajectories
        
    
    Y_list = task12(Q)
    Q0 = Q0_heuristic(Y_list)
    Qk = Q0
    print('Q:\n',Q,'\n')
    print('Q0:\n',Q0,'\n')
    eps = 1e-3
    error = np.linalg.norm(Q-Qk, ord=np.inf)
    print(f'True error before MCEM (partial information): {error}')
    diff = np.inf
    it = 0
    while diff > eps:
        ###
        # Step 1: simulate trajectories
        trajectories = simulate_trajectories(Qk,Y_list)
        
        # Step 2: calculate N_ij and S_i
        N_ij = np.zeros((N_states,N_states))
        S_i = np.zeros(N_states)
        for traj in trajectories:
            for k in range(len(traj)):
                if k == len(traj)-1:
                    N_ij[traj[k][0], death_state] += 1
                    S_i[traj[k][0]] += traj[k][1]
                else:
                    N_ij[traj[k][0],traj[k+1][0]] += 1
                    S_i[traj[k][0]] += traj[k][1]    
        
        # Step 3: estimate Qk
        Qk_prev = Qk
        Qk = estimate_Q(N_ij, S_i)
        
        ###  convergence details
        diff = np.linalg.norm(Qk_prev-Qk, ord=np.inf)
        print(f'iteration {it}, convergence norm={diff}, eps={eps}')
        it += 1
    
    print(f'True error after MCEM: {np.linalg.norm(Q-Qk, ord=np.inf)}')
    print('\nQk:\n',Qk)
    
    
if __name__ == '__main__':
    np.random.seed(42)
    # task12(Q)
    task13(Q)
    pass
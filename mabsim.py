import numpy as np
from tqdm import tqdm_notebook as tqdm
from collections import Iterable
import matplotlib.pyplot as plt


def run_widget(A, g, horizon):

    #time-horizon
    tau = int(horizon)  	   

    M = mabs(A, g, tau, repetitions=1)

    print("algorithm: ", str(g))
    print("horizon: ", tau)

    M.run(tqdm_leave=True)

    return M


def create_widget(A, G, default_horizon=1000):	

    from ipywidgets import FloatLogSlider, interactive, fixed

    slider = FloatLogSlider(
        value=default_horizon,
        base=10,
        min=1, # max exponent of base
        max=5, # min exponent of base
        step=1, # exponent step
        description='time-horizon ($n$)',
        readout_format='d'
    )

    return interactive(run_widget, {'manual' : True, 'manual_name' : 'Run Simulation'}, A=fixed(A), g=G, horizon=slider)

#-----------------------------------------------------------------------------------------------------

class mabs:

    def __init__(self, A, G, horizon, repetitions=1, window=None, inibudget=0.0, min_r=-1.0, max_r=+1.0, run=False, save_only_means=True):

        #time-horizon (0, 1 ... t ... tau)
        self.tau = horizon
        #self.T = np.arange(self.tau)          #range for time (0 ... tau-1)
        #self.T1 = np.arange(1, self.tau+1)    #range for time (1 ... tau)
        #self.T01 = np.arange(0, self.tau+1)   #range for time (0, 1 ... tau)
        self.T = range(self.tau)          #range for time (0 ... tau-1)
        self.T1 = range(1, self.tau+1)    #range for time (1 ... tau)
        self.T01 = range(0, self.tau+1)   #range for time (0, 1 ... tau)

        #arms (1 ... a ... k)
        self.A = A if isinstance(A, Iterable) else [A]

        #number of arms
        self.k = len(self.A)
        #self.K = np.arange(self.k)   #range for arms (0 ... k-1)
        #self.K1 = self.K+1           #range for arms (1 ... k)
        self.K = range(self.k)   #range for arms (0 ... k-1)
        self.K1 = range(1,self.k+1)           #range for arms (1 ... k)

        #arms properties
        self.mu_a = np.array([a.mean for a in A])  #means
        self.mu_star = np.max(self.mu_a)           #best mean
        self.a_star = np.argmax(self.mu_a)         #best arm index
        self.mu_worst = np.min(self.mu_a)          #worst mean
        self.a_worst = np.argmin(self.mu_a)        #worst arm index

        #budget
        self.b_0 = inibudget
        
        #extended reward domain
        self.min_r = min_r
        self.max_r = max_r
        self.delta_r = max_r - min_r

        #algorithms (1 ... j ... m)
        self.G = G if isinstance(G, Iterable) else [G]
        self.m = len(self.G)

        #repetitions (1 ... i ... n)
        self.n = repetitions

        #window
        if (window is not None):
            self.win = max(2, min(window, horizon-1))
        else:
            self.win = window

        #if save all sim data
        self.save_only_means = save_only_means		

        #run
        if run:
            self.run()


    def run(self, tqdm_desc_it="iterations", tqdm_desc_alg="algorithms", tqdm_desc_rep="repetitions", tqdm_leave=False, tqdm_disable=False, prev_draw=True):

        # Initialize Rewards and History of selected Actions (3d matrices [t x j x i])
        X = np.zeros((self.n, self.m, self.tau), dtype=float)
        R = np.zeros((self.n, self.m, self.tau), dtype=float)
        H = np.zeros((self.n, self.m, self.tau), dtype=int)

        # For each repetition
        #for i in tqdm(range(self.n), desc=tqdm_desc_rep, leave=(tqdm_leave and self.m == 1), disable=(tqdm_disable or self.n == 1)):
        for i in tqdm(range(self.n), desc=tqdm_desc_rep, leave=tqdm_leave, disable=(tqdm_disable or self.n == 1)):

            # Draw
            if prev_draw:
                X_prev_a = np.array([a.draw_nparray((self.tau, self.n)) for a in self.A])	

            # For each algorithm
            #for j, g in enumerate(tqdm(self.G, desc=tqdm_desc_alg, leave=tqdm_leave, disable=(tqdm_disable or self.m == 1))):
            for j, g in enumerate(self.G):

                # Initialize
                g.startGame()

                # Loop on time
                for t in tqdm(self.T, desc=tqdm_desc_it, leave=tqdm_leave, disable=(tqdm_disable or self.n > 1 or self.m > 1) ):
                    # The algorithm chooses the arm to play
                    a_t = g.choice()
                    # The arm played gives a reward
                    if prev_draw:
                        x_t = X_prev_a[a_t, t, i]
                    else:
                        x_t = self.A[a_t].draw()
                    # The reward is returned to the algorithm
                    g.getReward(a_t, x_t)
                    # Save both
                    X[i, j, t] = x_t
                    H[i, j, t] = a_t

        R = X * self.delta_r + self.min_r
        
        #actions history, with initial action index being 1, not 0
        H1 = H+1

        #actions map (bool 4d matrix)
        H_a = np.array([[[[True if (H[i,j,t]==a) else False for t in self.T] for a in self.K] for j in range(self.m)] for i in range(self.n)], dtype='bool')

        #progressive actions count (int 4d matrix [t x j x i x a])
        N_a = np.cumsum(H_a, axis=3)

        #averaged progressive actions count (float 3d matrix [t x j x a]) #averaged over repetitions
        self.MN_a = np.mean(N_a, axis=0)		

        #progressive actions frequency (float 4d matrix [t x j x i x a])
        F_a = N_a / self.T1

        #averaged progressive actions frequency (float 3d matrix [t x j x a]) #averaged over repetitions
        self.MF_a = np.mean(F_a, axis=0)

        if (self.win is not None):

            #window count (int 4d matrix [t x j x i x a])
            NW_a = np.concatenate((N_a[:,:,:,:self.win], N_a[:,:,:,self.win:] - N_a[:,:,:,:-self.win]), axis=3)

            #averaged window count (float 3d matrix [t x j x a]) #averaged over repetitions
            self.MNW_a = np.mean(NW_a, axis=0)		

            #window frequency (float 4d matrix [t x j x i x a])
            FW_a = np.concatenate((N_a[:,:,:,:self.win] / np.arange(1,self.win+1, dtype='float'), (N_a[:,:,:,self.win:] - N_a[:,:,:,:-self.win]) / float(self.win)), axis=3) 

            #averaged window frequency (float 3d matrix [t x j x a]) #averaged over repetitions
            self.MFW_a = np.mean(FW_a, axis=0)		

        #final arm pull count (int 3d matrix [j x i x a])
        n_a = N_a[:,:,:,self.tau-1]

        #averaged final arm pull count (float 2d matrix [j x a]) #averaged over repetitions
        self.mn_a = np.mean(n_a, axis=0)

        #final arm pull frequency (float 3d matrix [j x i x a])
        f_a = F_a[:,:,:,self.tau-1]

        #averaged final arm pull frequency (float 2d matrix [j x a]) #averaged over repetitions
        self.mf_a = np.mean(f_a, axis=0)

        #progressive cumulative rewards (float 3d matrix [t x j x i])
        SR = np.cumsum(R, axis=2, dtype='float')

        #averaged progressive cumulative rewards (float 2d matrix [t x j]) #averaged over repetitions
        self.MSR = np.mean(SR, axis=0)

        #final rewards (float 2d matrix [j x i])
        sr = SR[:,:,self.tau-1]

        #averaged final rewards (float 1d matrix [j]) #averaged over repetitions
        self.msr = np.mean(sr, axis=0)
        #and standard deviation
        self.dsr = np.std(sr, axis=0)

        #progressive average rewards (float 3d matrix [t x j x i]) #averaged over time
        MR = SR / self.T1

        #averaged progressive average rewards (float 2d matrix [t x j]) #averaged over time and repetitions
        self.MMR = np.mean(MR, axis=0)

        #regret (float 3d matrix [t x j x i])
        L = self.mu_star - R

        #averaged regret (float 2d matrix [t x j])
        #self.ML = np.mean(L, axis=0)
        #progressive average regret (float 3d matrix [t x j x i]) #averaged over time
        ML = self.mu_star - MR

        #averaged average regret (float 2d matrix [t x j]) #averaged over time and repetitions
        self.MML = np.mean(ML, axis=0)

        #cumulated regret (float 3d matrix [t x j x i])
        SL = np.cumsum(L, axis=2, dtype='float')

        #averaged cumulated regret (float 2d matrix [t x j]) #averaged over repetitions
        self.MSL = np.mean(SL, axis=0)

        #final cumulated regret (float 2d matrix [j x i])
        sl = SL[:,:,self.tau-1]

        #averaged final cumulated regret (float 1d matrix [j]) #averaged over repetitions
        self.msl = np.mean(sl, axis=0)
        #and standard deviation
        self.dsl = np.std(sl, axis=0)
        
        #rewards map (float 4d matrix [t x j x i x a])
        R_a = np.array([[[[R[i,j,t] if (H[i,j,t]==a) else 0.0 for t in self.T] for a in self.K] for j in range(self.m)] for i in range(self.n)], dtype='float')

        #averaged rewards map (float 3d matrix [t x j x a]) #averaged over repetitions
        self.MR_a = np.mean(R_a, axis=0)

        #progressive rewards map (int 4d matrix [t x j x i x a])
        SR_a = np.cumsum(R_a, axis=3)

        #averaged progressive rewards map (float 3d matrix [t x j x a]) #averaged over repetitions
        self.MSR_a = np.mean(SR_a, axis=0)

        #final rewards per action (float 3d matrix [j x i x a])
        sr_a = SR_a[:,:,:,self.tau-1]

        #averaged final rewards per action (float 2d matrix [j x a]) #averaged over repetitions
        self.msr_a = np.mean(sr_a, axis=0)

        #reward proportion per action (float 3d matrix [j x i x a])
        fr_a = sr_a / SR[:,:,self.tau-1,np.newaxis]

        #averaged proportion per action (float 2d matrix [j x a]) #averaged over repetitions
        self.mfr_a = np.mean(fr_a, axis=0)

        #progressive budget (float 3d matrix [t x j x i])
        # i.e. the progressive cumulative rewards plus initial budget
        B = SR + self.b_0

        #progressive on negative counter of episodes (float 3d matrix [t x j])
        # i.e. the number of episodes where, at each time t, alg j is running on negative budget
        N = np.sum(B >= 0, axis=0)
        
        #averaged progressive budget (float 2d matrix [t x j]) #averaged over repetitions
        #self.MB = np.mean(B, axis=0)
        self.MB = self.MSR + self.b_0

        #final budget (float 2d matrix [j x i])
        b = B[:,:,self.tau-1]

        #averaged final budget (float 1d matrix [j]) #averaged over repetitions
        self.mb = np.mean(b, axis=0)

        #time map on negative budget (int 3d matrix [t x j x i])
        TNB = np.array([[[1 if(v<0) else 0 for v in B_ij] for B_ij in B_i] for B_i in B])

        #time dead map (int 3d matrix [t x j x i])
        TD = np.maximum.accumulate(TNB, axis=2)

        #progressive survival counter of episodes (float 3d matrix [t x j])
        self.A = 1 - np.mean(TD, axis=0)
        
        #time map of the averaged budget on negative (int 2d matrix [t x j])
        self.TNMB = np.array([[1 if(v<0) else 0 for v in MB_j] for MB_j in self.MB])

        #survival time (before ruin or end) (int 2d matrix [j x i])
        Z = np.reshape(np.ones(self.n*self.m, dtype='int'), [self.n, self.m, 1]) #add 1 at the end		
        TNBZ = np.block([TNB, Z])
        self.TTNB = np.array([[np.nonzero(v_tj==1)[0][0] for v_tj in v_t] for v_t in TNBZ])		

        #averaged survival time (before ruin or end) (int 1d matrix [j])
        self.MTTNB = np.mean(self.TTNB, axis=0)
        #and std dev
        self.DTTNB = np.std(self.TTNB, axis=0)

        #cumulated time progression on negative budget
        STNB = np.cumsum(TNB, axis=2)
        self.STNMB = np.cumsum(self.TNMB, axis=1) 
        #self.MSTNB = np.mean(self.STNB, axis=0)

        #final cumulated time on negative budget
        stnb = STNB[:,:,self.tau-1]

        self.stnmb = self.STNMB[:,self.tau-1]

        #averaged final cumulated time on negative budget
        self.mstnb = np.mean(stnb, axis=0)
        #and std dev
        self.dstnb = np.std(stnb, axis=0)

        #ruin episodes (int 1d matrix [j])
        self.senb = np.count_nonzero(stnb, axis=0) 
        #rate
        self.renb = 1.0 - self.senb / self.n

        #negative budget progression
        NB = np.array([[[v if(v<0) else 0 for v in B_ij] for B_ij in B_i] for B_i in B])

        #average negative budget progression
        self.NMB = np.array([[v if(v<0) else 0 for v in MB_j] for MB_j in self.MB])

        #cumulated negative budget progression
        SNB = np.cumsum(NB, axis=2, dtype='float')

        #self.MSNB = np.mean(SNB, axis=0)

        #cumulated negative budget progression on average
        self.SNMB = np.cumsum(self.NMB, axis=1, dtype='float') 

        #final cumulated negative budget
        snb = SNB[:,:,self.tau-1]

        self.snmb = self.SNMB[:,self.tau-1]

        #final cumulated negative budget (float 1d matrix [j]) #averaged over repetitions
        self.msnb = np.mean(snb, axis=0)
        #and its std deviation
        self.dsnb = np.std(snb, axis=0)

        if(not self.save_only_means):
            self.R = R
            self.H = H
            self.H1 = H1
            self.H_a = H_a
            self.R_a = R_a
            self.N_a = N_a
            self.F_a = F_a
            self.n_a = n_a
            self.f_a = f_a
            self.NW_a = NW_a
            self.SR = SR
            self.sr = sr
            self.MR = MR
            self.L = L
            self.ML = ML
            self.SL = SL
            self.B = B
            self.b = b
            self.TNB = TNB
            self.STNB = STNB
            self.NB = NB
            self.SNB = SNB
            self.snb = snb

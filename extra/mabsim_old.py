import numpy as np
from tqdm import tqdm_notebook as tqdm
from collections import Iterable
import matplotlib.pyplot as plt

	
def _simulate_one(
		arms, algorithm, horizon, 
		rewards, choices, 
		tqdm_desc="iterations", tqdm_leave=False, tqdm_disable=False):
	
	# Initialize
	algorithm.startGame()
	
	# Loop on time
	for t in range(horizon):
		# The algorithm chooses the arm to play
		a_t = algorithm.choice()
		# The arm played gives a reward
		r_t = arms[a_t].draw()
		# The reward is returned to the algorithm
		algorithm.getReward(a_t, r_t)
		# Save both
		rewards[t] = r_t
		choices[t] = a_t


def _simulate_one_rep(
		arms, algorithm, horizon, repetitions, 
		rewards, choices, 
		tqdm_desc="repetitions", tqdm_leave=False, tqdm_disable=False):

	# For each repetition
	for i in tqdm(range(repetitions), desc=tqdm_desc, leave=tqdm_leave, disable=tqdm_disable):
		_simulate_one(arms, algorithm, horizon, rewards[i], choices[i], tqdm_disable=True)

		
def _simulate_all(
		arms, algorithms, horizon, 
		rewards, choices,
		tqdm_desc="algorithms", tqdm_leave=False, tqdm_disable=False):

	# For each algorithm
	for j, g in enumerate(tqdm(algorithms, desc=tqdm_desc, leave=tqdm_leave, disable=tqdm_disable)):
		_simulate_one(arms, g, horizon, rewards[0, j], choices[0, j], tqdm_disable=True)


def _simulate_all_rep(
		arms, algorithms, horizon, repetitions, 
		rewards, choices,
		tqdm_desc_alg="algorithms", tqdm_desc_rep="repetitions", tqdm_leave=False, tqdm_disable=False):

	# For each algorithm
	for j, g in enumerate(tqdm(algorithms, desc=tqdm_desc_alg, leave=tqdm_leave, disable=tqdm_disable)):
		# For each repetition
		for i in tqdm(range(repetitions), desc=tqdm_desc_rep, leave=tqdm_leave, disable=tqdm_disable):
			_simulate_one(arms, g, horizon, rewards[i, j], choices[i, j], tqdm_disable=True)

	
def simulate(
		arms, algorithms, horizon, repetitions=1, 
		tqdm_desc_it="iterations", tqdm_desc_alg="algorithms", tqdm_desc_rep="repetitions", tqdm_leave=False, tqdm_disable=False):

	# Verify if single or multiple algorithms to be evaluated
	num_algs = len(algorithms) if isinstance(algorithms, Iterable) else 1

	# Initialize
	rewards, choices = np.zeros((repetitions, num_algs, horizon), dtype=float), np.zeros((repetitions, num_algs, horizon), dtype=int)
	
	# Simulate
	if isinstance(algorithms, Iterable):
		if repetitions > 1:
			_simulate_all_rep(arms, algorithms, horizon, repetitions, rewards, choices, tqdm_desc_alg=tqdm_desc_alg, tqdm_desc_rep=tqdm_desc_rep, tqdm_leave=tqdm_leave, tqdm_disable=tqdm_disable)
			return rewards, choices
		else:
			_simulate_all(arms, algorithms, horizon, rewards, choices, tqdm_desc=tqdm_desc_alg, tqdm_leave=tqdm_leave, tqdm_disable=tqdm_disable)
			return rewards[0], choices[0]
	else:
		if repetitions > 1:
			_simulate_one_rep(arms, algorithms, horizon, repetitions, rewards, choices, tqdm_desc=tqdm_desc_rep, tqdm_leave=tqdm_leave, tqdm_disable=tqdm_disable)
			return np.reshape(rewards, (repetitions, horizon)), np.reshape(choices, (repetitions, horizon))
		else:
			_simulate_one(arms, algorithms, horizon, rewards[0, 0], choices[0, 0], tqdm_desc=tqdm_desc_it, tqdm_leave=tqdm_leave, tqdm_disable=tqdm_disable)
			return rewards[0, 0], choices[0, 0]

	return rewards, choices

#--------------------------------------------------------------------------------------------------------------
#from an array, return the cumulative sum array on the last dimension (result has same dimensions)
#
# [[1 2 3]   =>   [[ 1  3  6]
#  [4 5 6]]        [ 4  9 15]]
#
def cumulative_sum(values):
	return np.cumsum(values, axis=values.ndim-1)

#--------------------------------------------------------------------------------------------------------------
#from the cumulative sum array, return the progressive average on the last dimension (result has the same dimensions)
#
# [[ 1  3  6]   =>   [[1.0  1.5  2.0]
#  [ 4  9 15]]        [4.0  4.5  5.0]]
#
def progressive_average_from_cumulative(cumulative_values):
	return cumulative_values / np.arange(1, cumulative_values.shape[cumulative_values.ndim-1]+1, dtype=float)

#--------------------------------------------------------------------------------------------------------------
#from an array, return the progressive average (same dimensions)
#
# [[1 2 3]   =>   [[1.0  1.5  2.0]
#  [4 5 6]]        [4.0  4.5  5.0]]
#
def progressive_average(values):
	return np.cumsum(values, axis=values.ndim-1, dtype=float) / np.arange(1, values.shape[values.ndim-1]+1, dtype=float)

#--------------------------------------------------------------------------------------------------------------
#from an array, return the average (reducing 1 dimension)
#
# [[1 2 3]   =>   [2.0
#  [4 5 6]]        5.0]
#
def average(values):
	return np.mean(values, axis=0, dtype=float)

#--------------------------------------------------------------------------------------------------------------
#from an array, return the window average (same dimensions)
#
# [[1 2 3]   =>   [[0.5  1.5  2.5]		, for w=2
#  [4 5 6]]        [2.0  4.5  5.5]]
#
def window_average(values, w, compensate=False):
	return 0

#--------------------------------------------------------------------------------------------------------------
#from an array, return the window average (same dimensions)
def window_average_from_cumulative_map(cumulative_map, w, compensate=False):
    valid_y = np.true_divide(np.subtract(cumulative_map[w:], cumulative_map[:-w]), float(w))
    if compensate:
        begin_y = np.true_divide(cumulative_map[:w], np.arange(1,w+1))
    else:
        begin_y = np.true_divide(cumulative_map[:w], float(w))
    return np.append(begin_y, valid_y, axis=0)

#--------------------------------------------------------------------------------------------------------------
#from an array, return the window average (same dimensions)
def window_average_from_map(map, w, compensate=False):
    return window_average_from_cumulative_map(cumulative_sum(map), w, compensate)

#--------------------------------------------------------------------------------------------------------------
#return an array with the cumulative rewards corresponding to the given best mean in a given horizon
def mean_star_sequence(mustar, n):
	return np.linspace(mustar, mustar*n, n)
	
#--------------------------------------------------------------------------------------------------------------
#from an array of rewards, and a given optimal reward, return an array of instant regrets (same dimensions)
def regret(rewards, mustar=1.0):
    return mustar - rewards
	
#--------------------------------------------------------------------------------------------------------------
#from an array of rewards, and a given optimal reward, return an array of cumulated regrets (same dimensions)
def cumulative_regret(rewards, mustar=1.0):
	#return mustar * np.arange(1, rewards.shape[rewards.ndim-1]+1, dtype=float) - cumulative_sum(rewards)	
	return np.cumsum(mustar - rewards, axis=rewards.ndim-1, dtype=float)

#--------------------------------------------------------------------------------------------------------------
#from an array of history of selected actions, return a boolean map (increase 1 dimension)
def actions_map(k, actions):
	return np.array([[1 if (actions[t]==i) else 0 for t in range(len(actions))] for i in range(k)], dtype='bool')

#--------------------------------------------------------------------------------------------------------------
#from an array of history of selected actions, return a boolean map (increase 1 dimension)
def actions_progressive_count_from_map(map):
	return np.cumsum(map, axis=1)

#--------------------------------------------------------------------------------------------------------------
def actions_progressive_count(k, actions):
	return actions_progressive_count_from_map(actions_map(k, actions))

#--------------------------------------------------------------------------------------------------------------
def actions_count(k, actions):
	return np.bincount(actions, minlength=k)

#--------------------------------------------------------------------------------------------------------------
def actions_freq_from_count(actions_count):
	return actions_count / np.arange(1, actions_count.shape[actions_count.ndim-1]+1, dtype=float)

#--------------------------------------------------------------------------------------------------------------
def actions_progressive_freq(k, actions):
	return actions_freq_from_count(actions_progressive_count(k, actions))

#--------------------------------------------------------------------------------------------------------------
def final_rewards_per_action(k, actions, rewards):
	r = np.zeros(k, dtype='float')
	for t in range(len(actions)):
		r[actions[t]] += rewards[t]
	return r

#--------------------------------------------------------------------------------------------------------------
def star_map_from_actions_map(map, i=0):
	return map[i]

def star_map(actions, i=0):
	return np.array([1 if (actions[t]==i) else 0 for t in range(len(actions))], dtype='bool')

def star_count_from_star_map(map, i=0):
	return np.cumsum(map[i])
	#return np.cumsum(map, axis=1)

def star_count(k, actions, i=0):
	return star_count_from_star_map(star_map(actions, i), i)

def star_freq_from_actions_freq(actions_freq, i=0):
	return actions_freq[i]

def star_freq_from_star_count(star_count, i=0):
	return star_count / np.arange(1, star_count.shape[star_count.ndim-1]+1, dtype=float)

def star_freq(k, actions, i=0):
	return star_freq_from_count(star_count(k, actions), i)
	

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

	def __init__(self, A, G, horizon, repetitions=1, window=20, run=False):

		#time-horizon (0, 1 ... t ... tau)
		self.tau = horizon
		self.T = np.arange(self.tau)          #range for time (0 ... tau-1)
		self.T1 = np.arange(1, self.tau+1)    #range for time (1 ... tau)
		self.T01 = np.arange(0, self.tau+1)   #range for time (0, 1 ... tau)

		#arms (1 ... a ... k)
		self.A = A if isinstance(A, Iterable) else [A]
		
		#number of arms
		self.k = len(self.A)
		self.K = np.arange(self.k)   #range for arms (0 ... k-1)
		self.K1 = self.K+1           #range for arms (1 ... k)

		#arms properties
		self.mu_a = np.array([a.mean for a in A])  #means
		self.mu_star = np.max(self.mu_a)           #best mean
		self.a_star = np.argmax(self.mu_a)         #best arm index
		self.mu_worst = np.min(self.mu_a)          #worst mean
		self.a_worst = np.argmin(self.mu_a)        #worst arm index
		
		#algorithms (1 ... j ... m)
		self.G = G if isinstance(G, Iterable) else [G]
		self.m = len(self.G)
		
		#repetitions (1 ... i ... n)
		self.n = repetitions
		
		#window
		self.win = max(2, min(window, horizon-1))
		
		#run
		if run:
			self.run()
			
			
	def run(self, tqdm_desc_it="iterations", tqdm_desc_alg="algorithms", tqdm_desc_rep="repetitions", tqdm_leave=False, tqdm_disable=False, prev_draw=True):

		#Rewards for every arm in the given horizon
		#draws = M.draw_each_nparray(shape=(n,))
		
	
		# Initialize Rewards and History of selected Actions (3d matrices [t x j x i])
		self.R = np.zeros((self.n, self.m, self.tau), dtype=float)
		self.H = np.zeros((self.n, self.m, self.tau), dtype=int)

		# Draw
		if prev_draw:
			self.RR_a = np.array([a.draw_nparray((self.tau, self.n)) for a in self.A])	
		
		# For each algorithm
		for j, g in enumerate(tqdm(self.G, desc=tqdm_desc_alg, leave=tqdm_leave, disable=(tqdm_disable or self.m == 1))):

			# For each repetition
			for i in tqdm(range(self.n), desc=tqdm_desc_rep, leave=(tqdm_leave and self.m == 1), disable=(tqdm_disable or self.n == 1)):

				# Initialize
				g.startGame()

				# Loop on time
				for t in tqdm(self.T, desc=tqdm_desc_it, leave=tqdm_leave, disable=(tqdm_disable or self.n > 1 or self.m > 1) ):
					# The algorithm chooses the arm to play
					a_t = g.choice()
					# The arm played gives a reward
					if prev_draw:
						r_t = self.RR_a[a_t, t, i]
					else:
						r_t = self.A[a_t].draw()
					# The reward is returned to the algorithm
					g.getReward(a_t, r_t)
					# Save both
					self.R[i, j, t] = r_t
					self.H[i, j, t] = a_t

		#actions history, with initial action index being 1, not 0
		self.H1 = self.H+1
		
		#actions map (bool 4d matrix)
		self.H_a = np.array([[[[True if (self.H[i,j,t]==a) else False for t in self.T] for a in self.K] for j in range(self.m)] for i in range(self.n)], dtype='bool')

		#progressive actions count (int 4d matrix [t x j x i x a])
		self.N_a = np.cumsum(self.H_a, axis=3)

		#averaged progressive actions count (float 3d matrix [t x j x a])
		self.MN_a = np.mean(self.N_a, axis=0)		
		
		#progressive actions frequency (float 4d matrix [t x j x i x a])
		self.F_a = self.N_a / self.T1

		#averaged progressive actions frequency (float 3d matrix [t x j x a])
		self.MF_a = np.mean(self.F_a, axis=0)		
		
		#window count (int 4d matrix [t x j x i x a])
		self.NW_a = np.concatenate((self.N_a[:,:,:,:self.win], self.N_a[:,:,:,self.win:] - self.N_a[:,:,:,:-self.win]), axis=3)

		#averaged window count (float 3d matrix [t x j x a])
		self.MNW_a = np.mean(self.NW_a, axis=0)		
		
		#window frequency (float 4d matrix [t x j x i x a])
		self.FW_a = np.concatenate((self.N_a[:,:,:,:self.win] / np.arange(1,self.win+1, dtype='float'), (self.N_a[:,:,:,self.win:] - self.N_a[:,:,:,:-self.win]) / float(self.win)), axis=3) 

		#averaged window frequency (float 3d matrix [t x j x a])
		self.MFW_a = np.mean(self.FW_a, axis=0)		
		
		#final arm pull count (int 3d matrix [j x i x a])
		self.n_a = self.N_a[:,:,:,self.tau-1]

		#averaged final arm pull count (float 2d matrix [j x a])
		self.mn_a = np.mean(self.n_a, axis=0)
		
		#final arm pull frequency (float 3d matrix [j x i x a])
		self.f_a = self.F_a[:,:,:,self.tau-1]

		#averaged final arm pull frequency (float 2d matrix [j x a])
		self.mf_a = np.mean(self.f_a, axis=0)
		
		#progressive cumulative rewards (float 3d matrix [t x j x i])
		self.SR = np.cumsum(self.R, axis=2, dtype='float')

		#averaged progressive cumulative rewards (float 2d matrix [t x j])
		self.MSR = np.mean(self.SR, axis=0)
		
		#final rewards (float 2d matrix [j x i])
		self.sr = self.SR[:,:,self.tau-1]

		#averaged final rewards (float 1d matrix [j])
		self.msr = np.mean(self.sr, axis=0)
		
		#progressive average rewards (float 3d matrix [t x j x i])
		self.MR = self.SR / self.T1

		#averaged progressive average rewards (float 2d matrix [t x j])
		self.MMR = np.mean(self.MR, axis=0)
		
		#regret (float 3d matrix [t x j x i])
		self.L = self.mu_star - self.R

		#averaged regret (float 2d matrix [t x j])
		#self.ML = np.mean(self.L, axis=0)
		#progressive average regret (float 3d matrix [t x j x i])
		self.ML = self.mu_star - self.MR

		#averaged average regret (float 2d matrix [t x j])
		self.MML = np.mean(self.ML, axis=0)
		
		#cumulated regret (float 3d matrix [t x j x i])
		self.SL = np.cumsum(self.L, axis=2, dtype='float')

		#averaged cumulated regret (float 2d matrix [t x j])
		self.MSL = np.mean(self.SL, axis=0)
		
		#rewards map (float 4d matrix [t x j x i x a])
		self.R_a = np.array([[[[self.R[i,j,t] if (self.H[i,j,t]==a) else 0.0 for t in self.T] for a in self.K] for j in range(self.m)] for i in range(self.n)], dtype='float')

		#averaged rewards map (float 3d matrix [t x j x a])
		self.MR_a = np.mean(self.R_a, axis=0)
		
		#progressive rewards map (int 4d matrix [t x j x i x a])
		self.SR_a = np.cumsum(self.R_a, axis=3)

		#averaged progressive rewards map (float 3d matrix [t x j x a])
		self.MSR_a = np.mean(self.SR_a, axis=0)
		
		#final rewards per action (float 3d matrix [j x i x a])
		self.sr_a = self.SR_a[:,:,:,self.tau-1]

		#averaged final rewards per action (float 2d matrix [j x a])
		self.msr_a = np.mean(self.sr_a, axis=0)
		
		#reward proportion per action (float 3d matrix [j x i x a])
		self.fr_a = self.sr_a / self.SR[:,:,self.tau-1,np.newaxis]

		#averaged proportion per action (float 2d matrix [j x a])
		self.mfr_a = np.mean(self.fr_a, axis=0)
		
		
	def plot_history(self, i=0, j=0, xlabel='$t$', ylabel='Arm', title='History of pulled arms', alpha=0.5, markersize=None, show=True):

		plt.plot(self.T1, self.H1[i,j], 'o', markersize=markersize, alpha=alpha)

		plt.yticks(self.K1)
		plt.ylim([0.5, self.k+0.5])
		plt.gca().invert_yaxis()    

		if xlabel is not None:
			plt.xlabel(xlabel)

		if ylabel is not None:
			plt.ylabel(ylabel)

		if title is not None:
			plt.title(title)

		if show:
			plt.show()

			
	def plot_progression(self, Y, X=None, names=None, linestyles=None, linecolors=None, xlabel="$t$", ylabel="Value", title=None, show=True):

		if Y.ndim > 1:
			if X is None:
				X = range(len(Y[0]))
				
			for i, Y_i in enumerate(Y):
				line, = plt.plot(X, Y_i)
				if linestyles is not None:
					line.set_linestyle(linestyles[i])	
				if linecolors is not None:
					line.set_color(linecolors[i])	
			
		else:
			if X is None:
				X = range(len(Y))
			plt.plot(X, Y)

		if names is not None:
			plt.legend(names)

		if xlabel is not None:
			plt.xlabel(xlabel)

		if ylabel is not None:
			plt.ylabel(ylabel)

		if title is not None:
			plt.title(title)

		if show:
			plt.show()
			
			
	def plot_action_count_progression(self, i=None, j=0, xlabel="$t$", ylabel="Number of pulls", title="Arm pull counter", show=True):
		if self.n == 1:
			i = 0
		#add zeros at time zero
		Z = np.reshape(np.zeros(self.k, dtype='int'), [self.k, 1])
		if i is None:
			Y = np.block([Z, self.MN_a[j]])
		else:
			Y = np.block([Z, self.N_a[i,j]])
		#prepare labels
		names = [f"$N_{a}$" for a in self.K1]
		#call plot progression
		self.plot_progression(Y, X=self.T01, names=names, ylabel=ylabel, xlabel=xlabel, title=title, show=show)

		
	def plot_action_freq_progression(self, i=None, j=0, xlabel="$t$", ylabel="Pull Frequency", title="Arm Selection Frequency", show=True):
		if self.n == 1:
			i = 0
		#prepare labels
		names = [f"$F_{a}$" for a in self.K1]
		#prepare data
		if i is None:
			Y = self.MF_a[j]
		else:
			Y = self.F_a[i,j]
		#call plot progression
		self.plot_progression(Y, X=self.T1, names=names, ylabel=ylabel, xlabel=xlabel, title=title, show=show)

		
	def plot_precision_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Best Arm Pull Frequency", title="Precision", show=True):

		#i: repetition, j: algorithm
		if self.m == 1:
			j = 0
		if self.n == 1:
			i = 0

		if j is None:
			if names is None:
				names=[str(g) for g in self.G]		
			if i is None:
				Y = self.MF_a[:,self.a_star]
			else:
				Y = self.F_a[i,:,self.a_star]
		else:
			if names is None:
				names=['best', 'others']		
			#prepare best and others frequencies
			if i is None:
				Y = np.array([self.MF_a[j, self.a_star], 1-self.MF_a[j, self.a_star]])
			else:
				Y = np.array([self.F_a[i, j, self.a_star], 1-self.F_a[i, j, self.a_star]])
		
		#call plot progression
		self.plot_progression(Y, X=self.T1, names=names, ylabel=ylabel, xlabel=xlabel, title=title, show=show)


	def plot_cumulated_reward_progression(self, i=None, j=None, xlabel="$t$", ylabel="Cumulated Reward", title="Cumulated Reward", show=True):		

		#i: repetition, j: algorithm
		if self.m == 1:
			j = 0
		if self.n == 1:
			i = 0

		#prepare data
		if j is None:  #comparing algorithms
			names=[str(g) for g in self.G]
			if i is None:  #averaged over repetitions      
				Y = self.MSR
			else:		   #in a specific repetition
				Y = self.SR[i]
		else:	#specific algorithm		   
			names=[str(self.G[j])]
			if i is None:  #averaged over repetitions      
				Y = self.MSR[j]
			else: 		   #in a specific repetition
				Y = self.SR[i,j]
		#call plot		
		self.plot_progression(Y, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, show=show)

        
	def plot_budget_progression(self, i=None, j=None, inibudget=0.0, xlabel="$t$", ylabel="Budget", title="Budget", show=True):		

		#i: repetition, j: algorithm
		if self.m == 1:
			j = 0
		if self.n == 1:
			i = 0

		#prepare data
		if j is None:  #comparing algorithms
			names=[str(g) for g in self.G]
			if i is None:  #averaged over repetitions      
				Y = self.MSR
			else:		   #in a specific repetition
				Y = self.SR[i]
		else:	#specific algorithm		   
			names=[str(self.G[j])]
			if i is None:  #averaged over repetitions      
				Y = self.MSR[j]
			else: 		   #in a specific repetition
				Y = self.SR[i,j]
		Y = Y + inibudget                
		#call plot		
		self.plot_progression(Y, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, show=show)


	def plot_negative_budget(self, i=None, j=None, inibudget=0.0, xlabel="$t$", ylabel="Negative Budget", title="Negative Budget", show=True):

		#i: repetition, j: algorithm
		if self.m == 1:
			j = 0
		if self.n == 1:
			i = 0

		#prepare data
		if j is None:  #comparing algorithms
			names=[str(g) for g in self.G]
			if i is None:  #averaged over repetitions      
				Y = self.MSR
			else:		   #in a specific repetition
				Y = self.SR[i]
		else:	#specific algorithm		   
			names=[str(self.G[j])]
			if i is None:  #averaged over repetitions      
				Y = self.MSR[j]
			else: 		   #in a specific repetition
				Y = self.SR[i,j]
		Y = Y + inibudget
		Y = np.array([1 if(v<0) else 0 for v in Y])
		#call plot		
		self.plot_progression(Y, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, show=show)

	def plot_negative_budget_progression(self, i=None, j=None, inibudget=0.0, xlabel="$t$", ylabel="Cumulated Time on Negative Budget", title="Cumulated Time on Negative Budget", show=True):

		#i: repetition, j: algorithm
		if self.m == 1:
			j = 0
		if self.n == 1:
			i = 0

		#prepare data
		if j is None:  #comparing algorithms
			names=[str(g) for g in self.G]
			if i is None:  #averaged over repetitions      
				Y = self.MSR
			else:		   #in a specific repetition
				Y = self.SR[i]
		else:	#specific algorithm		   
			names=[str(self.G[j])]
			if i is None:  #averaged over repetitions      
				Y = self.MSR[j]
			else: 		   #in a specific repetition
				Y = self.SR[i,j]
		Y = Y + inibudget
		Y = np.array([-1 if(v<0) else 0 for v in Y])
		Y = np.cumsum(Y)
		#call plot		
		self.plot_progression(Y, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, show=show)

	def plot_cumulated_negative_budget_progression(self, i=None, j=None, inibudget=0.0, xlabel="$t$", ylabel="Cumulated Negative Budget", title="Cumulated Negative Budget", show=True):

		#i: repetition, j: algorithm
		if self.m == 1:
			j = 0
		if self.n == 1:
			i = 0

		#prepare data
		if j is None:  #comparing algorithms
			names=[str(g) for g in self.G]
			if i is None:  #averaged over repetitions      
				Y = self.MSR
			else:		   #in a specific repetition
				Y = self.SR[i]
		else:	#specific algorithm		   
			names=[str(self.G[j])]
			if i is None:  #averaged over repetitions      
				Y = self.MSR[j]
			else: 		   #in a specific repetition
				Y = self.SR[i,j]
		Y = Y + inibudget
		Y = np.array([v if(v<0) else 0 for v in Y])
		Y = np.cumsum(Y)
		#call plot		
		self.plot_progression(Y, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, show=show)

        
	def plot_average_reward_progression(self, i=None, j=None, xlabel="$t$", ylabel="Average Reward", title="Average Reward", show=True):	

		#i: repetition, j: algorithm
		if self.m == 1:
			j = 0
		if self.n == 1:
			i = 0

		#prepare data
		if j is None:  #comparing algorithms
			names=[str(g) for g in self.G]
			if i is None:  #averaged over repetitions      
				Y = self.MMR
			else:		   #in a specific repetition
				Y = self.MR[i]
		else:	#specific algorithm		   
			names=[str(self.G[j])]
			if i is None:  #averaged over repetitions      
				Y = self.MMR[j]
			else: 		   #in a specific repetition
				Y = self.MR[i,j]
		#call plot		
		self.plot_progression(Y, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, show=show)


	def plot_cumulated_reward_regret_progression(self, i=None, j=None, xlabel="$t$", ylabel="Cumulated Reward and Regret", title="Cumulated Reward and Regret", show=True):	

		#prepare data
		if j is None:  #comparing algorithms
			names=[str(g) for g in self.G]
			if i is None:  #averaged over repetitions      
				Y1 = self.MSR
				Y2 = -self.MSL
			else:		   #in a specific repetition
				Y1 = self.SR[i]
				Y2 = -self.SL[i]
		else:	#specific algorithm		   
			names=[str(self.G[j])]
			if i is None:  #averaged over repetitions      
				Y1 = self.MSR[j]
				Y2 = -self.MSL[j]
			else: 		   #in a specific repetition
				Y1 = self.SR[i,j]
				Y2 = -self.SL[i,j]
		#call plot		
		self.plot_progression(Y1, X=self.T1, names=None, xlabel=None, ylabel=None, title=None, show=False)
		self.plot_progression(Y2, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, show=show, linestyles=['--']*self.m, linecolors=['C' + str(j) for j in range(self.m)])

		
	def plot_average_reward_regret_progression(self, i=None, j=None, xlabel="$t$", ylabel="Average Reward and Regret", title="Average Reward and Regret", show=True):	

		#i: repetition, j: algorithm
		if self.m == 1:
			j = 0
		if self.n == 1:
			i = 0

		#prepare data
		if j is None:  #comparing algorithms
			names=[str(g) for g in self.G]
			if i is None:  #averaged over repetitions      
				Y1 = self.MMR
				Y2 = -self.MML
			else:		   #in a specific repetition
				Y1 = self.MR[i]
				Y2 = -self.ML[i]
		else:	#specific algorithm		   
			names=[str(self.G[j])]
			if i is None:  #averaged over repetitions      
				Y1 = self.MMR[j]
				Y2 = -self.MML[j]
			else: 		   #in a specific repetition
				Y1 = self.MR[i,j]
				Y2 = -self.ML[i,j]
		#call plot		
		self.plot_progression(Y1, X=self.T1, names=None, xlabel=None, ylabel=None, title=None, show=False)
		self.plot_progression(Y2, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, show=show, linestyles=['--']*self.m, linecolors=['C' + str(j) for j in range(self.m)])


	def plot_cumulated_regret_progression(self, i=None, j=None, xlabel="$t$", ylabel="Cumulated Regret", title="Cumulated Regret", show=True):	
		
		#i: repetition, j: algorithm
		if self.m == 1:
			j = 0
		if self.n == 1:
			i = 0

		#prepare data
		if j is None:  #comparing algorithms
			names=[str(g) for g in self.G]
			if i is None:  #averaged over repetitions      
				Y = -self.MSL
			else:		   #in a specific repetition
				Y = -self.SL[i]
		else:	#specific algorithm		   
			names=[str(self.G[j])]
			if i is None:  #averaged over repetitions      
				Y = -self.MSL[j]
			else: 		   #in a specific repetition
				Y = -self.SL[i,j]
		#call plot		
		self.plot_progression(Y, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, show=show)


	def plot_average_regret_progression(self, i=None, j=None, xlabel="$t$", ylabel="Average Regret", title="Average Regret", show=True):	
		
		#prepare data
		if j is None:  #comparing algorithms
			names=[str(g) for g in self.G]
			if i is None:  #averaged over repetitions      
				Y = -self.MML
			else:		   #in a specific repetition
				Y = -self.ML[i]
		else:	#specific algorithm		   
			names=[str(self.G[j])]
			if i is None:  #averaged over repetitions      
				Y = -self.MML[j]
			else: 		   #in a specific repetition
				Y = -self.ML[i,j]
		#call plot		
		self.plot_progression(Y, X=self.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, show=show)

		
	def plot_freq_spectrum(self, F, bar_size=2, interpolation='hanning', cmap='gray_r', xlabel="$t$", ylabel="Actions", title="Frequency Spectrum", show=True):
		
		bs = max(bar_size, 2)   #bar size (>1)
		h = self.k*bs+1     	#fig height (rows depends on the number of arms and bar size)
		w = len(F[0])			#fig width  (columns depends on time)

		img_map = np.zeros([h,w])   #image map

		for c in range(h):
			if (c % bs != 0):
				img_map[c] = F[c//bs]

		plt.imshow(img_map, aspect="auto", interpolation=interpolation, cmap=cmap)
		plt.yticks(np.arange(1, h, step=bs), self.K1)

		##ax = plt.gca()
		#plt.xticks(np.arange(-.5, 10, 1))
		#plt.xticklabels(np.arange(1, 12, 1))
		
		plt.colorbar()

		if xlabel is not None:
			plt.xlabel(xlabel)

		if ylabel is not None:
			plt.ylabel(ylabel)

		if title is not None:
			plt.title(title)

		if show:
			plt.show()

			
			
	def plot_action_window_freq_spectrum(self, i=None, j=0, bar_size=2, interpolation='none', cmap='gray_r', xlabel="$t$", ylabel="Arms", title="Arm Pull Local Frequency Spectrum", show=True):

		#i: repetition, j: algorithm
		if self.n == 1:
			i = 0
		
		#add zeros at time zero
		Z = np.reshape(np.zeros(self.k, dtype='float'), [self.k, 1])
		if i is None:
			Y = np.block([Z, self.MFW_a[j]])
		else:
			Y = np.block([Z, self.FW_a[i,j]])
		#call plot
		self.plot_freq_spectrum(Y, bar_size=bar_size, interpolation=interpolation, cmap=cmap, xlabel=xlabel, ylabel=ylabel, title=title)		


		
	def plot_comp_arms(self, Y, names=None, xlabel="Arms", ylabel=None, title="Arms Comparison", show=True):

		if Y.ndim > 1:
			w = 0.8 / len(Y)
			for i, Y_i in enumerate(Y):
				plt.bar(self.K1 + w*i, Y_i, width=w)   #color=plt.cm.get_cmap(i)
		else:
			plt.bar(self.K1, Y)
		
		if names is not None:
			plt.legend(names)
	
		if xlabel is not None:
			plt.xlabel(xlabel)

		if ylabel is not None:
			plt.ylabel(ylabel)

		if title is not None:
			plt.title(title)

		if show:
			plt.show()

			
			
	def plot_comp_arm_count(self, i=None, j=None, xlabel="Arm (Selected Action)", ylabel="Number of Actions Taken", title="Selected Actions", show=True):
		
		#i: repetition, j: algorithm
		if self.m == 1:
			j = 0
		if self.n == 1:
			i = 0
		
		if j is None:  #comparing algorithms
			names=[str(g) for g in self.G]
			if i is None:   #averaging on repetitions
				Y = self.mn_a
			else:           #specific repetition
				Y = self.n_a[i]
		else:          #specific algorithm
			names=None
			if i is None:   #averaging on repetitions
				Y = self.mn_a[j]
			else:           #specific repetition
				Y = self.n_a[i,j]
		#call plot
		self.plot_comp_arms(Y, names=names, xlabel=xlabel, ylabel=ylabel, title=title)

		
	def plot_comp_arm_rewards(self, i=None, j=None, xlabel="Arms", ylabel="Total Rewards", title="Total Rewards per Arm", show=True):
		if self.m == 1:
			j = 0
		if self.n == 1:
			i = 0
		if j is None:  #comparing algorithms
			names=[str(g) for g in self.G]
			if i is None:
				Y = self.msr_a
			else:
				Y = self.sr_a[i]
		else:          #specific algorithm
			names=None
			if i is None:
				Y = self.msr_a[j]
			else:
				Y = self.sr_a[i,j]
		#call plot
		self.plot_comp_arms(Y, names=names, xlabel=xlabel, ylabel=ylabel, title=title)

		
	def plot_comp_algs(self, i=None, xlabel="Algorithm", ylabel="Value", title="Comparison", sort=True, show=True):

		if self.n == 1:
			i = 0

		if i is None:
			Y = self.msr
		else:
			Y = self.sr[i]
		
		x = np.arange(self.m, dtype='int')
		names = [str(g) for g in self.G]
		
		low = min(Y)
		high = max(Y)
		plt.ylim([low, high])

		#sort
		if sort:
			idx = np.argsort(Y)[::-1]  #desc order
			Y = Y[idx]
			names = [names[i] for i in idx]
		
		plt.xticks(x, names, rotation='vertical')		
		plt.bar(x, Y, align='center', alpha=0.5)

		if xlabel is not None:
			plt.xlabel(xlabel)

		if ylabel is not None:
			plt.ylabel(ylabel)

		if title is not None:
			plt.title(title)

		if show:
			plt.show()

	
			
	def plot_comp_freq_prop(self, i=None, j=0, names=['Cumulated Reward Proportion', 'Pull Frequency'], xlabel="$t$", ylabel="Cumulated Reward Proportion and Pull Frequency", title="Cumulated Reward and Number of Pulls", show=True):

		if self.n == 1:
			i = 0

		if i is None:
			plt.bar(self.K1, self.mfr_a[j], width=0.8)
			plt.bar(self.K1, self.mf_a[j], width=0.4, alpha=0.5)
		else:
			plt.bar(self.K1, self.fr_a[i,j], width=0.8)
			plt.bar(self.K1, self.f_a[i,j], width=0.4, alpha=0.5)
		
		if names is not None:
			plt.legend(names)

		if xlabel is not None:
			plt.xlabel(xlabel)

		if ylabel is not None:
			plt.ylabel(ylabel)

		if title is not None:
			plt.title(title)

		if show:
			plt.show()
		
		
		
	def plot_reward_regret(self, i=None, j=0, names=['Cumulated Reward', 'Cumulated Regret', 'Best Strategy', 'Worst Possible Regret'], xlabel="$t$", ylabel="Reward and Regret", title="Best Strategy vs Cumulated Reward vs Regret ", show=True):

		#i: repetition, j: algorithm
		if self.n == 1:
			i = 0
	
		#best policy
		SS = np.linspace(self.mu_star, self.mu_star*self.tau, self.tau)
		#worst regret
		l = -(self.mu_star-self.mu_worst)
		#worst cumulated regret
		SW = np.linspace(l, l*self.tau, self.tau)
		
		#best, reward, regret
		if i is None:
			Y = np.array([self.MSR[j], -self.MSL[j], SS, SW])
		else:
			Y = np.array([self.SR[i,j], -self.SL[i,j], SS, SW])

		self.plot_progression(Y, X=self.T1, show=False, names=names, title=title, ylabel=ylabel)

		if i is None:
			plt.fill_between(self.T1, 0, self.MSR[j], alpha=0.5)
			plt.fill_between(self.T1, 0, -self.MSL[j], alpha=0.5)
		else:
			plt.fill_between(self.T1, 0, self.SR[i,j], alpha=0.5)
			plt.fill_between(self.T1, 0, -self.SL[i,j], alpha=0.5)

		if xlabel is not None:
			plt.xlabel(xlabel)

		if ylabel is not None:
			plt.ylabel(ylabel)

		if title is not None:
			plt.title(title)

		if show:
			plt.show()

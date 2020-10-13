import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math


def plot_progression(S, T1=None, names=None, linestyles=None, xlabel="$t$", ylabel="Value", title=None, show=True):

	if S.ndim > 1:
		if T1 is None:
			T1 = range(1,len(S[0])+1)
		if linestyles is None:
			for s in S:
				plt.plot(T1, s)
		else:
			for s in S:
				plt.plot(T1, s, linestyle=linestyles[i])
	else:
		if T1 is None:
			T1 = range(1,len(S)+1)
		plt.plot(T1, S)

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

		
def plot_freq(F, k=None, K1=None, n=None, bar_width=2, interpolation='hanning', cmap='gray_r', xlabel="$t$", ylabel="Frequency", title=None, show=True):
    
	if k is None:
		k = len(F)      #number of arms (series)

	if K1 is None:
		K1 = np.arange(1,k+1)      #range of arms (for labels)
		
	if n is None:
		n = len(F[0])   #time-horizon (number of samples)

	w = bar_width #bar width (>1)
	h = k*w+1     #fig height

	img_map = np.zeros([h,n])   #image map

	for i in range(h):
		if (i % w != 0):
			img_map[i] = F[i//w]

	plt.imshow(img_map, aspect="auto", interpolation=interpolation, cmap=cmap)
	plt.yticks(np.arange(1, h, step=w), K1)

	plt.colorbar()

	if xlabel is not None:
		plt.xlabel(xlabel)

	if ylabel is not None:
		plt.ylabel(ylabel)

	if title is not None:
		plt.title(title)

	if show:
		plt.show()

		
def plot_history(H=None, H1=None, k=None, K1=None, n=None, T1=None, xlabel='$t$', ylabel='Arm', title='History of pulled arms', alpha=0.5, markersize=None, show=True):

	#H is the history of pulls starting from index 0 for arms
	#H1 the same, just shifted to have the first arm as index 1
	if H1 is None:
		if H is None:
			raise ValueError("You must define either H or H1.")
		else:
			H1 = H+1

	#n is the time-horizon
	#and T1 is the range, starting from t=1
	if T1 is None:
		if n is None:
			n = len(H1)
		T1 = range(1,n+1)
		
	#k is the number of arms
	#and K1 is the range of arm indexes, starting from 1
	if K1 is None:
		if k is None:
			k = max(H1)
		K1 = range(1,k+1)
	else:
		if k is None:
			k = len(K1)

	plt.plot(T1, H1, 'o', markersize=markersize, alpha=alpha)

	plt.yticks(K1)
	plt.ylim([0.5, k+0.5])
	plt.gca().invert_yaxis()    

	if xlabel is not None:
		plt.xlabel(xlabel)

	if ylabel is not None:
		plt.ylabel(ylabel)

	if title is not None:
		plt.title(title)

	if show:
		plt.show()
	
	
def plot_comp_algs(y, names, xlabel="Algorithm", ylabel="Value", title="Comparison", show=True):

	x = range(len(names))	
	low = min(y)
	high = max(y)
	plt.ylim([low, high])
	plt.bar(x, y, align='center', alpha=0.5)
	plt.xticks(x, names, rotation='vertical')		

	if xlabel is not None:
		plt.xlabel(xlabel)

	if ylabel is not None:
		plt.ylabel(ylabel)

	if title is not None:
		plt.title(title)

	if show:
		plt.show()


def plot_comp_arms(n_a, K1=None, xlabel="Arm (Selected Action)", ylabel="Number of Actions Taken", title="Selected Actions", show=True):

	if K1 is None:
		K1 = np.arange(1,len(n_a)+1, dtype='int')      #range of arms (for labels)
		
	plt.bar(K1, n_a)

	if xlabel is not None:
		plt.xlabel(xlabel)

	if ylabel is not None:
		plt.ylabel(ylabel)

	if title is not None:
		plt.title(title)

	if show:
		plt.show()



def plot_reward_regret(reward, regret, best_strategy, T1=None, names=['Cumulated Reward', 'Cumulated Regret', 'Best Strategy'], xlabel="$t$", ylabel="Reward and Regret", title="Best Strategy vs Cumulated Reward vs Regret ", show=True):

	#best, reward, regret
	Y = np.array([reward, -regret, best_strategy])

	plot_progression(Y, show=False, names=names, title=title, ylabel=ylabel)

	plt.fill_between(T1, 0, reward)
	plt.fill_between(T1, 0, -regret)

	if xlabel is not None:
		plt.xlabel(xlabel)

	if ylabel is not None:
		plt.ylabel(ylabel)

	if title is not None:
		plt.title(title)

	if show:
		plt.show()

		
def nrows_ncols(N):
    """(nrows, ncols) pour créer un subplots de N figures avec les bonnes dimensions."""
    nrows = int(np.ceil(np.sqrt(N)))
    ncols = N // nrows
    while N > nrows * ncols:
        ncols += 1
    nrows, ncols = max(nrows, ncols), min(nrows, ncols)
    return nrows, ncols
	
def plot_hist_regret(rewards, names, horizon, mustar=1):
    nrows, ncols = nrows_ncols(len(names))
    fig, axes = plt.subplots(nrows, ncols, sharex=False, sharey=False)
    fig.suptitle("Histogram of regret at $t = T = {}$".format(horizon))

    # XXX See https://stackoverflow.com/a/36542971/
    ax0 = fig.add_subplot(111, frame_on=False)  # add a big axes, hide frame
    ax0.grid(False)  # hide grid
    #ax0.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')  # hide tick and tick label of the big axes
    ax0.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)  # hide tick and tick label of the big axes
    # Add only once the ylabel, xlabel, in the middle
    ax0.set_ylabel("Distribution")
    ax0.set_xlabel("Regret")

    for i, r in enumerate(rewards):
        x, y = i % nrows, i // nrows
        ax = axes[x, y] if ncols > 1 else axes[x]
        regret = mustar * horizon - r
        #ax.hist(regret, normed=True, bins=25)
        ax.hist(regret, density=True, bins=25)
        ax.set_title(names[i])
    plt.show()
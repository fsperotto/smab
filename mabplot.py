import matplotlib.pyplot as plt
import numpy as np

class mabplt:

    """ 
    Constructor
     M : the MAB simulation
    """
    def __init__(self, M):
        self.M = M


    """ 
    Plot the graph
    """
    def _call_plot(self, xlabel=None, ylabel=None, title=None, names=None, filename=None, show=True):

        if names is not None:
            plt.legend(names)

        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)

        if title is not None:
            plt.title(title)

        if filename is not None:
            plt.tight_layout()
            plt.savefig(filename)

        if show:
            plt.show()


    """ 
    Plot a line graph
    """
    def _plot_progression(self, Y, X=None, names=None, linestyles=None, linecolors=None, xlabel="$t$", ylabel="Value", reorder='desc', showlast=False, title=None, filename=None, show=True):

        #number of algorithms
        m = len(Y)
        
        if (names is not None):
            names = np.pad(np.array(names[:m]), (0, max(0, m-len(names))), 'wrap') 
        if (linestyles is not None):
            linestyles = np.pad(np.array(linestyles[:m]), (0, max(0, m-len(linestyles))), 'wrap') 
        if (linecolors is not None):
            linecolors = np.pad(np.array(linecolors[:m]), (0, max(0, m-len(linecolors))), 'wrap') 
        
        if Y.ndim > 1:

            #ordering
            if reorder is not None:
                idx=np.argsort(Y[:,-1])
                if reorder == 'desc':
                    idx = idx[::-1]
                Y=Y[idx]
                if names is not None:
                    names = names[idx]
                if linestyles is not None:
                    linestyles = linestyles[idx]
                if linecolors is not None:
                    linecolors = linecolors[idx]

            if X is None:
                X = range(len(Y[0]))

            for i, Y_i in enumerate(Y):
                line, = plt.plot(X, Y_i)
                if linestyles is not None:
                    line.set_linestyle(linestyles[i % len(linestyles)])
                if linecolors is not None:
                    line.set_color(linecolors[i % len(linecolors)])
                if showlast:
                    plt.annotate('%0.2f'%Y_i[-1], xy=(1, Y_i[-1]), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')                    

        else:
            if X is None:
                X = range(len(Y))
            plt.plot(X, Y)

        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, names=names, filename=filename, show=show)


    """ 
    Plot a bar graph (series correspond to arms)
    """
    def _plot_comp_arms(self, Y, E=None, names=None, xlabel="Arms", ylabel=None, title="Arms Comparison", filename=None, show=True):

        if Y.ndim > 1:
            w = 0.8 / len(Y)
            for i, Y_i in enumerate(Y):
                yerr = None if E==None else E[i]
                plt.bar(self.M.K1 + w*i, Y_i, yerr=yerr, width=w)   #color=plt.cm.get_cmap(i)
        else:
            plt.bar(self.M.K1, Y, yerr=E)

        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, names=names, filename=filename, show=show)


    """ 
    Plot the history (temporal map) of actions taken (series correspond to arms)
     i : repetition
     j : algorithm 
    """
    def plot_history(self, i=None, j=None, xlabel='$t$', ylabel='Arm', title='History of pulled arms', alpha=0.5, markersize=None, filename=None, show=True):

        if (i is None) and (self.M.n == 1):
            i=0
        else:    
            raise("parameter i is needed.")

        if (j is None) and (self.M.m == 1):
            j=0
        else:    
            raise("parameter j is needed.")

        plt.plot(self.M.T1, self.M.H1[i,j], 'o', markersize=markersize, alpha=alpha)

        plt.yticks(self.M.K1)
        plt.ylim([0.5, self.M.k+0.5])
        plt.gca().invert_yaxis()    

        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, filename=filename, show=show)


    """ 
    Plot the progression of the actions counter (series correspond to arms)
     i : repetition (None for an average over repetitions)
     j : algorithm 
    """
    def plot_action_count_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Number of pulls", title="Arm pull counter", linestyles=None, linecolors=None, show=True):

        #verify parameters
        if j is None:
            print("No multialgorithm implementation. Algorithm must be indicated. First algorithm we be displayed.")
            j = 0

        #prepare labels
        if names is None:
            names = [f"$N_{a}$" for a in self.M.K1]

        #prepare data
        if i is None:
            Y = self.M.MN_a[j]
        else:
            Y = self.M.N_a[i,j]

        #add zeros at time zero
        Z = np.reshape(np.zeros(self.M.k, dtype='int'), [self.M.k, 1])
        Y = np.block([Z, Y])

        #call plot progression
        self._plot_progression(Y, X=self.M.T01, names=names, ylabel=ylabel, xlabel=xlabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    """ 
    Plot the progression of the actions frequency (series correspond to arms)
     i : repetition (None for an average over repetitions)
     j : algorithm 
    """
    def plot_action_freq_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Pull Frequency", title="Arm Selection Frequency", linestyles=None, linecolors=None, show=True):

        #verify parameters
        if j is None:
            print("No multialgorithm implementation. Algorithm must be indicated. First algorithm we be displayed.")
            j = 0

        #prepare labels
        if names is None:
            names = [f"$F_{a}$" for a in self.M.K1]

        #prepare data
        if i is None:
            Y = self.M.MF_a[j]  #averaged over repetitions
        else:
            Y = self.M.F_a[i,j]  

        #call plot progression
        self._plot_progression(Y, X=self.M.T1, names=names, ylabel=ylabel, xlabel=xlabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    """ 
    Plot the progression of the precision (frequency of the best arm)
     i : repetition (None for an average over repetitions)
     j : algorithm (None for all)
    """
    def plot_precision_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Best Arm Pull Frequency", title="Precision", linestyles=None, linecolors=None, show=True):

        #all algorithms
        if j is None:
            if names is None:
                names=[str(g) for g in self.M.G]		
            if i is None:  #average over repetitions
                Y = self.M.MF_a[:,self.M.a_star]
            else:  #specific repetition
                Y = self.M.F_a[i,:,self.M.a_star]
        #specific algorithm
        else:
            if names is None:
                names=['best', 'others']		
            #prepare best and others frequencies
            if i is None:
                Y = np.array([self.M.MF_a[j, self.M.a_star], 1-self.M.MF_a[j, self.M.a_star]])
            else:
                Y = np.array([self.M.F_a[i, j, self.M.a_star], 1-self.M.F_a[i, j, self.M.a_star]])

        #call plot progression
        self._plot_progression(Y, X=self.M.T1, names=names, ylabel=ylabel, xlabel=xlabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    """ 
    Plot the progression of the cumulated reward (sum of rewards over time)
     i : repetition (None for an average over repetitions)
     j : algorithm (None for all)
    """
    def plot_cumulated_reward_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Cumulated Reward", title="Cumulated Reward", linestyles=None, linecolors=None, show=True):

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.M.G]
            if i is None:  #averaged over repetitions      
                Y = self.M.MSR
            else:		   #in a specific repetition
                Y = self.M.SR[i]
            Z = np.reshape(np.zeros(self.M.m, dtype='float'), [self.M.m, 1])
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.M.G[j])]
            if i is None:  #averaged over repetitions      
                Y = self.M.MSR[j]
            else: 		   #in a specific repetition
                Y = self.M.SR[i,j]
            Z = np.array([0.0], dtype='float')

        #add zeros at time zero
        Y = np.block([Z, Y])

        #call plot		
        self._plot_progression(Y, X=self.M.T01, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    def plot_budget_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Budget", title="Budget", linestyles=None, linecolors=None, show=True):		

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.M.G]
            if i is None:  #averaged over repetitions      
                Y = self.M.MB
            else:		   #in a specific repetition
                Y = self.M.B[i]
            Z = np.reshape(np.repeat(self.M.b_0, self.M.m), [self.M.m, 1])
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.M.G[j])]
            if i is None:  #averaged over repetitions      
                Y = self.M.MB[j]
            else: 		   #in a specific repetition
                Y = self.M.B[i,j]
            Z = np.array([self.M.b_0], dtype='float')

        #add zeros at time zero
        Y = np.block([Z, Y])

        #call plot
        self._plot_progression(Y, X=self.M.T01, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)

        
    def plot_survival_progression(self, j=None, names=None, xlabel="$t$", ylabel="Survival", title="Survival", linestyles=None, linecolors=None, show=True):

        #prepare data
        if j is None:  #comparing algorithms
            Z = np.reshape(np.ones(self.M.m, dtype='float'), [self.M.m, 1])
            Y = self.M.SC
            if names is None:
                names=[str(g) for g in self.M.G]
        else:	#specific algorithm		   
            Z = np.array([1.0], dtype='float')
            Y = self.M.A[j]
            if names is None:
                names=[str(self.M.G[j])]

        #add zeros at time zero
        Y = np.block([Z, Y])

        #call plot		
        self._plot_progression(Y, X=self.M.T01, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)

        
    def plot_negative_budget_time_map(self, i=None, j=None, names=None, inibudget=0.0, xlabel="$t$", ylabel="Time on Negative Budget", title="Time on Negative Budget", linestyles=None, linecolors=None, show=True):

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.M.G]
            if i is None:  #averaged over repetitions      
                Y = self.M.TNMB #MSR
            else:		   #in a specific repetition
                Y = self.M.TNB[i]
            ##for i, Y_i in enumerate(Y):
            ##	Y[i] = np.array([1 if(v<inibudget) else 0 for v in Y_i])
            #Y = np.array([[1 if(v<-inibudget) else 0 for v in Y_i] for Y_i in Y])
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.M.G[j])]
            if i is None:  #averaged over repetitions      
                Y = self.M.TNMB[j]
            else: 		   #in a specific repetition
                Y = self.M.TNB[i,j]
            #Y = np.array([1 if(v<-inibudget) else 0 for v in Y])
        #call plot		
        self._plot_progression(Y, X=self.M.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    def plot_negative_budget_time_progression(self, i=None, j=None, names=None, inibudget=0.0, xlabel="$t$", ylabel="Cumulated Time on Negative Budget", title="Cumulated Time on Negative Budget", linestyles=None, linecolors=None, show=True):

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.M.G]
            if i is None:  #averaged over repetitions      
                Y = self.M.STNMB #MSR
            else:		   #in a specific repetition
                Y = self.M.STNB[i]
            #Y = np.array([[-1 if(v<-inibudget) else 0 for v in Y_i] for Y_i in Y])
            #Y = np.cumsum(Y, axis=1, dtype='float')
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.M.G[j])]
            if i is None:  #averaged over repetitions      
                Y = self.M.STNMB[j]
            else: 		   #in a specific repetition
                Y = self.M.STNB[i,j]
            #Y = np.array([-1 if(v<-inibudget) else 0 for v in Y])
            #Y = np.cumsum(Y, axis=0, dtype='float')

        #call plot		
        self._plot_progression(Y, X=self.M.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    def plot_negative_budget_progression(self, i=None, j=None, names=None, inibudget=0.0, xlabel="$t$", ylabel="Negative Budget", title="Negative Budget", linestyles=None, linecolors=None, show=True):

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.M.G]
            if i is None:  #averaged over repetitions      
                Y = self.M.NMB
            else:		   #in a specific repetition
                Y = self.M.NB[i]
            #Y = np.array([[v+inibudget if(v<-inibudget) else 0 for v in Y_i] for Y_i in Y])
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.M.G[j])]
            if i is None:  #averaged over repetitions      
                Y = self.M.NMB[j]
            else: 		   #in a specific repetition
                Y = self.M.NB[i,j]
            #Y = np.array([v+inibudget if(v<-inibudget) else 0 for v in Y])

        #call plot		
        self._plot_progression(Y, X=self.M.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)

    def plot_cumulated_negative_budget_progression(self, i=None, j=None, names=None, inibudget=0.0, xlabel="$t$", ylabel="Cumulated Negative Budget", title="Cumulated Negative Budget", linestyles=None, linecolors=None, show=True):

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.M.G]
            if i is None:  #averaged over repetitions      
                Y = self.M.SNMB
            else:		   #in a specific repetition
                Y = self.M.SNB[i]
            #Y = np.array([[v+inibudget if(v<-inibudget) else 0 for v in Y_i] for Y_i in Y])
            #Y = np.cumsum(Y, axis=1, dtype='float')
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.M.G[j])]
            if i is None:  #averaged over repetitions      
                Y = self.M.SNMB[j]
            else: 		   #in a specific repetition
                Y = self.M.SNB[i,j]
            #Y = np.array([v+inibudget if(v<-inibudget) else 0 for v in Y])
            #Y = np.cumsum(Y, axis=0, dtype='float')

        #call plot		
        self._plot_progression(Y, X=self.M.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    def plot_average_reward_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Average Reward", title="Average Reward", linestyles=None, linecolors=None, show=True):	

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.M.G]
            if i is None:  #averaged over repetitions      
                Y = self.M.MMR
            else:		   #in a specific repetition
                Y = self.M.MR[i]
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.M.G[j])]
            if i is None:  #averaged over repetitions      
                Y = self.M.MMR[j]
            else: 		   #in a specific repetition
                Y = self.M.MR[i,j]

        #call plot		
        self._plot_progression(Y, X=self.M.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    def plot_cumulated_reward_regret_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Cumulated Reward and Regret", title="Cumulated Reward and Regret", show=True):	

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.M.G]
            if i is None:  #averaged over repetitions      
                Y1 = self.M.MSR
                Y2 = -self.M.MSL
            else:		   #in a specific repetition
                Y1 = self.M.SR[i]
                Y2 = -self.M.SL[i]
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.M.G[j])]
            if i is None:  #averaged over repetitions      
                Y1 = self.M.MSR[j]
                Y2 = -self.M.MSL[j]
            else: 		   #in a specific repetition
                Y1 = self.M.SR[i,j]
                Y2 = -self.M.SL[i,j]

        #call plot		
        self._plot_progression(Y1, X=self.M.T1, names=None, xlabel=None, ylabel=None, title=None, show=False)
        self._plot_progression(Y2, X=self.M.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, show=show, linestyles=['--']*self.M.m, linecolors=['C' + str(j) for j in range(self.M.m)])


    def plot_average_reward_regret_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Average Reward and Regret", title="Average Reward and Regret", show=True):	

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.M.G]
            if i is None:  #averaged over repetitions      
                Y1 = self.M.MMR
                Y2 = -self.M.MML
            else:		   #in a specific repetition
                Y1 = self.M.MR[i]
                Y2 = -self.M.ML[i]
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.M.G[j])]
            if i is None:  #averaged over repetitions      
                Y1 = self.M.MMR[j]
                Y2 = -self.M.MML[j]
            else: 		   #in a specific repetition
                Y1 = self.M.MR[i,j]
                Y2 = -self.M.ML[i,j]

        #call plot		
        self._plot_progression(Y1, X=self.M.T1, names=None, xlabel=None, ylabel=None, title=None, show=False)
        self._plot_progression(Y2, X=self.M.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, show=show, linestyles=['--']*self.M.m, linecolors=['C' + str(j) for j in range(self.M.m)])


    def plot_cumulated_regret_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Cumulated Regret", title="Cumulated Regret", linestyles=None, linecolors=None, show=True, inverse=False):

        w = -1 if inverse else 1
        
        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.M.G]
            if i is None:  #averaged over repetitions      
                Y = w*self.M.MSL
            else:		   #in a specific repetition
                Y = w*self.M.SL[i]
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.M.G[j])]
            if i is None:  #averaged over repetitions      
                Y = w*self.M.MSL[j]
            else: 		   #in a specific repetition
                Y = w*self.M.SL[i,j]

        #call plot		
        self._plot_progression(Y, X=self.M.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    def plot_average_regret_progression(self, i=None, j=None, names=None, xlabel="$t$", ylabel="Average Regret", title="Average Regret", linestyles=None, linecolors=None, show=True, inverse=False):	

        w = -1 if inverse else 1

        #prepare data
        if j is None:  #comparing algorithms
            if names is None:
                names=[str(g) for g in self.M.G]
            if i is None:  #averaged over repetitions      
                Y = w*self.M.MML
            else:		   #in a specific repetition
                Y = w*self.M.ML[i]
        else:	#specific algorithm		   
            if names is None:
                names=[str(self.M.G[j])]
            if i is None:  #averaged over repetitions      
                Y = w*self.M.MML[j]
            else: 		   #in a specific repetition
                Y = w*self.M.ML[i,j]

        #call plot
        self._plot_progression(Y, X=self.M.T1, names=names, xlabel=xlabel, ylabel=ylabel, title=title, linestyles=linestyles, linecolors=linecolors, show=show)


    def plot_freq_spectrum(self, F, bar_size=2, interpolation='hanning', cmap='gray_r', xlabel="$t$", ylabel="Actions", title="Frequency Spectrum", filename=None, show=True):

        bs = max(bar_size, 2)   #bar size (>1)
        h = self.M.k*bs+1     	#fig height (rows depends on the number of arms and bar size)
        w = len(F[0])			#fig width  (columns depends on time)

        img_map = np.zeros([h,w])   #image map

        for c in range(h):
            if (c % bs != 0):
                img_map[c] = F[c//bs]

        plt.imshow(img_map, aspect="auto", interpolation=interpolation, cmap=cmap)
        plt.yticks(np.arange(1, h, step=bs), self.M.K1)

        ##ax = plt.gca()
        #plt.xticks(np.arange(-.5, 10, 1))
        #plt.xticklabels(np.arange(1, 12, 1))

        plt.colorbar()

        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, filename=filename, show=show)



    def plot_action_window_freq_spectrum(self, i=None, j=None, bar_size=2, interpolation='none', cmap='gray_r', xlabel="$t$", ylabel="Arms", title="Arm Pull Local Frequency Spectrum", show=True):

        #verify parameters
        if j is None:
            print("No multialgorithm implementation. Algorithm must be indicated. First algorithm we be displayed.")
            j = 0

        #add zeros at time zero
        Z = np.reshape(np.zeros(self.M.k, dtype='float'), [self.M.k, 1])
        if i is None:
            Y = np.block([Z, self.M.MFW_a[j]])
        else:
            Y = np.block([Z, self.M.FW_a[i,j]])

        #call plot
        self.plot_freq_spectrum(Y, bar_size=bar_size, interpolation=interpolation, cmap=cmap, xlabel=xlabel, ylabel=ylabel, title=title)		


    def plot_comp_arm_count(self, i=None, j=None, xlabel="Arm (Selected Action)", ylabel="Number of Actions Taken", title="Selected Actions", show=True):

        if j is None:  #comparing algorithms
            names=[str(g) for g in self.M.G]
            if i is None:   #averaging on repetitions
                Y = self.M.mn_a
            else:           #specific repetition
                Y = self.M.n_a[i]
        else:          #specific algorithm
            names=None
            if i is None:   #averaging on repetitions
                Y = self.M.mn_a[j]
            else:           #specific repetition
                Y = self.M.n_a[i,j]

        #call plot
        self._plot_comp_arms(Y, names=names, xlabel=xlabel, ylabel=ylabel, title=title)


    def plot_comp_arm_rewards(self, i=None, j=None, xlabel="Arms", ylabel="Total Rewards", title="Total Rewards per Arm", show=True):

        if j is None:  #comparing algorithms
            names=[str(g) for g in self.M.G]
            if i is None:
                Y = self.M.msr_a
            else:
                Y = self.M.sr_a[i]
        else:          #specific algorithm
            names=None
            if i is None:
                Y = self.M.msr_a[j]
            else:
                Y = self.M.sr_a[i,j]

        #call plot
        self._plot_comp_arms(Y, names=names, xlabel=xlabel, ylabel=ylabel, title=title)


    def _call_plot_comp_algs(self, Y, E=None, xlabel="Algorithms", ylabel="Value", title="Comparison", sort=True, names=None, names_rotation='vertical', bar_labels=False, compact_view=True, filename=None, show=True):

        x = np.arange(self.M.m, dtype='int')
        
        if names is None:
            names = [str(g) for g in self.M.G]

        if compact_view:
            low = min(Y)
            high = max(Y)
            plt.ylim([low, high])

        #sort
        if sort:
            idx = np.argsort(Y)[::-1]  #desc order
            Y = Y[idx]
            names = [names[i] for i in idx]

        plt.xticks(x, names, rotation=names_rotation)
        plt.bar(x, Y, yerr=E, align='center', alpha=0.5)

        #bar labels
        if bar_labels:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')
        
        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, filename=filename, show=show)
        
        
    def plot_comp_algs_total_rewards(self, i=None, xlabel="Algorithm", ylabel="Total Reward", title="Comparison (Total Reward)", sort=True, names=None, names_rotation='vertical', bar_labels=False, compact_view=True, filename=None, show=True):

        Y = self.M.msr if i is None else self.M.sr[i]
        self._call_plot_comp_algs(Y, xlabel=xlabel, ylabel=ylabel, title=title, sort=sort, names=names, names_rotation=names_rotation, bar_labels=bar_labels, compact_view=compact_view, filename=filename, show=show)


    def plot_comp_algs_survival_time(self, i=None, xlabel="Algorithm", ylabel="Average Survival Time", title="Comparison (Average Survival Time)", sort=True, names=None, names_rotation='vertical', bar_labels=False, compact_view=True, filename=None, show=True):

        Y = self.M.MTTNB if i is None else self.M.TTNB[i]
        E = self.M.DTTNB if i is None else None
        self._call_plot_comp_algs(Y, E=E, xlabel=xlabel, ylabel=ylabel, title=title, sort=sort, names=names, names_rotation=names_rotation, bar_labels=bar_labels, compact_view=compact_view, filename=filename, show=show)


    def plot_comp_algs_ruined_episodes(self, xlabel="Algorithm", ylabel="Survival Episodes", title="Comparison (Survival Episodes)", sort=True, names=None, names_rotation='vertical', bar_labels=False, compact_view=True, filename=None, show=True):

        Y = self.M.n - self.M.senb
        self._call_plot_comp_algs(Y, xlabel=xlabel, ylabel=ylabel, title=title, sort=sort, names=names, names_rotation=names_rotation, bar_labels=bar_labels, compact_view=compact_view, filename=filename, show=show)


    def plot_comp_algs_cumulated_negative_budget(self, i=None, xlabel="Algorithm", ylabel="Cumulated Negative Budget", title="Comparison", sort=True, names=None, names_rotation='vertical', bar_labels=False, compact_view=True, filename=None, show=True):

        #Y = self.M.snmb if i is None else self.M.snb[i]
        #E = self.M.dnmb if i is None else self.M.dnb[i]
        Y = self.M.msnb if i is None else self.M.snb[i]
        E = self.M.dsnb if i is None else None
        self._call_plot_comp_algs(Y, E=E, xlabel=xlabel, ylabel=ylabel, title=title, sort=sort, names=names, names_rotation=names_rotation, bar_labels=bar_labels, compact_view=compact_view, filename=filename, show=show)

    def plot_comp_algs_cumulated_regret(self, i=None, xlabel="Algorithm", ylabel="Cumulated Regret", title="Comparison", sort=True, names=None, names_rotation='vertical', bar_labels=False, compact_view=True, filename=None, show=True):

        Y = self.M.msl if i is None else self.M.sl[i]
        E = self.M.dsl if i is None else None
        self._call_plot_comp_algs(Y, E=E, xlabel=xlabel, ylabel=ylabel, title=title, sort=sort, names=names, names_rotation=names_rotation, bar_labels=bar_labels, compact_view=compact_view, filename=filename, show=show)



    def plot_comp_freq_prop(self, i=None, j=None, names=['Cumulated Reward Proportion', 'Pull Frequency'], xlabel="$t$", ylabel="Cumulated Reward Proportion and Pull Frequency", title="Cumulated Reward and Number of Pulls", filename=None, show=True):

        #verify parameters
        if j is None:
            print("No multialgorithm implementation. Algorithm must be indicated. First algorithm we be displayed.")
            j = 0

        if i is None:
            plt.bar(self.M.K1, self.M.mfr_a[j], width=0.8)
            plt.bar(self.M.K1, self.M.mf_a[j], width=0.4, alpha=0.5)
        else:
            plt.bar(self.M.K1, self.M.fr_a[i,j], width=0.8)
            plt.bar(self.M.K1, self.M.f_a[i,j], width=0.4, alpha=0.5)

        if names is not None:
            plt.legend(names)

        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, filename=filename, show=show)


    def plot_reward_regret(self, i=None, j=None, names=['Cumulated Reward', 'Cumulated Regret', 'Best Strategy', 'Worst Possible Regret'], xlabel="$t$", ylabel="Reward and Regret", title="Best Strategy vs Cumulated Reward vs Regret ", filename=None, show=True):

        #verify parameters
        if j is None:
            print("No multialgorithm implementation. Algorithm must be indicated. First algorithm we be displayed.")
            j = 0

        #best policy
        SS = np.linspace(self.M.mu_star, self.M.mu_star*self.M.tau, self.M.tau)
        #worst regret
        l = -(self.M.mu_star-self.M.mu_worst)
        #worst cumulated regret
        SW = np.linspace(l, l*self.M.tau, self.M.tau)

        #best, reward, regret
        if i is None:
            Y = np.array([self.M.MSR[j], -self.M.MSL[j], SS, SW])
        else:
            Y = np.array([self.M.SR[i,j], -self.M.SL[i,j], SS, SW])

        self._plot_progression(Y, X=self.M.T1, show=False, names=names, title=title, ylabel=ylabel)

        if i is None:
            plt.fill_between(self.M.T1, 0, self.M.MSR[j], alpha=0.5)
            plt.fill_between(self.M.T1, 0, -self.M.MSL[j], alpha=0.5)
        else:
            plt.fill_between(self.M.T1, 0, self.M.SR[i,j], alpha=0.5)
            plt.fill_between(self.M.T1, 0, -self.M.SL[i,j], alpha=0.5)

        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, filename=filename, show=show)


    def plot_survival_histogram(self, j=None, xlabel="$t$", ylabel="Time before ruin", title="Survival Time Histogram", filename=None, show=True):

        #verify parameters
        if j is None:
            print("No multialgorithm implementation. Algorithm must be indicated. First algorithm we be displayed.")
            j = 0

        #prepare histogram bins in log scale
        logbins=np.geomspace(10, self.M.tau+1, 50, endpoint=True, dtype='int')

        #prepare data
        Y=self.M.TTNB.transpose()[j]

        plt.hist(Y, bins=logbins) #log=True
        plt.xscale('log')

        self._call_plot(xlabel=xlabel, ylabel=ylabel, title=title, filename=filename, show=show)


def plot_gaussian_distributions(means, sigmas, minr, maxr):

    #show distributions
    x = np.linspace(minr, maxr, 1000)
    idx = np.argsort(means)[::-1] #order
    #for i, mu in enumerate(means):
    for i in idx:
        mu = means[i]
        sigma = sigmas[i]
        plt.plot(x, mlab.normpdf(x, mu, sigma), label="$\mu_{" + str(i+1) + "}=" + str(mu) + "$")
    #plt.legend()
    plt.show()

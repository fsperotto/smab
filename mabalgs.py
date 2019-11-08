import numpy as np
from numba import jit
from math import sqrt, log, factorial
from scipy.stats import beta
from scipy.special import binom
from scipy.special import beta as beta_func
from scipy.special import betainc as reg_inc_beta_func
from scipy.special import gamma as gamma_func
#from scipy.integrate import quad as integral
#from scipy.integrate import fixed_quad as integral
from scipy.integrate import quadrature as integral
from mpmath import hyp2f1 as hypergeo_func
from SMPyBandits.Policies import EpsilonGreedy, EpsilonDecreasing, UCB, UCBalpha, klUCB, BayesUCB, BasePolicy, IndexPolicy
from SMPyBandits.Policies.BasePolicy import BasePolicy
from SMPyBandits.Policies.IndexPolicy import IndexPolicy


#####################################################


class ClassicEpsilonGreedy(EpsilonGreedy):

    def __str__(self):
        return f"Epsilon-greedy($\epsilon={self._epsilon}$)"
    
    def __init__(self, nbArms, epsilon=0.1, lower=0., amplitude=1.):
        super(ClassicEpsilonGreedy, self).__init__(nbArms, epsilon=epsilon, lower=lower, amplitude=amplitude)
  
    def choice(self):
        # Generate random number
        p = np.random.rand()
        """With a probability of epsilon, explore (uniform choice), otherwise exploit based on empirical mean rewards."""
        if p < self.epsilon: # Proba epsilon : explore
            #return np.random.randint(0, self.nbArms - 1)
            return np.random.randint(0, self.nbArms)
        else:  # Proba 1 - epsilon : exploit
            # Uniform choice among the best arms
            #biased_means = self.rewards / (1 + self.pulls)
            estimated_means = self.rewards / np.maximum(1, self.pulls)
            return np.random.choice(np.flatnonzero(estimated_means == np.max(estimated_means)))


class ClassicEpsilonDecreasing(ClassicEpsilonGreedy):
    r""" The epsilon-decreasing random policy.

    - :math:`\varepsilon(t) = \min(1, \varepsilon_0 / \max(1, t))`
    - Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, epsilon=1.0, lower=0., amplitude=1.):
        super(ClassicEpsilonDecreasing, self).__init__(nbArms, epsilon=epsilon, lower=lower, amplitude=amplitude)

    def __str__(self):
        return f"EpsilonDecreasing({self._epsilon})"

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def epsilon(self):
        r"""Decreasing :math:`\varepsilon(t) = \min(1, \varepsilon_0 / \max(1, t))`."""
        return min(1, self._epsilon / max(1, self.t))


class ClassicOptimisticGreedy(ClassicEpsilonGreedy):
    
    def __init__(self, nbArms, epsilon=0.0, init_estimation=10.0, lower=0., amplitude=1.):
        super(ClassicEpsilonGreedy, self).__init__(nbArms, epsilon=epsilon, lower=lower, amplitude=amplitude)
        #self.estimated_means = np.repeat(init_estimation, nbArms)
        self.init_estimation = init_estimation

    def __str__(self):
        return f"OptimisticGreedy({self.init_estimation})"
        
    def choice(self):
        # Generate random number
        p = np.random.rand()
        """With a probability of epsilon, explore (uniform choice), otherwhise exploit based on empirical mean rewards."""
        if p < self.epsilon: # Proba epsilon : explore
            #return np.random.randint(0, self.nbArms - 1)
            return np.random.randint(0, self.nbArms)
        else:  # Proba 1 - epsilon : exploit
            # Uniform choice among the best arms
            estimated_means = (self.rewards + self.init_estimation) / (self.pulls + 1)
            return np.random.choice(np.flatnonzero(estimated_means == np.max(estimated_means)))

#####################################################


class Budgeted:

    def __init__(self, inibudget=10.0, min_r=-1.0, max_r=+1.0):
        self.inibudget=inibudget
        self.min_r = min_r
        self.max_r = max_r
        self.delta_r = max_r - min_r
        self.budget = inibudget
        self.totalreward = 0.0
        
    def startGame(self):
        self.budget = self.inibudget
        self.totalreward = 0.0

    def getReward(self, reward):
        self.totalreward += reward
        self.budget += reward * self.delta_r + self.min_r

#####################################################


class Estimator:

    def __init__(self, nbArms):
        #self.totalreward = 0.0
        self.estmeans = np.zeros(nbArms)

    def startGame(self):
        #self.totalreward = 0.0
        self.estmeans.fill(0.0)

    def getReward(self, arm, reward):
        #self.totalreward += reward
        self.estmeans[arm] = (self.estmeans[arm] * (self.pulls[arm]-1) + reward) / self.pulls[arm]


class BernoulliEstimator(Estimator):

    def __init__(self, nbArms):
        Estimator.__init__(self, nbArms)
        self.successes = np.zeros(nbArms, dtype='int')

    def startGame(self):
        Estimator.startGame(self)
        self.successes.fill(0)

    def getReward(self, arm, reward):
        Estimator.getReward(self, arm, reward)
        if (reward > 0):
            self.successes[arm] += 1



#####################################################


class SafeAlg(Budgeted, Estimator):

    def __init__(self, nbArms, inibudget=10.0, min_r=-1.0, max_r=+1.0, safebudget=1.0):
        Budgeted.__init__(self, inibudget=inibudget, min_r=min_r, max_r=max_r)
        Estimator.__init__(self, nbArms)
        self.safebudget = safebudget

    def startGame(self):
        Budgeted.startGame(self)
        Estimator.startGame(self)

    def getReward(self, arm, reward):
        Budgeted.getReward(self, reward)
        Estimator.getReward(self, arm, reward)

    def choice(self):
        #sufficient budget
        if self.budget > self.safebudget:
            return None
        #low budget
        else:
            if np.max(self.estmeans) > 0:
                # Uniform choice among the best arms
                return np.random.choice(np.flatnonzero(self.estmeans == np.max(self.estmeans)))
            else:
                return None

#####################################################


class SafeUCB(UCB, SafeAlg):

    def __str__(self):
        return f"Safe-UCB($b_s={self.safebudget}$)"

    def __init__(self, nbArms, inibudget=10.0, safebudget=1.0, min_r=-1.0, max_r=+1.0, lower=0.0, amplitude=1.0):
        UCB.__init__(self, nbArms, lower=lower, amplitude=amplitude)
        SafeAlg.__init__(self, nbArms, inibudget=inibudget, min_r=min_r, max_r=max_r, safebudget=safebudget)

    def startGame(self):
        UCB.startGame(self)
        SafeAlg.startGame(self)

    def getReward(self, arm, reward):
        UCB.getReward(self, arm, reward)
        SafeAlg.getReward(self, arm, reward)

    def choice(self):
        r = SafeAlg.choice(self)
        if r is None:
            r = UCB.choice(self)
        return r


class SafeKLUCB(klUCB, SafeAlg):

    def __str__(self):
        return f"Safe-KL-UCB($b_s={self.safebudget}$)"

    def __init__(self, nbArms, inibudget=10.0, safebudget=1.0, min_r=-1.0, max_r=+1.0, lower=0.0, amplitude=1.0):
        klUCB.__init__(self, nbArms, lower=lower, amplitude=amplitude)
        SafeAlg.__init__(self, nbArms, inibudget=inibudget, min_r=min_r, max_r=max_r, safebudget=safebudget)

    def startGame(self):
        klUCB.startGame(self)
        SafeAlg.startGame(self)

    def getReward(self, arm, reward):
        klUCB.getReward(self, arm, reward)
        SafeAlg.getReward(self, arm, reward)

    def choice(self):
        r = SafeAlg.choice(self)
        if r is None:
            r = klUCB.choice(self)
        return r


class SafeUCBalpha(UCBalpha, SafeAlg):

    def __str__(self):
        return f"Safe-UCB($a={self.alpha}, b_s={self.safebudget}$)"
        #return r"UCB($\alpha={:.3g}$)".format(self.alpha)

    def __init__(self, nbArms, alpha=4.0, inibudget=10.0, safebudget=1.0, min_r=-1.0, max_r=+1.0, lower=0.0, amplitude=1.0):
        UCBalpha.__init__(self, nbArms, alpha=alpha, lower=lower, amplitude=amplitude)
        SafeAlg.__init__(self, nbArms, inibudget=inibudget, min_r=min_r, max_r=max_r, safebudget=safebudget)

    def startGame(self):
        UCBalpha.startGame(self)
        SafeAlg.startGame(self)

    def getReward(self, arm, reward):
        UCBalpha.getReward(self, arm, reward)
        SafeAlg.getReward(self, arm, reward)

    def choice(self):
        r = SafeAlg.choice(self)
        if r is None:
            r = UCBalpha.choice(self)
        return r


class SafeEpsilonGreedy(ClassicEpsilonGreedy, SafeAlg):

    def __str__(self):
        return f"Safe-$\epsilon$-greedy($\epsilon={self._epsilon}, b_s={self.safebudget}$)"

    def __init__(self, nbArms, epsilon=0.1, inibudget=10.0, safebudget=1.0, min_r=-1.0, max_r=+1.0, lower=0.0, amplitude=1.0):
        ClassicEpsilonGreedy.__init__(self, nbArms, epsilon=epsilon, lower=lower, amplitude=amplitude)
        SafeAlg.__init__(self, nbArms, inibudget=inibudget, min_r=min_r, max_r=max_r, safebudget=safebudget)

    def startGame(self):
        ClassicEpsilonGreedy.startGame(self)
        SafeAlg.startGame(self)

    def getReward(self, arm, reward):
        ClassicEpsilonGreedy.getReward(self, arm, reward)
        SafeAlg.getReward(self, arm, reward)

    def choice(self):
        r = SafeAlg.choice(self)
        if r is None:
            r = ClassicEpsilonGreedy.choice(self)
        return r

    
#####################################################

    
class GamblerBayesUCB(BayesUCB, Budgeted):
    """ The Bayes-UCB policy, replacing $t$ by $b$.

    - By default, it uses a Beta posterior (:class:`Policies.Posterior.Beta`), one by arm.
    -Reference: [Kaufmann, Cappé & Garivier - AISTATS, 2012].
    """

    def __init__(self, nbArms, inibudget=10.0, min_r=-1.0, max_r=+1.0, lower=0.0, amplitude=1.0):
        BayesUCB.__init__(self, nbArms, lower=lower, amplitude=amplitude)
        Budgeted.__init__(self, inibudget=inibudget, min_r=min_r, max_r=max_r)

    def startGame(self):
        BayesUCB.startGame(self)
        Budgeted.startGame(self)

    def getReward(self, arm, reward):
        BayesUCB.getReward(self, arm, reward)
        Budgeted.getReward(self, reward)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, giving :math:`S_k(t)` rewards of 1, by taking the :math:`1 - \frac{1}{\min(b, 2)}` quantile from the Beta posterior:

        .. math:: I_k(t) = \mathrm{Quantile}\left(\mathrm{Beta}(1 + S_k(t), 1 + N_k(t) - S_k(t)), 1 - \frac{1}{\min(b,2)}\right).
        """
        return self.posterior[arm].quantile(1. - 1. / (1 + min2, self.budget))
    

    
class PositiveGamblerUCB(UCB, Budgeted, BernoulliEstimator):
    
    def __init__(self, nbArms, inibudget=10.0, min_r=-1.0, max_r=+1.0, lower=0.0, amplitude=1.0):
        UCB.__init__(self, nbArms, lower=lower, amplitude=amplitude)
        Budgeted.__init__(self, inibudget=inibudget, min_r=min_r, max_r=max_r)
        BernoulliEstimator.__init__(self, nbArms)

    def startGame(self):
        UCB.startGame(self)
        Budgeted.startGame(self)
        BernoulliEstimator.startGame(self)

    def getReward(self, arm, reward):
        UCB.getReward(self, arm, reward)
        Budgeted.getReward(self, reward)
        BernoulliEstimator.getReward(self, arm, reward)
            
    def computeIndex(self, arm):
        
        if self.pulls[arm] < 1:
        
            return float('+inf')
        
        else:
        
            #PositiveHoeffdingGambler
            #v = 1/2 * np.exp(-(2.0 * self.pulls[arm]**2 * self.estmeans[arm]**2) / (self.pulls[arm] * self.amplitude**2)) 
            v = 1/2 * np.exp(-(2.0 * self.pulls[arm]**2 * self.estmeans[arm]**2) / self.pulls[arm]) 
            if (self.estmeans[arm] > 0) :
                v = 1-v
            
            #v = beta.cdf(0.5, self.pulls[arm]-self.successes[arm]+1, self.successes[arm]+1)  #PositiveBernoulliGambler
            
            #u = sqrt((2 * max(1, log(self.t))) / self.pulls[arm])   #UCB1 
            u = sqrt((2 * log(self.budget)) / self.pulls[arm]) if (self.budget >= 1) else 0
            
            return  v + u 

    def computeAllIndex(self):
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)
            
            
            
    
class GamblerBernoulliUCB(UCB, Budgeted, BernoulliEstimator):
    
    def __init__(self, nbArms, inibudget=10.0, min_r=-1.0, max_r=+1.0, lower=0.0, amplitude=1.0):
        UCB.__init__(self, nbArms, lower=lower, amplitude=amplitude)
        Budgeted.__init__(self, inibudget=inibudget, min_r=min_r, max_r=max_r)
        BernoulliEstimator.__init__(self, nbArms)

    def startGame(self):
        UCB.startGame(self)
        Budgeted.startGame(self)
        BernoulliEstimator.startGame(self)

    def getReward(self, arm, reward):
        UCB.getReward(self, arm, reward)
        Budgeted.getReward(self, reward)
        BernoulliEstimator.getReward(self, arm, reward)
            
    def computeIndex(self, arm):
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            #v = self.estmeans[arm]
            #v = self.rewards[arm] / self.pulls[arm]
            v = beta.cdf(0.5, self.pulls[arm]-self.successes[arm]+1, self.successes[arm]+1)
            #u = sqrt((2 * max(1, log(self.t))) / self.pulls[arm])
            u = sqrt((2 * log(self.budget)) / self.pulls[arm]) if (self.budget >= 1) else 0
            return  v + u 

    def computeAllIndex(self):
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)
            
            
# class for the marab algorithm
class MaRaB(UCB):
    
    def __str__(self):
        return f"MaRaB(${self.alpha}$)"
    
    def __init__(self, nbArms, alpha=0.05, C=1e-6, lower=0., amplitude=1.):
        super(MaRaB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.alpha = alpha
        self.C = C
        self.reward_samples = [np.array([0.0]) for a in range(nbArms)]
        
    def startGame(self):
        UCB.startGame(self)
        self.reward_samples = [np.array([0.0]) for a in range(self.nbArms)]
                                       
    def getReward(self, arm, reward):
        UCB.getReward(self, arm, reward)
        self.reward_samples[arm] = np.sort(np.append(self.reward_samples[arm], [reward]))
        
    def computeIndex(self, arm):
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            # calculating empirical cvar
            e = np.ceil(self.alpha*self.pulls[arm]).astype(int)
            empirical_cvar = self.reward_samples[arm][:e].mean()
            # calculating lower confidence bound
            lcb = np.sqrt(np.log(np.ceil(self.alpha*self.t))/self.pulls[arm])
            # adding score to scores list
            return  empirical_cvar - self.C * lcb

    def computeAllIndex(self):
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)
        
        
#inspired by MARAB, we note the empirical frequency and mean of both positive and negative rewards
class EmpSurv(UCB):
    
    def __str__(self):
        return f"EmpSurv"
    
    def __init__(self, nbArms, C=1e-6, lower=0., amplitude=1.):
        super(MaRaB, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.C = C
        self.positive_mean = nb.repeat(0.0, self.nbArms, dtype='float')
        self.negative_mean = nb.repeat(0.0, self.nbArms, dtype='float')
        self.positive_count = nb.repeat(0, self.nbArms, dtype='int')
        self.negative_count = nb.repeat(0, self.nbArms, dtype='int')

    def startGame(self):
        UCB.startGame(self)
        self.positive_mean = nb.repeat(0.0, self.nbArms, dtype='float')
        self.negative_mean = nb.repeat(0.0, self.nbArms, dtype='float')
        self.positive_count = nb.repeat(0, self.nbArms, dtype='int')
        self.negative_count = nb.repeat(0, self.nbArms, dtype='int')
                                       
    def getReward(self, arm, reward):
        UCB.getReward(self, arm, reward)
        if (reward >= 0):
            self.positive_count[arm]+=1
            self.positive_mean[arm] = (self.positive_mean[arm] * (self.positive_count[arm]-1) + reward) / self.positive_count[arm]
        else:
            self.negative_count[arm]+=1
            self.negative_mean[arm] = (self.negative_mean[arm] * (self.negative_count[arm]-1) + reward) / self.negative_count[arm]
        
    def computeIndex(self, arm):
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            v = self.positive_count[arm] / self.pulls[arm]  #positive frequency (= estimated probability)
            #u = sqrt((2 * max(1, log(self.t))) / self.pulls[arm])
            #u = sqrt((2 * log(self.budget)) / self.pulls[arm]) if (self.budget >= 1) else 0
            return  v # + u 

    def computeAllIndex(self):
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)

#####################################################

    
class BanditGambler(IndexPolicy, Budgeted, BernoulliEstimator):

    def __init__(self, nbArms, inibudget=10.0, min_r=-1.0, max_r=+1.0, lower=0.0, amplitude=1.0):
        IndexPolicy.__init__(self, nbArms, lower=lower, amplitude=amplitude)
        Budgeted.__init__(self, inibudget=inibudget, min_r=min_r, max_r=max_r)
        BernoulliEstimator.__init__(self, nbArms)

    #@jit
    def ruin_estimated_prob(self, arm):
        #x = self.successes[arm]
        #y = self.pulls[arm]-self.successes[arm]
        #b = max(1.0, self.budget)
        return beta.cdf(0.5, self.successes[arm]+1, self.pulls[arm]-self.successes[arm]+1) + integral(lambda p, x, y, b : ((1-p)/p)**b * beta.pdf(p, x+1, y+1), 0.5, 1.0, (self.successes[arm], self.pulls[arm]-self.successes[arm], max(1.0, self.budget)))[0]

    def startGame(self):
        IndexPolicy.startGame(self)
        Budgeted.startGame(self)
        BernoulliEstimator.startGame(self)

    def getReward(self, arm, reward):
        IndexPolicy.getReward(self, arm, reward)
        Budgeted.getReward(self, reward)
        BernoulliEstimator.getReward(self, arm, reward)

    def computeIndex(self, arm):
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return 1.0 - self.ruin_estimated_prob(arm)

    #def computeAllIndex(self):
    #    for arm in range(self.nbArms):
    #        self.index[arm] = self.computeIndex(arm)
    #        
    #def choice(self):
    #    self.computeAllIndex()
    #    return np.random.choice(np.flatnonzero(self.index == np.max(self.index)))
        

#Dependencies
from typing import TypeVar, Generic
import numpy as np
import numpy.ma as ma
from numpy.random import binomial, randint, uniform, choice, rand
from math import sqrt, log
from scipy.stats import beta
#from scipy.integrate import quad as integral
#from scipy.integrate import fixed_quad as integral
from scipy.integrate import quadrature as integral
import pandas as pd
from numba import jit
from tqdm import tqdm_notebook as tqdm
from collections import Iterable
#from IPython.display import display
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
import datetime
#%matplotlib inline
#%matplotlib notebook
#import pickle
#from google.colab import files

type = TypeVar('T')

""" partially copied from SMPyBandits"""

class Domain():

    def __str__(self):
        return f"Domain ($r_min={self.r_min}, r_max={self.r_max}$)"

    def __init__(self, r_min=0.0, r_max=1.0):
        """ class for reward domain. 
            Arms always return values into the interval [0, 1].
            For budgeted problems, the domain is used for redimensioning r
        """
        #assert r_max >= r_min, "Error, the maximal reward must be greater than the minimal."  # DEBUG
        if r_max < r_min:
            print("SMAB warning: the maximal reward must be greater than the minimal; they were swaped.")
            r_max, rmin = r_min, r_max
        self.r_min = r_min  #: Lower values for rewards
        self.r_max = r_max  #: Higher values for rewards
        self.r_amp = r_max - r_min  #: Larger values for rewards
        self.r_0_1 = ((self.r_max==1.0) and (self.r_min==0.0))

################################################################################

class RandomArm():
    """ Base class for an arm class.
        return uniformly distributed random values between 0 and 1
    """
    def __str__(self):
        return f"Random Arm"

    def __init__(self):
        """ Base class for an arm class."""
        self.mean = 0.5

    def draw(self, shape=None):
        """ Draw a numpy array of random samples, of a certain shape. If shape is None, return a single sample"""
        return uniform(low=0.0, high=1.0, size=shape)

################################################################################

class BernoulliArm(RandomArm):
    """ Bernoulli distributed arm."""

    def __str__(self):
        return f"Bernoulli Arm ($p={self.p}$)"
    
    def __init__(self, p):
        """New arm."""
        super().__init__()
        #assert 0.0 <= p <= 1.0, "Error, the parameter probability for Bernoulli class has to be in [0, 1]."  # DEBUG
        if p > 1.0:
            print("SMAB warning: parameter p cannot be greater than 1.0; fixing it to 1.0")
            p = 1.0
        if p < 0.0:
            print("SMAB warning: parameter p cannot be negative; fixing it to 0.0")
            p = 0.0
        self.p = p  #: Parameter p for this Bernoulli arm
        self.mean = p

    # --- Random samples
    def draw(self, shape=None):
        """ Draw a numpy array of random samples, of a certain shape. If shape is None, return a single sample"""
        return binomial(1, self.p, size=shape)

################################################################################

class BasePolicy():
    """ Base class for any policy. """

    def __str__(self):
        return f"Base Policy ($k={self.k}, w={self.w}$)"
    
    def __init__(self, k, w=1):
        """ New policy."""
        # Parameters
        #assert k > 0, "Error, the number of arms must be a positive integer."  # DEBUG
        if k < 1:
            print("SMAB warning: parameter k must be a positive integer; fixing it to 2")
            k = 2
        if k < 0:
            print("SMAB warning: parameter w cannot be negative; fixing it to 0")
            w = 0
        self.k = int(k)  #: Number of Arms
        self.w = int(w)  #: if w>0, each arm must be played at least w times on the beginning (initial trials)
        # Internal state
        self.t = 0  #: Internal time-step
        self.n_i = np.zeros(self.k, dtype=int)  #: Number of pulls of each arm
        self.i_last = 0   #last pulled arm

    def reset(self):
        """ Start the game (fill pulls with 0)."""
        self.t = 0
        self.n_i.fill(0)
        self.i_last = 0

    def choose(self):
        if ( (self.w > 0) and (self.t < (self.k * self.w)) ):
          # play each arm w times, in order
          self.i_last = self.t % self.k
        else:
          # otherwise: undefined
          self.i_last = None
        return self.i_last

    def observe(self, r):
        """ Receive reward, increase t, pulls, and update."""
        #update internal state
        self._update(r)
        #evaluate
        self._evaluate()

    def _update(self, r):
        self.t += 1
        self.n_i[self.i_last] += 1

    def _evaluate(self):
        """ update utility after last observation """
        pass

################################################################################

class RandomPolicy(BasePolicy):
    """ Choose an arm uniformly at random. """

    def __str__(self):
        return f"Random Policy ($k={self.k}, w={self.w}$)"
    
    def choose(self):
        # base choice: verify mandatory initial rounds
        super().choose()
        # otherwise: random choice
        if self.i_last is None:
            # uniform choice among the arms
            self.i_last = randint(self.k)
        return self.i_last
        
    
################################################################################

class FixedPolicy(BasePolicy):
    """ Choose always the same arm. """

    def __str__(self):
        return f"Fixed Policy ($k={self.k}, w={self.w}, i={self.fixed_i}$)"
    
    def __init__(self, k, w=1, fixed_i=None):
        """ New fixed policy."""
        # Parameters
        super().__init__(k, w)
        if (fixed_i is None):
            #choose the fixed policy at random
            self.fixed_i = randint(self.k)
        else:
            #the fixed policy is given
            self.fixed_i = fixed_i
            
    def choose(self):
        # base choice: verify mandatory initial rounds
        super().choose()
        # otherwise: random choice
        if self.i_last is None:
            # fixed choice
            self.i_last = self.fixed_i
        return self.i_last
        
    
################################################################################

class IndexPolicy(BasePolicy):
    """ Class that implements a generic index policy.
        by default, implements the empirical means method
        The naive Empirical Means policy for bounded bandits: like UCB but without a bias correction term. 
        Note that it is equal to UCBalpha with alpha=0, only quicker.
    """

    def __str__(self):
        return f"Empirical Means ($k={self.k}, w={self.w}$)"
    
    def __init__(self, k, v_ini=None, w=1):
        """ New generic index policy. """
        super().__init__(k, w)
        self.s_i = np.full(k, 0.0)  #: cumulated rewards for each arm
        self.v_ini = v_ini  if  (v_ini is not None)  else  0.0   #: initial value (index or utility) for the arms
        self.v_i = np.full(k, v_ini)  #: value (index or utility) for each arm
        self.bests = np.arange(k)   #list of best arms (with equivalent highest utility), candidates

    def reset(self):
        """ Initialize the policy for a new game."""
        super().reset()
        self.s_i.fill(0.0)
        self.v_i.fill(self.v_ini)
        self.bests = np.arange(self.k)

    def choose(self):
        r""" choose an arm with maximal index (uniformly at random):
        .. math:: A(t) \sim U(\arg\max_{1 \leq k \leq K} I_k(t)).
        .. note:: In almost all cases, there is a unique arm with maximal index, so we loose a lot of time with this generic code, but I couldn't find a way to be more efficient without loosing generality.
        """
        # base choice: verify mandatory initial rounds
        super().choose()
        # otherwise: index choice
        if self.i_last is None:
          # Uniform choice among the best arms
          self.i_last = choice(self.bests)
        return self.i_last

    def observe(self, r):
        """ Receive reward, increase t, pulls, and update."""
        super().observe(r)   # update() and evaluate()
        #define bests
        self.bests = self._calc_bests()

    def _update(self, r):
        """ update estimated means after last observation """
        super()._update(r)
        self.s_i[self.i_last] += r

    def _evaluate(self):
        """ update utility after last observation 
            in this case, the utility is the estimated mean
        """
        i = self.i_last
        n_i = self.n_i[i]
        s_i = self.s_i[i]
        self.v_i[i] = s_i / n_i    # value corresponds to the empirical mean
        #self.v_i[i] = (v * ((n-1) / n)) + (r / n)

    def _calc_bests(self):
        """ define best arms - all with equivalent highest utility - then candidates """
        return np.flatnonzero(self.v_i == np.max(self.v_i))
    
################################################################################

EmpiricalMeansPolicy = IndexPolicy

################################################################################

class EpsilonGreedyPolicy(EmpiricalMeansPolicy, RandomPolicy):
    r""" The epsilon-greedy random policy.
    - At every time step, a fully uniform random exploration has probability :math:`\varepsilon(t)` to happen, otherwise an exploitation is done.
    """

    def __str__(self):
        return f"Epsilon-Greedy ($k={self.k}, w={self.w}, eps={self.eps}$)"
    
    def __init__(self, k, v_ini=None, w=1, eps=0.9):
        EmpiricalMeansPolicy.__init__(self, k, v_ini=v_ini, w=w)
        #assert 0 <= eps <= 1, "Error: the 'epsilon' parameter for EpsilonGreedy class has to be in [0, 1]."  # DEBUG
        if eps > 1.0:
            print("SMAB warning: parameter epsilon cannot be greater than 1.0; fixing it to 1.0")
            eps = 1.0
        if eps < 0.0:
            print("SMAB warning: parameter epsilon cannot be negative; fixing it to 0.0")
            eps = 0.0
        self.eps = eps

    #alternative: randomize instant utilities
    #def _calc_bests(self):
    #    # Generate random number
    #    p = rand()
    #    """With a probability of epsilon, explore (uniform choice), otherwise exploit based on empirical mean rewards."""
    #    if p < self.eps: # Proba epsilon : explore
    #        return np.array([randint(self.k)])
    #    else:  # Proba 1 - epsilon : exploit
    #        return super()._calc_bests()

    def choose(self):
        """With a probability of epsilon, explore (uniform choice), otherwise exploit based on empirical mean rewards."""
        # base choice: verify mandatory initial rounds
        BasePolicy.choose(self)
        # otherwise:
        if self.i_last is None:
          # Generate random number
          rnd_t = rand()
          # Proba epsilon : explore
          if rnd_t < self.eps: 
            RandomPolicy.choose(self)
          # Proba 1 - epsilon : exploit
          else:
            EmpiricalMeansPolicy.choose(self)
        return self.i_last

################################################################################
        
class SoftMaxPolicy(EmpiricalMeansPolicy):
    r"""The Boltzmann Exploration (Softmax) index policy, with a constant temperature :math:`\eta_t`.
    - Reference: [Algorithms for the multi-armed bandit problem, V.Kuleshov & D.Precup, JMLR, 2008, §2.1](http://www.cs.mcgill.ca/~vkules/bandits.pdf) and [Boltzmann Exploration Done Right, N.Cesa-Bianchi & C.Gentile & G.Lugosi & G.Neu, arXiv 2017](https://arxiv.org/pdf/1705.10257.pdf).
    - Very similar to Exp3 but uses a Boltzmann distribution.
      Reference: [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems, S.Bubeck & N.Cesa-Bianchi, §3.1](http://sbubeck.com/SurveyBCB12.pdf)
    """

    def __str__(self):
        return f"SoftMax ($k={self.k}, w={self.w}, eta={self.eta}$)"
    
    def __init__(self, k, v_ini=None, w=1, eta=None):
        super().__init__(k, v_ini=v_ini, w=w)
        #assert eta > 0, "Error: the temperature parameter for Softmax class has to be > 0."
        if (eta is not None) and (eta <= 0.0):
            print("SMAB warning: the temperature parameter for Softmax has to be positive; setting it to default.")
            eta = None
        if eta is None:  # Use a default value for the temperature
            eta = np.sqrt(np.log(k) / k)
        self.eta = eta

    def _evaluate(self):
        r"""Update the trusts probabilities according to the Softmax (ie Boltzmann) distribution on accumulated rewards, and with the temperature :math:`\eta_t`.
        .. math::
           \mathrm{trusts}'_k(t+1) &= \exp\left( \frac{X_k(t)}{\eta_t N_k(t)} \right) \\
           \mathrm{trusts}(t+1) &= \mathrm{trusts}'(t+1) / \sum_{k=1}^{K} \mathrm{trusts}'_k(t+1).
        If :math:`X_k(t) = \sum_{\sigma=1}^{t} 1(A(\sigma) = k) r_k(\sigma)` is the sum of rewards from arm k.
        """
        i = self.i_last
        n_i = self.n_i[i]
        s_i = self.s_i[i]
        eta = self.eta
        self.v_i[i] = np.exp(s_i / (eta * n_i))

    def choose(self):
        """random selection with softmax probabilities, thank to :func:`numpy.random.choice`."""
        # base choice: verify mandatory initial rounds
        BasePolicy.choose(self)
        # otherwise:
        if self.i_last is None:
          # pondered choice among the arms based on their normalize v_i
          s = np.sum(self.v_i)
          if s > 0:
            self.i_last = choice(self.k, p=(np.array(self.v_i/s,dtype='float64')))
          else:
            self.i_last = randint(self.k)
        return self.i_last

################################################################################

class UCB1Policy(IndexPolicy):

    def __str__(self):
        return f"UCB1 ($k={self.k}, w={self.w}$)"
                  
    def _evaluate(self):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:
        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{2 \log(t)}{N_k(t)}}.
        """
        #calculate utility following UCB formula
        i = self.i_last
        n_i = self.n_i[i]
        mu_i = self.s_i[i] / n_i
        t = self.t
        if self.n_i[i] == 0:
            self.v_i[i] = float('+inf')
        else:
            self.v_i[i] = mu_i + sqrt((2 * log(t)) / n_i)


################################################################################

class BernKLUCBPolicy(IndexPolicy):

    def __str__(self):
        return f"Bernoulli KL-UCB ($k={self.k}, w={self.w}$)"
                  
    @jit
    def _klBern(self, x, y):
        r""" Kullback-Leibler divergence for Bernoulli distributions.
        .. math:: \mathrm{KL}(\mathcal{B}(x), \mathcal{B}(y)) = x \log(\frac{x}{y}) + (1-x) \log(\frac{1-x}{1-y}).
        """
        eps = 1e-15  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]
        x = min(max(x, eps), 1 - eps)
        y = min(max(y, eps), 1 - eps)
        return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))

    @jit
    def _klucbBern(self, x, d, precision=1e-6):
        """ KL-UCB index computation for Bernoulli distributions, using :func:`klucb`."""
        upperbound = min(1., self._klucbGauss(x, d, sig2x=0.25))  # variance 1/4 for [0,1] bounded distributions
        return self._klucb(x, d, upperbound, precision)

    @jit
    def _klucbGauss(self, x, d, sig2x=0.25):
        """ KL-UCB index computation for Gaussian distributions.
        - Note that it does not require any search.
        .. warning:: it works only if the good variance constant is given.
        .. warning:: Using :class:`Policies.klUCB` (and variants) with :func:`klucbGauss` is equivalent to use :class:`Policies.UCB`, so prefer the simpler version.
        """
        return x + sqrt(abs(2 * sig2x * d))

    @jit
    def _klucb(self, x, d, upperbound, precision=1e-6, lowerbound=float('-inf'), max_iterations=50):
        r""" The generic KL-UCB index computation.
        - ``x``: value of the cum reward,
        - ``d``: upper bound on the divergence,
        - ``kl``: the KL divergence to be used (:func:`klBern`, :func:`klGauss`, etc),
        - ``upperbound``, ``lowerbound=float('-inf')``: the known bound of the values ``x``,
        - ``precision=1e-6``: the threshold from where to stop the research,
        - ``max_iterations=50``: max number of iterations of the loop (safer to bound it to reduce time complexity).
        .. math::
            \mathrm{klucb}(x, d) \simeq \sup_{\mathrm{lowerbound} \leq y \leq \mathrm{upperbound}} \{ y : \mathrm{kl}(x, y) < d \}.
        .. note:: It uses a **bisection search**, and one call to ``kl`` for each step of the bisection search.
        For example, for :func:`klucbBern`, the two steps are to first compute an upperbound (as precise as possible) and the compute the kl-UCB index:
        >>> x, d = 0.9, 0.2   # mean x, exploration term d
        >>> upperbound = min(1., klucbGauss(x, d, sig2x=0.25))  # variance 1/4 for [0,1] bounded distributions
        """
        v = max(x, lowerbound)
        u = upperbound
        i = 0
        while ((i < max_iterations) and (u - v > precision)):
            i += 1
            m = (v + u) * 0.5
            if self._klBern(x, m) > d:
                u = m
            else:
                v = m
        return (v + u) * 0.5

    def _evaluate(self):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:
        .. math::
            \hat{\mu}_k(t) &= \frac{X_k(t)}{N_k(t)}, \\
            U_k(t) &= \sup\limits_{q \in [a, b]} \left\{ q : \mathrm{kl}(\hat{\mu}_k(t), q) \leq \frac{c \log(t)}{N_k(t)} \right\},\\
            I_k(t) &= U_k(t).
        If rewards are in :math:`[a, b]` (default to :math:`[0, 1]`) and :math:`\mathrm{kl}(x, y)` is the Kullback-Leibler divergence between two distributions of means x and y (see :mod:`Arms.kullback`),
        and c is the parameter (default to 1).
        """
        c = 1.0
        #tolerance = 1e-4
        i = self.i_last
        n_i = self.n_i[i]
        mu_i = self.s_i[i] / n_i
        t = self.t
        if n_i == 0:
            self.v_i[i] = float('+inf')
        else:
            self.v_i[i] = self._klucbBern(mu_i, c * log(t) / n_i)

################################################################################

class ThompsonPolicy(IndexPolicy):
    r"""The Thompson (Bayesian) index policy.
    - By default, it uses a Beta posterior (:class:`Policies.Posterior.Beta`), one by arm.
    - Prior is initially flat, i.e., :math:`a=\alpha_0=1` and :math:`b=\beta_0=1`.
    - Reference: [Thompson - Biometrika, 1933].
    """

    def __str__(self):
        return f"Thompson (Beta) Sampling ($k={self.k}, w={self.w}$)"
                  
    def _evaluate(self):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, giving :math:`S_k(t)` rewards of 1, by sampling from the Beta posterior:
        .. math::
            A(t) &\sim U(\arg\max_{1 \leq k \leq K} I_k(t)),\\
            I_k(t) &\sim \mathrm{Beta}(1 + \tilde{S_k}(t), 1 + \tilde{N_k}(t) - \tilde{S_k}(t)).
        """
        for i in range(self.k):
          a = self.s_i[i] + 1
          b = self.n_i[i] - self.s_i[i] + 1
          self.v_i[i] = beta.rvs(a, b)


################################################################################

class BayesUCBPolicy(IndexPolicy):
    """ The Bayes-UCB policy.
    - By default, it uses a Beta posterior (:class:`Policies.Posterior.Beta`), one by arm.
    -Reference: [Kaufmann, Cappé & Garivier - AISTATS, 2012].
    """

    def __str__(self):
        return f"Bayes (Beta) UCB ($k={self.k}, w={self.w}$)"
                  
    def _evaluate(self):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, giving :math:`S_k(t)` rewards of 1, by taking the :math:`1 - \frac{1}{t}` quantile from the Beta posterior:
        .. math:: I_k(t) = \mathrm{Quantile}\left(\mathrm{Beta}(1 + S_k(t), 1 + N_k(t) - S_k(t)), 1 - \frac{1}{t}\right).
        """
        i = self.i_last
        #q = 1. - (1. / (1 + self.n_i[i]))
        q = 1. - (1. / (1 + self.t))
        a = self.s_i[i] + 1
        b = self.n_i[i] - self.s_i[i] + 1
        self.v_i[i] = beta.ppf(q, a, b)

################################################################################

# class for the marab algorithm
class MaRaBPolicy(IndexPolicy):

    def __str__(self):
        return f"Empirical MaRaB ($k={self.k}, w={self.w}, alpha={self.alpha} $)"
                  
    def __init__(self, k, v_ini=None, w=1, alpha=0.05, c=1e-6):
        super().__init__(k, v_ini=v_ini, w=w)
        self.alpha = alpha
        self.c = c
        self.reward_samples = [np.array([0.0]) for a in range(k)]
        
    def reset(self):
        super().reset()
        self.reward_samples = [np.array([0.0]) for a in range(self.k)]
                                       
    def _update(self, r):
        super()._update(r)
        self.reward_samples[self.i_last] = np.sort(np.append(self.reward_samples[self.i_last], [r]))
        
    def _evaluate(self):
        i = self.i_last
        # calculating empirical cvar
        e = np.ceil(self.alpha*self.n_i[i]).astype(int)
        empirical_cvar = self.reward_samples[i][:e].mean()
        # calculating lower confidence bound
        lcb = np.sqrt(np.log(np.ceil(self.alpha*self.t))/self.n_i[i])
        # adding score to scores list
        self.v_i[i] = empirical_cvar - self.c * lcb

################################################################################

class Budgeted:

    def __str__(self):
        return f"Budgeted ($k={self.k}, b_0={self.b_0}$)"
                  
    def __init__(self, k, d=None, b_0=None):
        if b_0 is None:
            b_0 = k
        self.b_0 = b_0
        self.d = d  if  (isinstance(d, Domain))  else  Domain()
        self.b = b_0   #budget
        self.s = 0.0   #total cumulated rewards
        
    def reset(self):
        self.b = self.b_0
        self.s = 0.0

    def _update(self, r):
        self.s += r
        self.b += r * self.d.r_amp + self.d.r_min

################################################################################

class Estimator:

    def __str__(self):
        return f"Average Reward Estimator ($k={self.k}$)"
                  
    def __init__(self, k):
        self.avg_i = np.zeros(k)

    def reset(self):
        self.avg_i.fill(0.0)

    def _update(self, r):
        self.avg_i[self.i_last] = (self.avg_i[self.i_last] * (self.n_i[self.i_last]-1) + r) / self.n_i[self.i_last]


class BernoulliEstimator(Estimator):

    def __str__(self):
        return f"Bernoulli Estimator ($k={self.k}$)"

    def __init__(self, k):
        Estimator.__init__(self, k)
        self.x_i = np.zeros(k, dtype='int')   #number of successes

    def reset(self):
        Estimator.startGame(self)
        self.x_i.fill(0)

    def _update(self, r):
        Estimator._update(self, r)
        if (r > 0):
            self.x_i[self.i_last] += 1



#####################################################


class AlarmedPolicy(Budgeted, Estimator):

    def __str__(self):
        return f"Alarmed ($k={self.k}, b_0={self.b_0}, omega={self.omega}$)"
                  
    def __init__(self, k, d=None, b_0=None, omega=1.0):
        Budgeted.__init__(self, k, d=d, b_0=b_0)
        Estimator.__init__(self, k)
        self.omega = omega   #safety-critical warning threshold for budget level

    def reset(self):
        Budgeted.reset(self)
        Estimator.reset(self)

    def _update(self, r):
        Budgeted._update(self, r)
        Estimator._update(self, r)

    def choose(self):
        #sufficient budget
        if self.b > self.omega:
            return None
        #low budget
        else:
            if np.max(self.avg_i) > 0:
                # greedy:
                #  = uniform choice among the best arms
                self.i_last = np.random.choice(np.flatnonzero(self.avg_i == np.max(self.avg_i)))
            else:
                # otherwise:
                #  = continue using ancestor policy
                self.i_last = None
        return self.i_last

#####################################################


class AlarmedUCBPolicy(UCB1Policy, AlarmedPolicy):

    def __str__(self):
        return f"Alarmed-UCB($omega={self.omega}$)"

    def __init__(self, k, v_ini=None, w=1, d=None, b_0=None, omega=1.0):
        UCB1Policy.__init__(self, k, v_ini=v_ini, w=w)
        AlarmedPolicy.__init__(self, k, d=d, b_0=b_0)

    def reset(self):
        UCB1Policy.reset(self)
        AlarmedPolicy.reset(self)

    def _update(self, r):
        UCB1Policy._update(self, r)
        AlarmedPolicy._update(self, r)

    def choose(self):
        if ( (self.w > 0) and (self.t < (self.k * self.w)) ):
          # play each arm w times, in order
          self.i_last = self.t % self.k
        else:
          AlarmedPolicy.choose(self)
          if self.i_last is None:
            UCB1Policy.choose(self)
        return self.i_last


class AlarmedBernKLUCBPolicy(BernKLUCBPolicy, AlarmedPolicy):

    def __str__(self):
        return f"Safe-KL-UCB($omega={self.omega}$)"

    def __init__(self, k, v_ini=None, w=1, d=None, b_0=None, omega=1.0):
        BernKLUCBPolicy.__init__(self, k, v_ini=v_ini, w=w)
        AlarmedPolicy.__init__(self, k, d=d, b_0=b_0)

    def reset(self):
        BernKLUCBPolicy.reset(self)
        AlarmedPolicy.reset(self)

    def _update(self, r):
        BernKLUCBPolicy._update(self, r)
        AlarmedPolicy._update(self, r)

    def choose(self):
        if ( (self.w > 0) and (self.t < (self.k * self.w)) ):
          # play each arm w times, in order
          self.i_last = self.t % self.k
        else:
          AlarmedPolicy.choose(self)
          if self.i_last is None:
            BernKLUCBPolicy.choose(self)
        return self.i_last


class AlarmedEpsilonGreedyPolicy(EpsilonGreedyPolicy, AlarmedPolicy):

    def __str__(self):
        return f"Safe-$\epsilon$-greedy($\epsilon={self._epsilon}, \omega={self.omega}$)"

    def __init__(self, k, v_ini=None, w=1, d=None, b_0=None, omega=1.0, eps=0.9):
        EpsilonGreedyPolicy.__init__(self, k, v_ini=v_ini, w=w, eps=eps)
        AlarmedPolicy.__init__(self, k, d=d, b_0=b_0)

    def reset(self):
        EpsilonGreedyPolicy.reset(self)
        AlarmedPolicy.reset(self)

    def _update(self, r):
        EpsilonGreedyPolicy._update(self, r)
        AlarmedPolicy._update(self, r)

    def choose(self):
        if ( (self.w > 0) and (self.t < (self.k * self.w)) ):
          # play each arm w times, in order
          self.i_last = self.t % self.k
        else:
          AlarmedPolicy.choose(self)
          if self.i_last is None:
            EpsilonGreedyPolicy.choose(self)
        return self.i_last

    
#####################################################


class BanditGamblerPolicy(IndexPolicy, Budgeted):

    def __init__(self, k, v_ini=None, w=1, d=None, b_0=None):
        #super().__init__(k, v_ini=v_ini, w=w, d=d, b_0=b_0)
        IndexPolicy.__init__(self, k, v_ini=v_ini, w=w)
        Budgeted.__init__(self, k, d=d, b_0=b_0)

    #@jit
    def ruin_estimated_prob(self, i):
        n_i = self.n_i[i]
        x_i = self.s_i[i]
        y_i = n_i - self.s_i[i]
        b = max(1.0, self.b)
        return beta.cdf(0.5, x_i+1, y_i+1) + integral(lambda p, x, y, b : ((1-p)/p)**b * beta.pdf(p, x+1, y+1), 0.5, 1.0, (x_i, y_i, b))[0]

    def reset(self):
        #super().reset()
        IndexPolicy.reset(self)
        Budgeted.reset(self)

    def _update(self, r):
        #super()._update(r)
        IndexPolicy._update(self, r)
        Budgeted._update(self, r)

    def _evaluate(self):
        i = self.i_last
        self.v_i[i] = 1.0 - self.ruin_estimated_prob(i)

################################################################################

class BanditGamblerUCBPolicy(BanditGamblerPolicy):

    def ruin_estimated_prob(self, i):
        n_i = self.n_i[i]
        x_i = self.s_i[i]
        y_i = n_i - self.s_i[i]
        b = max(1.0, self.b)
        factor = np.log(self.t)/self.t
        return beta.cdf(0.5, x_i+1, y_i+1) + integral(lambda p, x, y, b : ((1-p)/p)**b * beta.pdf(p, x*factor+1, y*factor+1), 0.5, 1.0, (x_i, y_i, b))[0]

################################################################################


class SMAB():
    """ Base survival MAB process. """

    def __init__(self, A, G, h, b_0, d=None, n=1, w=None, run=False, save_only_means=True):
        """
         A : List of Arms
         G : List of Algorithms
         h : max time-horizon
         d : rewards domain
         n : number of repetitions
         w : sliding window
         b_0 : initial budget
        """
        #domain of rewards ( by default on [0, 1] )
        self.d = d  if  isinstance(d, Domain)  else  Domain()

        #time-horizon (0, 1 ... t ... h)
        self.h = h   #time-horizon
        self.T = range(self.h)          #range for time (0 ... h-1)
        self.T1 = range(1, self.h+1)    #range for time (1 ... h)
        self.T01 = range(0, self.h+1)   #range for time (0, 1 ... h)

        #arms (1 ... i ... k)
        self.A = A if isinstance(A, Iterable) else [A]

        #number of arms
        self.k = len(self.A)
        self.K = range(self.k)          #range for arms (0 ... k-1)
        self.K1 = range(1,self.k+1)     #range for arms (1 ... k)

        #arms properties
        self.mu_a = np.array([a.mean for a in A])  #means
        self.mu_star = np.max(self.mu_a)           #best mean
        self.a_star = np.argmax(self.mu_a)         #best arm index
        self.mu_worst = np.min(self.mu_a)          #worst mean
        self.a_worst = np.argmin(self.mu_a)        #worst arm index

        #budget
        self.b_0 = b_0   
        
        #algorithms (1 ... g ... m)
        self.G = G if isinstance(G, Iterable) else [G]
        self.m = len(self.G)

        #repetitions (1 ... j ... n)
        self.n = n

        #window
        if (w is not None):
            self.w = max(2, min(w, horizon-1))
        else:
            self.w = w

        #if save all sim data
        self.save_only_means = save_only_means		

        #run
        if run:
            self.run()


    def run(self, tqdm_desc_it="iterations", tqdm_desc_alg="algorithms", tqdm_desc_rep="repetitions", tqdm_leave=False, tqdm_disable=False, prev_draw=True):

        #time-horizon (1 ... t ... h)
        #arms (1 ... i ... k)
        #repetitions (1 ... j ... n)
        #algorithms (1 ... g ... m)

        # Initialize Rewards and History of selected Actions (3d matrices [t x g x i])
        X = np.zeros((self.n, self.m, self.h), dtype=float)  #successes
        #R = np.zeros((self.n, self.m, self.h), dtype=float)  #rewards
        #SR = np.zeros((self.n, self.m, self.h), dtype=float)  #cumulated rewards
        H = np.full((self.n, self.m, self.h), -1, dtype=int) #history of actions
        #B = np.zeros((self.n, self.m, self.h), dtype=float)  #budget

        # Draw for every arm all repetitions
        if prev_draw:
            X_i_t_j = np.array([arm.draw((self.h, self.n)) for arm in self.A])	

        # For each repetition
        #for j in tqdm(range(self.n), desc=tqdm_desc_rep, leave=(tqdm_leave and self.m == 1), disable=(tqdm_disable or self.n == 1)):
        #for j in tqdm(range(self.n), desc=tqdm_desc_rep, leave=tqdm_leave, disable=(tqdm_disable or self.n == 1)):
        for j in tqdm(range(self.n)):

            # For each algorithm
            #for g, alg in enumerate(tqdm(self.G, desc=tqdm_desc_alg, leave=tqdm_leave, disable=(tqdm_disable or self.m == 1))):
            for g, alg in enumerate(self.G):

                # Initialize
                alg.reset()
                #s = 0.0

                # Loop on time
                #for t in tqdm(self.T, desc=tqdm_desc_it, leave=tqdm_leave, disable=(tqdm_disable or self.n > 1 or self.m > 1) ):
                for t in self.T:
                    # The algorithm chooses the arm to play
                    i = alg.choose()
                    # The arm played gives reward
                    if prev_draw:
                        x = X_i_t_j[i, t, j]
                    else:
                        x = self.A[i].draw()
                    # The reward is returned to the algorithm
                    alg.observe(x)
                    # Save both
                    H[j, g, t] = i
                    X[j, g, t] = x
                    #r = x * self.d.r_amp + self.d.r_min
                    #s += r
                    #R[j, g, t] = r
                    #SR[j, g, t] = s
                    #b = s + self.b_0
                    #B[j, g, t] = b
                    #if (b == 0):
                    #    break
        
        R = X * self.d.r_amp + self.d.r_min

        #actions history, with initial action index being 1, not 0
        H1 = H+1

        #actions map (bool 4d matrix)
        H_a = np.array([[[[True if (H[j,g,t]==i) else False for t in self.T] for i in self.K] for g in range(self.m)] for j in range(self.n)], dtype='bool')

        #progressive actions count (int 4d matrix [t x j x i x a])
        N_a = np.cumsum(H_a, axis=3)

        #averaged progressive actions count (float 3d matrix [t x j x a]) #averaged over repetitions
        self.MN_a = np.mean(N_a, axis=0)		

        #progressive actions frequency (float 4d matrix [t x j x i x a])
        F_a = N_a / self.T1

        #averaged progressive actions frequency (float 3d matrix [t x j x a]) #averaged over repetitions
        self.MF_a = np.mean(F_a, axis=0)

        if (self.w is not None):

            #window count (int 4d matrix [t x j x i x a])
            NW_a = np.concatenate((N_a[:,:,:,:self.w], N_a[:,:,:,self.w:] - N_a[:,:,:,:-self.w]), axis=3)

            #averaged window count (float 3d matrix [t x j x a]) #averaged over repetitions
            self.MNW_a = np.mean(NW_a, axis=0)		

            #window frequency (float 4d matrix [t x j x i x a])
            FW_a = np.concatenate((N_a[:,:,:,:self.w] / np.arange(1,self.w+1, dtype='float'), (N_a[:,:,:,self.w:] - N_a[:,:,:,:-self.w]) / float(self.w)), axis=3) 

            #averaged window frequency (float 3d matrix [t x j x a]) #averaged over repetitions
            self.MFW_a = np.mean(FW_a, axis=0)		

        #final arm pull count (int 3d matrix [j x i x a])
        #n_a = N_a[:,:,:,self.h-1]
        n_a = N_a[:,:,:,-1]

        #averaged final arm pull count (float 2d matrix [j x a]) #averaged over repetitions
        self.mn_a = np.mean(n_a, axis=0)

        #final arm pull frequency (float 3d matrix [j x i x a])
        f_a = F_a[:,:,:,-1]

        #averaged final arm pull frequency (float 2d matrix [j x a]) #averaged over repetitions
        self.mf_a = np.mean(f_a, axis=0)

        #progressive cumulative rewards (float 3d matrix [t x j x i])
        SR = np.cumsum(R, axis=2, dtype='float')

        #averaged progressive cumulative rewards (float 2d matrix [t x j]) #averaged over repetitions
        self.MSR = np.mean(SR, axis=0)

        #final rewards (float 2d matrix [j x i])
        sr = SR[:,:,-1]

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
        sl = SL[:,:,-1]

        #averaged final cumulated regret (float 1d matrix [j]) #averaged over repetitions
        self.msl = np.mean(sl, axis=0)
        #and standard deviation
        self.dsl = np.std(sl, axis=0)
        
        #rewards map (float 4d matrix [t x j x i x a])
        R_a = np.array([[[[R[j,g,t] if (H[j,g,t]==i) else 0.0 for t in self.T] for i in self.K] for g in range(self.m)] for j in range(self.n)], dtype='float')

        #averaged rewards map (float 3d matrix [t x j x a]) #averaged over repetitions
        self.MR_a = np.mean(R_a, axis=0)

        #progressive rewards map (int 4d matrix [t x j x i x a])
        SR_a = np.cumsum(R_a, axis=3)

        #averaged progressive rewards map (float 3d matrix [t x j x a]) #averaged over repetitions
        self.MSR_a = np.mean(SR_a, axis=0)

        #final rewards per action (float 3d matrix [j x i x a])
        sr_a = SR_a[:,:,:,-1]

        #averaged final rewards per action (float 2d matrix [j x a]) #averaged over repetitions
        self.msr_a = np.mean(sr_a, axis=0)

        #reward proportion per action (float 3d matrix [j x i x a])
        fr_a = sr_a / SR[:,:,-1,np.newaxis]

        #averaged proportion per action (float 2d matrix [j x a]) #averaged over repetitions
        self.mfr_a = np.mean(fr_a, axis=0)

        #progressive budget (float 3d matrix [t x j x i])
        # i.e. the progressive cumulative rewards plus initial budget
        B = SR + self.b_0

        ##progressive on negative counter of episodes (float 3d matrix [t x j])
        ## i.e. the number of episodes where, at each time t, alg j is running on negative budget
        #N = np.sum(B >= 0, axis=0)

        #averaged progressive budget (float 2d matrix [t x j]) #averaged over repetitions
        #self.MB = np.mean(B, axis=0)
        self.MB = self.MSR + self.b_0

        #final budget (float 2d matrix [j x i])
        b = B[:,:,-1]

        #averaged final budget (float 1d matrix [j]) #averaged over repetitions
        self.mb = np.mean(b, axis=0)

        #time map on non-positive budget (int 3d matrix [t x j x i])
        #TNB = np.array([[[1 if(v<=0) else 0 for v in B_ij] for B_ij in B_i] for B_i in B])
        TNB = (B <= 0).astype(int)
        
        #time dead map (int 3d matrix [t x j x i])
        TD = np.maximum.accumulate(TNB, axis=2)

        #progressive survival counter of episodes (float 3d matrix [t x j])
        self.SC = 1 - np.mean(TD, axis=0)
        #final survival counter
        self.sc = self.SC[:,-1]
        #final survival rate
        self.rsc = self.sc / self.n

        
        #progressive budget considering ruin (float 3d matrix [t x j x i])
        # i.e. the progressive cumulative rewards plus initial budget
        #masked_B = ma.masked_less_equal(B, 0.0)
        RB = ma.masked_less_equal(B, 0.0).filled(0.0)

        #averaged progressive budget considering ruin (float 2d matrix [t x j]) #averaged over repetitions
        self.MRB = np.mean(RB, axis=0)

        #final budget (float 2d matrix [j x i])
        rb = RB[:,:,-1]

        #averaged final budget (float 1d matrix [j]) #averaged over repetitions
        self.mrb = np.mean(rb, axis=0)
        
        ##time map of the averaged budget on negative (int 2d matrix [t x j])
        #self.TNMB = np.array([[1 if(v<0) else 0 for v in MB_j] for MB_j in self.MB])

        ##survival time (before ruin or end) (int 2d matrix [j x i])
        #Z = np.reshape(np.ones(self.n*self.m, dtype='int'), [self.n, self.m, 1]) #add 1 at the end		
        #TNBZ = np.block([TNB, Z])
        #self.TTNB = np.array([[np.nonzero(v_tj==1)[0][0] for v_tj in v_t] for v_t in TNBZ])		

        ##averaged survival time (before ruin or end) (int 1d matrix [j])
        #self.MTTNB = np.mean(self.TTNB, axis=0)
        ##and std dev
        #self.DTTNB = np.std(self.TTNB, axis=0)

        ##cumulated time progression on negative budget
        #STNB = np.cumsum(TNB, axis=2)
        #self.STNMB = np.cumsum(self.TNMB, axis=1) 
        ##self.MSTNB = np.mean(self.STNB, axis=0)
        #
        ##final cumulated time on negative budget
        #stnb = STNB[:,:,self.tau-1]
        #
        #self.stnmb = self.STNMB[:,self.tau-1]
        #
        ##averaged final cumulated time on negative budget
        #self.mstnb = np.mean(stnb, axis=0)
        ##and std dev
        #self.dstnb = np.std(stnb, axis=0)

        ##ruin episodes (int 1d matrix [j])
        #self.senb = np.count_nonzero(stnb, axis=0) 
        ##rate
        #self.renb = 1.0 - self.senb / self.n

        ##negative budget progression
        #NB = np.array([[[v if(v<0) else 0 for v in B_ij] for B_ij in B_i] for B_i in B])
        #
        ##average negative budget progression
        #self.NMB = np.array([[v if(v<0) else 0 for v in MB_j] for MB_j in self.MB])
        #
        ##cumulated negative budget progression
        #SNB = np.cumsum(NB, axis=2, dtype='float')
        #
        ##self.MSNB = np.mean(SNB, axis=0)
        #
        ##cumulated negative budget progression on average
        #self.SNMB = np.cumsum(self.NMB, axis=1, dtype='float') 
        #
        ##final cumulated negative budget
        #snb = SNB[:,:,self.tau-1]
        #
        #self.snmb = self.SNMB[:,self.tau-1]
        #
        ##final cumulated negative budget (float 1d matrix [j]) #averaged over repetitions
        #self.msnb = np.mean(snb, axis=0)
        ##and its std deviation
        #self.dsnb = np.std(snb, axis=0)

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

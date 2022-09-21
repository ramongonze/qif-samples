import sys
sys.path.append("..") # import util

import numpy as np
import itertools as it
from fractions import Fraction
from math import floor, ceil, factorial, log10
from scipy.special import binom
from multiprocessing import Pool
from libqif.core.secrets import Secrets
from libqif.core.channel import Channel
from libqif.core.hyper import Hyper
from libqif.core.gvulnerability import Gain
import matplotlib.pyplot as plt
from util import float_equal
from tqdm import tqdm
from termcolor import colored

class ModelBinaryTarget:
    
    def __init__(self, n:int, m:int, prior:str):
        """Consider a population of size n, a sample of size 1<=m<n.
        The adversary assumes a uniform distribution of all possible n+1
        frequencies of value 'a' in the population.

        Parameters
        ----------
        n : int 
            Population's size.
        
        m : int
            Sample's size

        prior : str
            Prior distribution on secrets. It can be:
            (i) 'in' for the adversary that knows the target is in the sample,
            (ii) 'out' for the adversary that knows the target is outside the sample,
            (iii) 'unk' for the adversary that doesn't know whether the target is in or outside the sample.
        """

        self.n = n
        self.m = m
        self.secrets = self._create_secrets(n,m,prior)
        self.gain = self._create_gain()
        self.channel = self._create_channel()
        self.hyper = Hyper(self.channel)

    def _create_secrets(self, n:int, m:int, prior:str):
        X = ["".join(x) for x in it.product("ab", repeat=n)]
        X = [(p,t) for p in X for t in np.arange(n)]
        prior_dist = self._create_prior_on_secrets(n, m, prior, X)
        return Secrets(X, prior_dist)        

    def _create_prior_on_secrets(self, n, m, prior, X):
        """Create the prior distribution on secrets.

        Parameters
        ----------
        n : int 
            Population's size.
        
        m : int
            Sample's size

        prior : str
            Prior distribution on secrets. It can be:
            (i) 'in' for the adversary that knows the target is in the sample,
            (ii) 'out' for the adversary that knows the target is outside the sample,
            (iii) 'unk' for the adversary that doesn't know whether the target is in or outside the sample.
        
        X : list
            List of secret labels.
        """
        prior_dist = np.zeros(n*2**n)
        if prior == 'in':
            for i in np.arange(prior_dist.size):
                if X[i][1] < m: # Target in the sample
                    prior_dist[i] = 1/(m*(n+1)* binom(n,X[i][0].count("a")))
        elif prior == 'out':
            for i in np.arange(prior_dist.size):
                if X[i][1] >= m: # Target outside the sample
                    prior_dist[i] = 1/((n-m)*(n+1)* binom(n,X[i][0].count("a")))
        elif prior == 'unk':
            for i in np.arange(prior_dist.size):
                prior_dist[i] = 1/(n*(n+1)* binom(n,X[i][0].count("a")))
        else:
            raise Exception("Unknown prior distribution")
        
        return prior_dist

    def _create_gain(self):
        """Adversary that has a single target from the population and she wants to infer the target's attribute value.
        She gains 1 if she guesses the target's value correctly or 0 otherwise.
        """
        W = ["a","b"]
        matrix = np.zeros((len(W), self.secrets.num_secrets))

        for i in np.arange(len(W)):
            for j in np.arange(self.secrets.num_secrets):
                t = self.secrets.labels[j][1]
                matrix[i][j] = 1 if self.secrets.labels[j][0][t] == W[i] else 0
        
        return Gain(self.secrets, W, matrix)

    def _create_channel(self):
        Y = list(range(self.m+1))
        matrix = np.zeros((self.secrets.num_secrets, len(Y)))

        for i in np.arange(self.secrets.num_secrets):
            # The entry will be 1 only in the histogram that has the same number of 1's as the fist m people of secret x
            matrix[i][self.secrets.labels[i][0][:self.m].count("a")] = 1
            
        return Channel(self.secrets, Y, matrix)

    def prior_vul(self):
        """Prior vulnerability 'ground truth'."""
        return self.gain.prior_vulnerability()

    def post_vul(self):
        """Posterior vulnerability 'ground truth'."""
        return self.gain.posterior_vulnerability(self.hyper)

    @staticmethod
    def post_vul_eq(p):
        """Close formula for posterior vulnerability of attribute inference attack."""
        n, m, prior = p # Population size, sample size, prior distribution
        if prior == 'in':
            return 3/4 + 1/(4*(floor(m/2)+ceil((m+1)/2)))
        elif prior == 'out':
            return 3/4 - 1/(4*(floor((m+1)/2)+ceil(m/2)+1))
        elif prior == 'unk':
            if m%2 == 0:
                return (3*n*m + 2*m + 2*n)/(4*n*(m+1))
            return (3*n*m + 2*m + 5*n + 2)/(4*n*(m+2))

def test_closed_privacy():
    """Check whether posterior vulnerability of closed formula 
    for privacy gain function matches the code."""    
    for n in tqdm(np.arange(1,15), desc="Test closed formula privacy"):
        for m in np.arange(1,n):
            for prior in ['in', 'out', 'unk']:
                post_code = ModelBinaryTarget(n, m, prior).post_vul()
                post_eq = ModelBinaryTarget.post_vul_eq((n,m,prior))
                if not float_equal(post_code, post_eq):
                    print(colored("[Failed]", "red") +\
                        " at prior = %s, n = %d, m = %d, CODE = %.3f, EQ = %.3f"%(prior,n,m,post_code,post_eq))
                    return
    print(colored("[Successful]", "green") + " - Closed formula matches the code")

def main():
    test_closed_privacy()
    
if __name__ == "__main__":
    main()
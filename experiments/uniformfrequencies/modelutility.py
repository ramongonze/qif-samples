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

class ModelBinaryUtility:
    
    def __init__(self, n:int, m:int):
        """Consider a population of size n, a sample of size 1<=m<=n.
        The adversary assumes a uniform distribution of all possible n+1
        frequencies of value 'a' in the population.

        Parameters
        ----------
        n : int 
            Population's size.
        
        m : int
            Sample's size
        """

        self.n = n
        self.m = m
        self.secrets = self._create_secrets(n,m)
        self.gain = self._create_gain()
        self.channel = self._create_channel()
        self.hyper = Hyper(self.channel)    

    def _create_secrets(self, n:int, m:int):
        X = ["".join(x) for x in it.product("ab", repeat=n)]
        prior_dist = self._create_prior_on_secrets(n, X)
        return Secrets(X, prior_dist)        

    def _create_prior_on_secrets(self, n, X):
        """Create the prior distribution on secrets.

        Parameters
        ----------
        n : int 
            Population's size.
        
        X : list
            List of secret labels.
        """
        prior_dist = np.zeros(len(X))
        for i in np.arange(prior_dist.size):
            prior_dist[i] = 1/((n+1) * binom(n, X[i].count('a')))        
        return prior_dist

    def _create_gain(self):
        """Adversary that wants to infer the frequency of value in in the population.
        Her guesses can be from 0/n to n/n.
        """
        W = [i/self.n for i in np.arange(self.n+1)]
        matrix = np.zeros((len(W), self.secrets.num_secrets))

        for i in np.arange(len(W)):
            for j in np.arange(self.secrets.num_secrets):
                matrix[i][j] = abs(W[i] - self.secrets.labels[j].count('a')/self.n)

        return Gain(self.secrets, W, matrix, is_loss=True)

    def _create_channel(self):
        Y = list(range(self.m+1))
        matrix = np.zeros((self.secrets.num_secrets, len(Y)))

        for i in np.arange(self.secrets.num_secrets):
            # The entry will be 1 only in the histogram that has the same number of 1's as the fist m people of secret x
            matrix[i][self.secrets.labels[i][:self.m].count("a")] = 1
            
        return Channel(self.secrets, Y, matrix)

    def prior_vul(self):
        """Prior vulnerability 'ground truth'."""
        return self.gain.prior_vulnerability()

    def post_vul(self):
        """Posterior vulnerability 'ground truth'."""
        return self.gain.posterior_vulnerability(self.hyper)

    @staticmethod
    def prior_vul_eq(inp):
        n, m = inp
        return 1/4 + 1/(4 * (floor(n/2) + ceil((n+1)/2)))

    @staticmethod
    def post_vul_eq(inp):
        n, m = inp
        post_vul = 0
        for y in np.arange(m+1):
            sums_by_k = []
            for k in np.arange(n+1):
                sum_on_yp = 0
                for yp in np.arange(n-m+1):
                    sum_on_yp = sum_on_yp + (binom(n-m,yp) * abs(k-y-yp))/binom(n,y+yp)
                sums_by_k.append(sum_on_yp)
            post_vul = post_vul + binom(m,y)*min(sums_by_k)
        return post_vul/(n*(n+1))

def test_close_prior():
    """Check whether prior vulnerability of closed formula 
    for utility loss function matches the code."""    
    for n in tqdm(np.arange(1,15), desc="Test closed formula utility"):
        for m in np.arange(1,n+1):
            prior_code = ModelBinaryUtility(n, m).prior_vul()
            prior_eq = ModelBinaryUtility.prior_vul_eq((n,m))
            if not float_equal(prior_code, prior_eq):
                print(colored("[Failed]", "red") +\
                    " at n = %d, m = %d, CODE = %.3f, EQ = %.3f"%(n,m,post_code,post_eq))
                return
    print(colored("[Successful]", "green") + " - Closed formula matches the code")

def test_closed_post():
    """Check whether posterior vulnerability of closed formula 
    for utility loss function matches the code."""    
    for n in tqdm(np.arange(2,15), desc="Test closed formula utility"):
        for m in np.arange(1,n+1):
            post_code = ModelBinaryUtility(n, m).post_vul()
            post_eq = ModelBinaryUtility.post_vul_eq((n,m))
            if not float_equal(post_code, post_eq):
                print(colored("[Failed]", "red") +\
                    " at n = %d, m = %d, CODE = %.3f, EQ = %.3f"%(n,m,post_code,post_eq))
                return
    print(colored("[Successful]", "green") + " - Closed formula matches the code")

def main():
    # test_close_prior()
    test_closed_post()
    
if __name__ == "__main__":
    main()
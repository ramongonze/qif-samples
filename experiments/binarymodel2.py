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


class ModelBinaryTUNK:
    
    def __init__(self, n:int, m:int):
        """Second model used to capture the adversary that doesn't
        know whether his target is in or out the sample.

        Parameters
        ----------
        n : int 
            Population's size.
        
        m : int
            Sample's size
        """

        self.n = n
        self.m = m
        self.secrets = self._create_secrets(n)
        self.gain = self._create_gain()
        self.channel = self._create_channel()
        self.hyper = Hyper(self.channel)

    def _create_secrets(self, n:int):
        X = ["".join(x) for x in it.product("01", repeat=n)]
        X = [(p,t) for t in np.arange(n) for p in X]
        prior = np.zeros(n*2**n)
        for i in np.arange(n*2**n):
            prior[i] = 1/(n * (n+1) * binom(n, X[i][0].count("1")))
        
        return Secrets(X, prior)

    def _create_gain(self):
        """Adversary has a single target and she doesn't know wheter the target is in or out the sample.
        She gains 1 if she guesses the target's value correctly or 0 otherwise.
        """
        W = ["0","1"]
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
            matrix[i][self.secrets.labels[i][0][:self.m].count("1")] = 1
            
        return Channel(self.secrets, Y, matrix)

    def prior_vul(self):
        """Prior vulnerability 'ground truth'."""
        return self.gain.prior_vulnerability()

    def post_vul_gt(self):
        """Posterior vulnerability 'ground truth'."""
        return self.gain.posterior_vulnerability(self.hyper)

    @staticmethod
    def post_vul_th(p):
        """Close formula for the same adversary of gain function gain_privacy_target_in_s."""
        n, m = p # Population size, sample size
        if m%2 == 0:
            return (3*n*m + 2*m + 2*n)/(4*n*(m+1))
        return (3*n*m + 2*m + 5*n + 2)/(4*n*(m+2))

def exp1():
    """Check wheter posterior vulnerability of closed formula matches
    the ground truth for 1 <= m <= n <= 14.
    """
    
    for n in tqdm(np.arange(1,15), desc="Experiment 1", ):
        for m in np.arange(1,n):
            post_gt = ModelBinaryTUNK(n, m).post_vul_gt()
            post_th = ModelBinaryTUNK.post_vul_th((n,m))
            if not float_equal(post_gt, post_th):
                print(colored("[Failed]", "red") +\
                    " at n = %d, m = %d, GT = %.3f, TH = %.3f"%(n,m,post_gt,post_th))
                return
    print(colored("[Successful]", "green") + " - Equation matches the ground truth")

def exp2():
    """Plot posterior vulnerabilities for a fixed n and varying m."""
    font = {
        'size'   : 12
    }
    plt.rc('font', **font)

    n_range = [10**7]
    m_range = np.array(range(1,101))/100
    for n in tqdm(n_range, desc="Experiment 1"):
        with Pool(6) as p:
            posts = p.map(ModelBinaryTUNK.post_vul_th, [(n,m) for m in m_range])
            plt.plot(m_range, posts)
            plt.scatter(m_range, posts, s=10, label="$n=10^{%d}$"%(int(log10(n))))

    plt.title("Posterior vulnerability for the unknown target and $n = 10^{%d}$"%(int(log10(n))))
    plt.xlabel("Sample size (% of population)")
    plt.ylabel("Posterior vulnerability")
    # plt.xscale("log")
    plt.xticks(np.array(range(0,101,10))/100)
    plt.yticks(np.linspace(0.5, 1, 5))
    plt.grid()
    plt.legend()
    plt.show()

def exp3():
    """Plot posterior vulnerability for a fixed m and varying n."""

    font = {
        'size'   : 12
    }
    plt.rc('font', **font)

    n_range = np.linspace(100, 10000, 100)
    m_range = [50, 100]
    for m in m_range:
        with Pool(6) as p:
            posts = p.map(ModelBinaryTUNK.post_vul_th, [(n,m) for n in n_range])
            plt.plot(n_range, posts)
            plt.scatter(n_range, posts, label="m = %d"%(m), s=5)

    plt.title("Posterior vulnerability when n inscreases")
    plt.xlabel("n")
    plt.ylabel("Posterior vulnerability")
    # plt.yticks(list(map(lambda x : round(x,2), np.linspace(1/2, 3/4, 5))))
    # plt.xscale("log")
    plt.grid()
    plt.legend()
    plt.show()

def main():
    # exp1()
    exp2() 
    # exp3()   

if __name__ == "__main__":
    main()
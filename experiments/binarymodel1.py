import numpy as np
import itertools as it
from fractions import Fraction
from math import floor, ceil, factorial
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

class ModelBinary:

    def __init__(self, n:int, m:int, gain:str):
        """Binary model.

        Parameters
        ----------
        n : int 
            Population's size.
        
        m : int
            Sample's size

        gain : str
            "privacy - target in sample" or "privacy - target out of sample"
        """

        self.n = n
        self.m = m
        self.secrets = self._create_secrets(n)
        self.gain = self._create_gain(gain)
        self.channel = self._create_channel()
        self.hyper = Hyper(self.channel)

    def _create_secrets(self, n:int):
        X = ["".join(x) for x in it.product("01", repeat=n)]
        prior = np.zeros(2**n)
        for i in np.arange(2**n):
            prior[i] = 1/((n+1) * binom(n, X[i].count("1")))
        
        return Secrets(X, prior)

    def _create_gain(self, gain):
        if gain == "privacy - target in sample":
            return self._create_gain_privacy_target_in_s()
        elif gain == "privacy - target out of sample":
            return self._create_gain_privacy_target_out_of_s()

    def _create_gain_privacy_target_in_s(self):
        """Adversary has a single target and she knows the target is in the sample.
        She gains 1 if she guesses the target's value correctly or 0 otherwise.
        """
        W = ["0","1"]
        matrix = np.zeros((len(W), self.secrets.num_secrets))

        for i in np.arange(len(W)):
            for j in np.arange(self.secrets.num_secrets):
                matrix[i][j] = 1 if self.secrets.labels[j][0] == W[i] else 0

        return Gain(self.secrets, W, matrix)

    def _create_gain_privacy_target_out_of_s(self):
        """Adversary has a single target and she knows the target is out of the sample.
        She gains 1 if she guesses the target's value correctly or 0 otherwise.
        """
        W = ["0","1"]
        matrix = np.zeros((len(W), self.secrets.num_secrets))

        for i in np.arange(len(W)):
            for j in np.arange(self.secrets.num_secrets):
                matrix[i][j] = 1 if self.secrets.labels[j][self.m] == W[i] else 0

        return Gain(self.secrets, W, matrix)

    def _create_channel(self):
        Y = list(range(self.m+1))
        matrix = np.zeros((self.secrets.num_secrets, len(Y)))

        for i in np.arange(self.secrets.num_secrets):
            # The entry will be 1 only in the histogram that has the same number of 1's as the fist m people of secret x
            matrix[i][self.secrets.labels[i][:self.m].count("1")] = 1
            
        return Channel(self.secrets, Y, matrix)

    def prior_vul(self):
        """Prior vulnerability 'ground truth'."""
        return self.gain.prior_vulnerability()

    def post_vul_gt(self):
        """Posterior vulnerability 'ground truth'."""
        return self.gain.posterior_vulnerability(self.hyper)

    @staticmethod
    def post_vul_privacy_target_in_s_th(p):
        """Close formula for the same adversary of gain function gain_privacy_target_in_s."""
        n,m = p
        return 3/4 + 1/(4*(floor((m+1)/2) + ceil(m/2)))
        # if m%2 == 1:
        #     return 3/4 + 1/(4*m)
        # return 3/4 + 1/(4*(m+1))

    @staticmethod
    def post_vul_privacy_target_out_s_th(p):
        """Close formula for the same adversary of gain function gain_privacy_target_out_of_s."""
        n,m = p
        return 3/4 - 1/(4*(floor(m/2)+ceil(m/2)+1))
        # if m%2 == 1:
        #     return 3/4 - 1/(4*(m+2))
        # return 3/4 - 1/(4*(m+1))

# Used to run in parallel
def get_post_vul_gt(p):
    n, m = p
    return ModelBinary(n, m).post_vul_gt()

def exp1():
    """Check wheter posterior vulnerability of closed formula is matches
    the ground truth for 1 <= m <= n <= 15.
    Adversary: Single target, she knows the target is in the sample.
    """
    
    for n in tqdm(np.arange(1,16), desc="Experiment 1", ):
        for m in np.arange(1,n):
            post_gt = ModelBinary(n, m, gain="privacy - target in sample").post_vul_gt()
            post_th = ModelBinary.post_vul_privacy_target_in_s_th((n,m))
            if not float_equal(post_gt, post_th):
                print(colored("[Failed]", "red") + " at n = %d, m = %d, GT = %.3f, TH = %.3f"%(n,m,post_gt,post_th))
                return
    print(colored("[Successful]", "green") + " - Equation matches the ground truth")

def exp2():
    """Create a graph m x posterior vulnerability (m = sample size) for a fixed n.
    Adversary: Single target, she know the target is in the sample.
    """
    font = {
        'size'   : 12
    }

    plt.rc('font', **font)

    n = 201
    range_m = list(np.arange(1,n+1,2)) 
    with Pool(6) as p:
        posts = p.map(ModelBinary.post_vul_privacy_target_in_s_th, [(n,m) for m in range_m])

    plt.title("Fixed n = %d"%(n))
    plt.ylim(-0.1, 1.1)
    plt.xticks(range(1,n+1,10))
    plt.xlabel("m")
    plt.ylabel("Posterior vulnerability")
    plt.scatter(range_m, posts)
    plt.text(n-10, posts[-1]+0.05, "%.5f"%(posts[-1]))
    plt.plot([n,n-5],[posts[-1]+0.01,posts[-1]+0.045])
    plt.show()    

def exp3():
    """Create a graph m x posterior vulnerability (m = sample size) when m = X% of n.
    Adversary: Single target, she know the target is in the sample.
    """
    font = {
        'size'   : 12
    }

    plt.rc('font', **font)
    m_values = [0.025, 0.05, 0.075, 0.1]
    colors = ['red', 'orange', 'green', 'blue']
    range_n = list(np.arange(101,1001,10))
    posts = dict()
    with Pool(6) as p:
        for m in m_values:
            posts[m] = p.map(ModelBinary.post_vul_privacy_target_in_s_th, [(n,int(m*n)+1 if int(m*n)%2 == 0 else int(m*n)) for n in range_n])

    plt.title("Posterior vulnerability when m = X% of n")
    plt.ylim(-0.1, 1.1)
    plt.xlim(50,1100)
    plt.xticks(range(100,1001,100))
    plt.xlabel("n")
    plt.ylabel("Posterior vulnerability")
    i = 0
    for m in m_values:
        plt.scatter(range_n, posts[m], label="m = %.1f%%"%(m*100), color=colors[i])
        plt.text(range_n[-1]+10, posts[m_values[0]][-1]-i*0.025 +0.025, "%.5f"%(posts[m][-1]), color=colors[i])
        i += 1
    plt.legend()
    plt.show()

def exp4():
    """Experiment 4. Check wheter posterior vulnerability of closed formula is matches
    the ground truth for 1 <= m <= n <= 15.
    Adversary: Single target, she knows the target is out of the sample.
    """

    for n in tqdm(np.arange(1,16), desc="Experiment 4", ):
        for m in np.arange(1,n):
            post_gt = ModelBinary(n, m, gain="privacy - target out of sample").post_vul_gt()
            post_th = ModelBinary.post_vul_privacy_target_out_s_th((n,m))
            if not float_equal(post_gt, post_th):
                print(colored("[Failed]", "red") + " - at n = %d, m = %d, GT = %.3f, TH = %.3f"%(n,m,post_gt,post_th))
                return
    print(colored("[Successful]", "green") + " - Equation matches the ground truth")

def main():
    exp1()
    # exp2()  
    # exp3()
    # exp4()

if __name__ == "__main__":
    main()
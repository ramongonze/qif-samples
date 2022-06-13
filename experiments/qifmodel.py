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

class ModelBinary:

    def __init__(self, n:int, m:int):
        """Binary model.

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
        self.gain = self._create_gain_adv_2i()
        self.channel = self._create_channel()
        self.hyper = Hyper(self.channel)

    def _create_secrets(self, n:int):
        X = ["".join(x) for x in it.product("01", repeat=n)]
        prior = np.zeros(2**n)
        for i in np.arange(2**n):
            prior[i] = 1/((n+1) * binom(n, X[i].count("1")))
        
        return Secrets(X, prior)

    def _create_gain_adv_2i(self):
        W = ["0","1"]
        matrix = np.zeros((len(W), self.secrets.num_secrets))

        for i in np.arange(len(W)):
            for j in np.arange(self.secrets.num_secrets):
                matrix[i][j] = 1 if self.secrets.labels[j][0] == W[i] else 0

        return Gain(self.secrets, W, matrix)

    def _create_channel(self):
        Y = list(range(self.m+1))
        matrix = np.zeros((self.secrets.num_secrets, len(Y)))

        for i in np.arange(self.secrets.num_secrets):
            # The entry will be 1 only in the histogram that has the same number of 1's as the fist m people of secret x
            matrix[i][self.secrets.labels[i][:self.m].count("1")] = 1
            
        return Channel(self.secrets, Y, matrix)

    def prior_vul(self):
        return self.gain.prior_vulnerability()

    def post_vul_gt(self):
        return self.gain.posterior_vulnerability(self.hyper)

    @staticmethod
    def post_vul_th(p):
        n,m = p
        return (m**2 - 2*floor(m/2)**2 + 3*m + floor(m/2)*(2*m - 2))/(2*m*(m+1))

# Used to run in parallel
def get_post_vul_gt(p):
    n, m = p
    return ModelBinary(n, m).post_vul_gt()

def exp1():
    """Experiment 1. Check the correctness of equation by comparying with the real matrix multiplication
    for n and m odd.
    """
    
    for n in np.arange(1,16):
        for m in np.arange(1,n+1):
            post_gt = ModelBinary(n, m).post_vul_gt()
            post_th = ModelBinary.post_vul_th((n,m))
            if not float_equal(post_gt, post_th):
                print("Failed at n = %d, m = %d, GT = %.3f, TH = %.3f"%(n,m,post_gt,post_th))
                return
    print("[Experiment 1] - Successful: Equation matches the ground truth")

def exp2():
    """Fixed n and varies m."""
    font = {
        'size'   : 12
    }

    plt.rc('font', **font)

    n = 201
    range_m = list(np.arange(1,n+1,2)) 
    with Pool(6) as p:
        posts = p.map(ModelBinary.post_vul_th, [(n,m) for m in range_m])

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
    """Posterior vulnerability when m = X% of n."""
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
            posts[m] = p.map(ModelBinary.post_vul_th, [(n,int(m*n)+1 if int(m*n)%2 == 0 else int(m*n)) for n in range_n])

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
    for n in np.arange(3,11):
        for m in np.arange(2,n+1):
            for y in np.arange(1,m+1):
                sum_ = 0
                for y2 in np.arange(n-m+1):
                    sum_ += (binom(m-1,y-1)*binom(n-m,y2)/binom(n,y+y2))
                if not float_equal(sum_, y/m):
                    print("[Experiment 4] - Failed. Equation doesn't match at:")
                    print("                 n = %d, m = %d, y = %d. sum = %.3f, y/m = %.3f"%(n,m,y,sum_,y/m))
                    return
    print("[Experiment 4] - Successfull")    

def exp5():
    for n in np.arange(1,50):
        for m in np.arange(1,n+1):
            for y in np.arange(m+1):
                sum_ = 0
                for y2 in np.arange(n-m+1):
                    sum_ += (binom(m-1,y-1)*binom(n-m,y2)/binom(n,y+y2))
                if not float_equal(((m+1)*sum_)/(n+1), y/m):
                    print("[Experiment 5] - Failed. Equation doesn't match at:")
                    print("                 n = %d, m = %d, y = %d. sum = %.3f, y/m = %.3f"%(n,m,y,sum_,y/m))
                    return
    print("[Experiment 5] - Successfull")  

def main():
    # exp1()
    # exp2()  
    exp3()
    # exp4()
    # exp5()

if __name__ == "__main__":
    main()
"""Given a n_max as argument this script generates posterior vulnerability for utility adversary
for all 1 <= m <= n <= n_max.

Usage:
    python post_vul_utility.py <n_min> <n_max> <output_file> <n_jobs>

n_min, n_max: Population size range
output_file: Output file name (with .csv extension)
n_jobs: Number of threads to be used

Output file format: Csv file containing a 2d matrix (n_max x n_max), where the value in position (i,j)
is the posterior vulnerability for n=i and m=j.
"""

import os
import sys
import numpy as np
from multiprocessing import Pool
from scipy.special import binom
from csv import writer

def post_vul_utility(inp):
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
    return "%.15f"%(post_vul/(n*(n+1)))

def main():
    n_min = int(sys.argv[1])
    n_max = int(sys.argv[2])
    output_file_name = sys.argv[3]
    n_jobs = int(sys.argv[4])
    
    # Check if output file exists
    if not os.path.exists(output_file_name):
        # Create an empty file
        f = open(output_file_name, "w")
        f.close()

    for n in np.arange(n_min,n_max+1):
        with Pool(n_jobs) as p:
            post = list(p.map(post_vul_utility, [(n,m) for m in np.arange(1,n+1)]))

        with open(output_file_name, "a") as csvfile:
            writer_object = writer(csvfile)
            writer_object.writerow(post)
            csvfile.close()

if __name__ == "__main__":
    main()
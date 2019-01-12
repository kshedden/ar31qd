import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/statsmodels_fork/statsmodels")

import pandas as pd
import numpy as np

method = 0

ase0 = pd.read_csv("person_sample_0_%d.csv" % method)
ase1 = pd.read_csv("person_sample_1_%d.csv" % method)
ase2 = pd.read_csv("person_sample_2_%d.csv" % method)

ase0 = ase0.rename(columns={"Total": "Total0"})
ase1 = ase1.rename(columns={"Total": "Total1"})
ase2 = ase2.rename(columns={"Total": "Total2"})

ase = pd.merge(ase0, ase1, left_on="Sample", right_on="Sample", how="outer")
ase = pd.merge(ase, ase2, left_on="Sample", right_on="Sample", how="outer")


def f(r): return 0.5*np.log((1+r)/(1-r))
def g(z): return (np.exp(2*z)-1)/(np.exp(2*z)+1)

for k1,k2 in (0,1),(0,2),(1,2):
    vn1 = "Total%d" % k1
    vn2 = "Total%d" % k2
    x = ase.loc[:, ["Person", "Sample", vn1, vn2]].dropna()
    a = ["MIG", "PIG", "CIG"][k1]
    b = ["MIG", "PIG", "CIG"][k2]
    r = np.corrcoef(x[vn1], x[vn2])[0, 1]
    n = x.iloc[:, 0].unique().size
    lb = g(f(r) - 2/np.sqrt(n))
    ub = g(f(r) + 2/np.sqrt(n))
    print("%s vs %s: %.3f (%.3f,%.3f)  %d samples, %d people" %
           (a, b, r, lb, ub, x.Sample.unique().size, x.Person.unique().size))

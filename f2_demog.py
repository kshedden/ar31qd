"""
Estimate the mean and standard deviation of the offspring HAZ at one year.

n=47 for this
"""

# ProcessMLE is not released yet, needs a recent master
import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/statsmodels_fork/statsmodels")

import matplotlib
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.process_regression import ProcessMLE

vname = "HAZ"

dk = pd.read_csv("imprint_full_10pct_0.csv")

df = pd.read_csv("/nfs/kshedden/Beverly_Strassmann/Cohort_2018.csv.gz")
mom = pd.read_csv("/nfs/kshedden/Beverly_Strassmann/F2_OneTime_Shedden.txt.gz", delimiter="\t")
kid = pd.read_csv("/nfs/kshedden/Beverly_Strassmann/F2_Long.csv.gz")

kid = kid.loc[kid.Surv_glob ==1, :]
kid = kid.loc[pd.notnull(kid.HAZ), :]

kid = kid[["ID_F2", "ID_mere", "Date_comb", "DateNaissance", "Sex", vname]]
kid["Age"] = (kid.Date_comb - kid.DateNaissance) / (60 * 60 * 24 * 365.25)
kid = kid.loc[kid.Age >= 0, :]

kid = kid.loc[pd.notnull(kid.Age), :]
kid = kid.loc[pd.notnull(kid.Sex), :]

age_mean = np.mean(kid.Age)
age_sd = np.std(kid.Age)
kid["Age_z"] = (kid.Age - age_mean) / age_sd

vname_cen = vname + "_z"
resp_mean = kid[vname].mean()
resp_sd = kid[vname].std()
kid[vname_cen] = (kid[vname] - resp_mean) / resp_sd

kid = kid.dropna()
kid = kid.sort_values(by=["ID_F2", "Age"])

fml = "%s ~ Age_z + I(Age_z**2) + Sex" % (vname + "_z")
if False:
    # The process model reduces to a random intercept model
    mod = ProcessMLE.from_formula(fml,
                                  scale_formula="Age_z",
                                  smooth_formula="1",
                                  noise_formula="1",
                                  time="Age",
                                  groups="ID_F2",
                                  data=kid)
    rslt = mod.fit(verbose=True)

mod = sm.MixedLM.from_formula(fml, groups="ID_F2", data=kid)
rslt = mod.fit()

# Drop people not in other analyses.
dk = dk.loc[~dk.ID_F2.isin([5084, 5126, 5128, 5138, 5143, 5149]), :]

new_exog = kid.reset_index().copy()
new_exog = new_exog.loc[new_exog.ID_F2.isin(dk.ID_F2), :]
new_exog = new_exog.groupby("ID_F2").head(1)
new_exog.loc[:, "Age"] = 1
new_exog.loc[:, "Age_z"] = (1 - age_mean) / age_sd
pr = rslt.predict(exog=new_exog)
pr = pr * resp_sd + resp_mean

haz_mean = pr.mean()
haz_sd = resp_sd * np.sqrt(rslt.scale + rslt.cov_re.iloc[0, 0])

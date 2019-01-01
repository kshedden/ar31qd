"""
Pre-gestational statistics for mothers (height, weight, BMI at time of conception).
"""

import numpy as np
import pandas as pd
from ase_data import get_data
import os
import statsmodels.api as sm

dk = pd.read_csv("imprint_full_10pct_0.csv")
df = pd.read_csv("/nfs/kshedden/Beverly_Strassmann/Cohort_2018.csv.gz")
mom = pd.read_csv("/nfs/kshedden/Beverly_Strassmann/F2_OneTime_Shedden.txt.gz", delimiter="\t")

# Data from very young ages is not relevant here
df = df.loc[df.Age_Yrs > 5, :]

# Build models based on women who become pregnant
df = df.loc[df.ID.isin(mom.ID_mere), :]

for c in "DOB_F1", "Pre_Gest_Date":
    mom[c] = pd.to_datetime(mom[c], errors='coerce')

vars = ["Ht_Ave_18", "Age_Yrs", "ID", "WT", "BMI_18"]
df = df[vars].dropna()

mom["Pre_gest_age"] = (mom.Pre_Gest_Date - mom.DOB_F1).dt.days / 365.25

dx = pd.merge(df, mom, left_on="ID", right_on="ID_mere")

age_mean = df.Age_Yrs.mean()
age_sd = df.Age_Yrs.std()
dx["Age_z"] = (dx.Age_Yrs - age_mean) / age_sd
for vname in "Ht_Ave_18", "WT", "BMI_18":

    resp_mean = df[vname].mean()
    resp_sd = df[vname].std()
    dx["resp_z"] = (dx[vname] - resp_mean) / resp_sd

    fml = "resp_z ~ bs(Age_z, 4)"
    mod = sm.MixedLM.from_formula(fml, groups="ID", data=dx)
    rslt = mod.fit()

    dp = dx.groupby("ID").head(1)
    dp["Age_z"] = (dp["Pre_gest_age"] - age_mean) / age_sd
    dp["pr"] = rslt.predict(exog=dp)

    dp.pr = resp_mean + resp_sd * dp.pr
    print(vname)
    print("%.3f" % dp.pr.mean())
    print("%.3f" % (resp_sd * np.sqrt(rslt.scale + rslt.cov_re.iloc[0, 0])))

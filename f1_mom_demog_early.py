"""
Mean and SD of mother's HAZ when she was two years old.
"""

import numpy as np
import pandas as pd
import os

dk = pd.read_csv("imprint_full_10pct_0.csv")

# Drop one kid from each sibship
drop = [60614, 59308, 60613]
dk = dk.loc[~dk.Person.isin(drop), :]

# Require average NonAltFreq to be present
dk = dk.loc[pd.notnull(dk.AvgNonAltFreq), :]

# Require the placenta weight to be present
dk = dk.loc[pd.notnull(dk.PlacentaWeight), :]

dm = pd.read_csv("F2_OneTime_Shedden.txt.gz", delimiter="\t")
dm = dm.rename(columns={"ID_mere": "ID_F1"})
for x in "DateNaissance", "DOB_F1":
    dm[x] = pd.to_datetime(dm[x])
dm["MomAge"] = dm.DateNaissance - dm.DOB_F1
dm["MomAge"] = dm.MomAge.dt.days.astype(np.float64)/365.25


for k in range(20):
    du = pd.read_csv(os.path.join("../Stature-SBP/imputed_data/HAZ_imp_%d.csv" % k))
    dx = pd.merge(du, dk, left_on="ID", right_on="ID_F1")
    dx = dx.groupby("ID")[["HAZ2"]].head(1)
    dx = dx.rename(columns={"HAZ2": "HAZ2_%d" % k})
    if k == 0:
        dxa = dx
    else:
        dxa = pd.merge(dxa, dx, left_index=True, right_index=True)

haz2_imp = dxa.mean(0)
haz2_var = np.mean(dxa.var(0) / dxa.shape[0]) + (1 + 1/20)*np.var(haz2_imp)
haz2_se = np.sqrt(haz2_var)
haz2 = haz2_imp.mean()

print(dxa.mean(1).describe())

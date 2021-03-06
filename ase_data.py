import numpy as np
import pandas as pd


def get_data(method):

    fn = "imprint_full_10pct_%d.csv" % method
    da = pd.read_csv(fn)

    # Drop exons that are not known to be imprinted
    da = da.loc[da.Icode <= 2, :]

    da["Mat"] = 1*(da.Icode == 0)
    da["Pat"] = 1*(da.Icode == 1)
    da["Cmp"] = 1*(da.Icode == 2)

    genecode, exoncode = {}, {}
    for i in range(da.shape[0]):
        r = da.iloc[i, :]
        if r.Mat == 1:
            genecode[r.Gene] = 0
            exoncode[r.Exon] = 2
        else:
            genecode[r.Gene] = 1
            exoncode[r.Exon] = 3

    # Only a few variables should have missing values
    allow_missing = ["Malaria", "PlacentaWeight", "RIN", "ICR", "GeneClass_c1_lnc2_nc3", "AvgNonAltFreq"]
    dm = pd.isnull(da.drop(allow_missing, axis=1)).sum(0)
    if (dm > 0).any():
        print("Unexpected missing values:")
        print(dm[dm > 0])
        raise ValueError("")

    da = da.drop(["Malaria", "ICR"], axis=1)

    # KidRank = 1 means that the kid is second or later born
    da["KidRank"] = (da["KidRank"] >= 2).astype(np.int)

    da["Boy"] = (da.Sex == 1).astype(np.int)

    # Convert units to kilograms
    da.BirthWeight /= 1000
    da.PlacentaWeight /= 1000

    # Save uncentered versions of these variables before centering
    da["Boy01"] = da.Boy.copy()
    da["Mat01"] = da.Mat.copy()
    da["Pat01"] = da.Pat.copy()
    da["Cmp01"] = da.Cmp.copy()

    # Center
    da.Boy -= da.Boy.mean()
    da.Mat -= da.Mat.mean()
    da.Pat -= da.Pat.mean()
    da.Cmp -= da.Cmp.mean()

    # Fix the placenta weight for a twin birth
    ii = da.Sample.isin(['5129_0', '5129_1'])
    da.loc[ii, "PlacentaWeight"] /= 2

    return da, genecode, exoncode

import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/statsmodels_fork/statsmodels")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from ase_data import get_data

# 0=gene/sample masking, 1=sample masking
method = int(sys.argv[1])

da, genecode, exoncode = get_data(method)

ii = pd.notnull(da.AvgNonAltFreq)
da = da.loc[ii, :]

for mm in 1047, 196, 113:
    ii = da.loc[da.MomID == mm, "Person"].unique()
    if len(ii) == 2:
        n0 = np.sum(da.MomID == ii[0])
        n1 = np.sum(da.MomID == ii[1])
        idx = ii[0] if n1 > n0 else ii[1]
        m = da.shape[0]
        da = da.loc[da.Person != idx, :]
        m -= da.shape[0]

# Ony keep MIGs and PIGs
da = da.loc[da.Icode <= 1, :]

ps = da.loc[:, ["Person", "Sample"]].drop_duplicates()
ps = list(zip(ps.Person, ps.Sample))

eg = da.loc[:, ["Exon", "Gene"]].drop_duplicates()
eg = list(zip(eg.Exon, eg.Gene))

ix = pd.MultiIndex.from_product([ps, eg]).tolist()
z = [[x[0][0], x[0][1], x[1][0], x[1][1]] for x in ix]

dx = pd.DataFrame(z, columns=["Person", "Sample", "Exon", "Gene"])

da = da.set_index(["Person", "Sample", "Exon", "Gene"])
dx = dx.set_index(["Person", "Sample", "Exon", "Gene"])

dr = pd.merge(dx, da, left_index=True, right_index=True, how='left')
dr["Obs"] = pd.notnull(dr.Imprinted).astype(np.float64)

# Fill missing for person/sample characteristics
dr = dr.reset_index().set_index(["Person", "Sample"])
dr.loc[:, "Boy"] = dr.Boy.fillna(value=dr.Boy.dropna().to_dict())
dr.loc[:, "RIN"] = dr.KidRank.fillna(value=dr.RIN.dropna().to_dict())
dr.loc[:, "AvgNonAltFreq"] = dr.KidRank.fillna(value=dr.AvgNonAltFreq.dropna().to_dict())
dr.loc[:, "X_chrM_TPMsum"] = dr.KidRank.fillna(value=dr.X_chrM_TPMsum.dropna().to_dict())
dr.loc[:, "KidRank"] = dr.KidRank.fillna(value=dr.KidRank.dropna().to_dict())
dr.loc[:, "BirthLength"] = dr.BirthLength.fillna(value=dr.BirthLength.dropna().to_dict())
dr = dr.reset_index()

# Fill missing for gene/exon characteristics
dr = dr.reset_index().set_index(["Gene", "Exon"])
dr.loc[:, "Icode"] = dr.Icode.fillna(value=dr.Icode.dropna().to_dict())
dr.loc[:, "Lib"] = dr.Lib.fillna(value=dr.Lib.dropna().to_dict())
dr.loc[:, "GeneClass"] = dr.GeneClass_c1_lnc2_nc3.fillna(value=dr.GeneClass_c1_lnc2_nc3.dropna().to_dict())
dr = dr.reset_index()

dr["Mat"] = 1*(dr.Icode == 0)
dr["Pat"] = 1*(dr.Icode == 1)

# Randomize
#p = dr.Obs.mean()
#dr["Obs"] = 1*(np.random.uniform(size=dr.shape[0]) < p)

out = open("nonmissing_%d.txt" % method, "w")
out.write("```\n")

rsltx = []
for k in 0,1:

    dx = dr.loc[dr.Icode == k, :]
    dx = dx[["Obs", "KidRank", "Lib", "Boy", "GeneClass", "BirthLength", "Person", "Sample",
             "Gene", "Exon", "RIN", "AvgNonAltFreq", "X_chrM_TPMsum"]]

    assert(pd.isnull(dx).any().any() is not False)

    dx["BirthLength_cen"] = dx.BirthLength - dx.BirthLength.mean()

    fml = "Obs ~ KidRank + C(Lib) + Boy + C(GeneClass) + BirthLength_cen + RIN + AvgNonAltFreq + X_chrM_TPMsum"
    vc_fml = {"Person": "0 + C(Person)", "Sample": "0 + C(Sample)", "Gene": "0 + C(Gene)", "Exon": "0 + C(Exon)"}

    model = sm.genmod.BinomialBayesMixedGLM.from_formula(fml, vc_fml, dx, vcp_p=3, fe_p=3)
    rslt = model.fit_vb(verbose=False)
    rsltx.append(rslt)
    out.write(["MIG:", "PIG:"][k] + "\n")
    out.write("%d imprinting status observations (%d observed, %d missing)\n" %
              (dx.shape[0], dx.Obs.sum(), (1 - dx.Obs).sum()))
    out.write("%d distinct people\n" % dx.Person.unique().size)
    out.write("%d distinct samples\n" % dx.Sample.unique().size)
    out.write("%d distinct genes\n" % dx.Gene.unique().size)
    out.write("%d distinct exons\n" % dx.Exon.unique().size)
    out.write(rslt.summary().as_text() + "\n\n")

x = dr.BirthLength.dropna()
dr["BirthLength_cen"] = (dr.BirthLength - x.mean()) / x.std()
fml = "Obs ~ KidRank + C(Lib) + Boy + Mat + C(GeneClass) + BirthLength_cen + RIN + AvgNonAltFreq + X_chrM_TPMsum"
vc_fml = {"Person": "0 + C(Person)", "Sample": "0 + C(Sample)", "Gene": "0 + C(Gene)", "Exon": "0 + C(Exon)"}

model = sm.genmod.BinomialBayesMixedGLM.from_formula(fml, vc_fml, data=dr, vcp_p=3, fe_p=3)
rslt = model.fit_vb()
out.write("Combined:\n")
out.write(rslt.summary().as_text() + "\n")

out.write("```\n")
out.close()

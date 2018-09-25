import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/statsmodels_fork/statsmodels")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from ase_data import get_data

# 0=gene/sample masking, 1=sample masking
method = int(sys.argv[1])

da, genecode, exoncode = get_data(method)

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
dr.loc[:, "KidRank"] = dr.KidRank.fillna(value=dr.KidRank.dropna().to_dict())
dr = dr.reset_index()

# Fill missing for gene/exon characteristics
dr = dr.reset_index().set_index(["Gene", "Exon"])
dr.loc[:, "Icode"] = dr.Icode.fillna(value=dr.Icode.dropna().to_dict())
dr.loc[:, "Lib"] = dr.Lib.fillna(value=dr.Lib.dropna().to_dict())
dr.loc[:, "GeneClass"] = dr.GeneClass_c1_lnc2_nc3.fillna(value=dr.GeneClass_c1_lnc2_nc3.dropna().to_dict())
dr = dr.reset_index()

dr["Mat"] = 1*(dr.Icode == 0)
dr["Pat"] = 1*(dr.Icode == 1)
dr["Complex"] = 1*(dr.Icode == 2)

# Randomize
#p = dr.Obs.mean()
#dr["Obs"] = 1*(np.random.uniform(size=dr.shape[0]) < p)

out = open("nonmissing_%d.txt" % method, "w")
out.write("```\n")

rsltx = []
for k in 0,1,2:

    dx = dr.loc[dr.Icode == k, :]

    fml = "Obs ~ KidRank + C(Lib) + Boy + C(GeneClass)"
    # fml = "Obs ~ 1"
    vc_fml = {"Person": "0 + C(Person)", "Sample": "0 + C(Sample)", "Gene": "0 + C(Gene)", "Exon": "0 + C(Exon)"}

    model = sm.genmod.BinomialBayesMixedGLM.from_formula(fml, vc_fml, dx, vcp_p=3, fe_p=3)
    rslt = model.fit_vb(verbose=True)
    rsltx.append(rslt)
    out.write(["MIG:", "PIG:", "CIG:"][k] + "\n")
    out.write("%d imprinting status observations (%d observed, %d missing)\n" %
              (dx.shape[0], dx.Obs.sum(), (1 - dx.Obs).sum()))
    out.write("%d distinct people\n" % dx.Person.unique().size)
    out.write("%d distinct samples\n" % dx.Sample.unique().size)
    out.write("%d distinct genes\n" % dx.Gene.unique().size)
    out.write("%d distinct exons\n" % dx.Exon.unique().size)
    out.write(rslt.summary().as_text() + "\n\n")

fml = "Obs ~ KidRank + C(Lib) + Boy + Mat + Pat"
#fml = "Obs ~ 1"
vc_fml = {"Person": "0 + C(Person)", "Sample": "0 + C(Sample)", "Gene": "0 + C(Gene)", "Exon": "0 + C(Exon)"}

model = sm.genmod.BinomialBayesMixedGLM.from_formula(fml, vc_fml, data=dr, vcp_p=3, fe_p=3)
rslt = model.fit_vb()
out.write("Combined:\n")
out.write(rslt.summary().as_text() + "\n")

out.write("```\n")
out.close()

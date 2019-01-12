import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/statsmodels_fork/statsmodels")

import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from ase_data import get_data
import statsmodels.api as sm
import patsy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

method = 0
da, genecode, exoncode = get_data(method)

ii = pd.notnull(da.AvgNonAltFreq)
da = da.loc[ii, :]

if True:
    # Mom IDs with multiple sibs: 1047, 196, 113
    for mm in 1047, 196, 113:
        ii = da.loc[da.MomID == mm, "Person"].unique()
        if len(ii) == 2:
            n0 = np.sum(da.MomID == ii[0])
            n1 = np.sum(da.MomID == ii[1])
            idx = ii[0] if n1 > n0 else ii[1]
            m = da.shape[0]
            da = da.loc[da.Person != idx, :]
            m -= da.shape[0]
        else:
            1/0
print("After dropping siblings", da.shape)

pdf = PdfPages("posterior_sensitivity_graphs.pdf")

out = open("posterior_sensitivity.txt", "w")
out.write("```\n")

# Imprinting code
for kc in 0, 1, 3:

    # Method of dropping genes
    for drop in 0, 1, 2, 3:

        if kc == 0:
            out.write("MIGs only\n")
        elif kc == 1:
            out.write("PIGs only\n")
        elif kc == 2:
            out.write("CIGs only\n")
        elif kc == 3:
            out.write("MIG+PIG only\n")

        if drop == 0:
            out.write("Drop genes with most data first\n\n")
        elif drop == 1:
            out.write("Drop genes with least data first\n\n")
        elif drop == 2:
            out.write("Drop samples with most data first\n\n")
        elif drop == 3:
            out.write("Drop samples with least data first\n\n")

        rslt0, par0 = [], []
        if kc < 3:
            dx = da.loc[da.Icode == kc, :].copy()
        else:
            dx = da.loc[da.Icode.isin([0, 1]), :].copy()

        dropped = []
        for j in range(5):

            vcn = ["Sample", "Exon", "Gene", "Person"]
            fml = "Imprinted ~ KidRank + C(Lib) + Boy"

            vc_fml = {"Sample": "0 + C(Sample)", "Exon": "0 + C(Exon)",
                      "Gene": "0 + C(Gene)", "Person": "0 + C(Person)"}

            if kc == 3:
                fml += " + Pat"
                fml = fml.replace("Pat", "Pat01")
                fml = fml.replace("C(Lib)", "C(Lib)*Pat01")
                fml = fml.replace("KidRank", "KidRank*Pat01")

            dy = dx.drop("PlacentaWeight", axis=1)

            if kc != 3:
                model = BinomialBayesMixedGLM.from_formula(fml, vc_fml, dy, vcp_p=3, fe_p=3)
            else:
                ident = []
                exog_vc = []

                for g in dy.Gene.unique():
                    ident.append(genecode[g])
                    exog_vc.append((dy.Gene == g).astype(np.int))

                for e in dy.Exon.unique():
                    ident.append(exoncode[e])
                    exog_vc.append((dy.Exon == e).astype(np.int))

                for p in dy.Person.unique():
                    ident.append(4)
                    exog_vc.append((dy.Person == p).astype(np.int))

                for s in dy.Sample.unique():
                    ident.append(5)
                    exog_vc.append((dy.Sample == s).astype(np.int))

                exog_vc = np.vstack(exog_vc).T
                ident = np.asarray(ident)

                endog, exog = patsy.dmatrices(fml, data=dy, return_type='dataframe')
                vcp_names = ["Gene(Mat)", "Gene(Pat)", "Exon(Mat)", "Exon(Pat)", "Person", "Sample"]
                model = BinomialBayesMixedGLM(endog, exog, exog_vc, ident, vcp_names=vcp_names, vcp_p=3, fe_p=3)

            print("kc=%d drop=%d j=%d" % (kc, drop, j), dy.shape)

            if kc != 3:
                model2 = BinomialBayesMixedGLM.from_formula(fml, vc_fml, dy, vcp_p=3, fe_p=3)
            else:
                model2 = BinomialBayesMixedGLM(endog, exog, exog_vc, ident, vcp_p=3, fe_p=3, vcp_names=vcp_names)

            rslt2 = model2.fit_vb(verbose=False)
            out.write("n=%d imprinting calls\n" % dy.shape[0])
            out.write("%d distinct samples\n" % dy.Sample.unique().size)
            out.write("%d distinct people\n" % dy.Person.unique().size)
            out.write("%d distinct moms\n" % dy.MomID.unique().size)
            out.write("%d distinct genes\n" % dy.Gene.unique().size)
            out.write("%d distinct exons\n" % dy.Exon.unique().size)
            out.write("Dropped: %s\n" % ", ".join(dropped))
            out.write(rslt2.summary().as_text() + "\n\n")

            rslt0.append(rslt2)
            par0.append(np.exp(rslt2.vcp_mean))

            if drop == 0:
                # Drop genes with most data first
                dd = dx.groupby("Gene").size().sort_values(ascending=False)
                dropped.append(dd.index[0])
                dx = dx.loc[dx.Gene != dd.index[0], :]
            elif drop == 1:
                # Drop genes with least data first
                dd = dx.groupby("Gene").size().sort_values()
                dropped.append(dd.index[0])
                dx = dx.loc[dx.Gene != dd.index[0], :]
            elif drop == 2:
                # Drop samples with most data first
                dd = dx.groupby("Sample").size().sort_values(ascending=False)
                dropped.append(dd.index[0])
                dx = dx.loc[dx.Sample != dd.index[0], :]
            elif drop == 3:
                # Drop samples with least data first
                dd = dx.groupby("Sample").size().sort_values()
                dropped.append(dd.index[0])
                dx = dx.loc[dx.Sample != dd.index[0], :]

        for jj in 0,1:
            par0 = np.asarray(par0)
            plt.clf()
            plt.grid(True)
            if jj == 0:
                plt.plot(par0[:, 0], par0[:, 3], 'o', alpha=0.6)
                plt.xlabel("Sample SD", size=15)
                plt.ylabel("Person SD", size=15)
                x = 1.1 * max(par0[:, 0].max(), par0[:, 3].max())
            else:
                plt.plot(par0[:, 1], par0[:, 2], 'o', alpha=0.6)
                plt.xlabel("Exon SD", size=15)
                plt.ylabel("Gene SD", size=15)
                x = 1.1 * max(par0[:, 1].max(), par0[:, 2].max())
            plt.title(["MIG", "PIG", "CIG", "MIG+PIG"][kc] + " imprinted genes, " +
                      {0: "drop genes from top", 1: "drop genes from bottom",
                       2: "drop samples from top", 3: "drop samples from bottom",
                       4: "drop 10% randomly"}[drop])
            plt.xlim(0, x)
            plt.ylim(0, x)
            pdf.savefig()

pdf.close()

out.write("```\n")
out.close()

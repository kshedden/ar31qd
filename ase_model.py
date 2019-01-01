import sys
sys.path.insert(
    0, "/afs/umich.edu/user/k/s/kshedden/statsmodels_fork/statsmodels")

import numpy as np
import pandas as pd
from ase_data import get_data
from collections import OrderedDict
import patsy
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import statsmodels.api as sm
import sys

# 0=gene/sample masking, 1=sample masking
method = int(sys.argv[1])

da, genecode, exoncode = get_data(method)

info = open("model_%d_info.csv" % method, "w")
out = open("model_%d.txt" % method, "w")
log = open("model_%d_log.txt" % method, "w")

out.write("```\n")
log.write("```\n")

# Imprinting code
for kc in (0, 1, 2, 3):

    # If true, drop one kid from each sib pair to get unrelated samples
    for ks in False, True:

        # If true, include placenta weight in the model
        for kp in False, True:

            log.write("Imprinting code: %d\n" % kc)
            if ks:
                log.write("Dropping all but one sibling from multi-sib families\n")
            if kp:
                log.write("Including PlacentaWeight in model\n")

            if kc < 3:
                # MIG/PIG/CIG models
                dx = da.loc[da.Icode == kc, :]
            else:
                # MIG+PIG model
                dx = da.loc[da.Icode.isin([0, 1]), :]

            if ks:
                # Mom IDs with multiple sibs: 1047, 196, 113
                for mm in 1047, 196, 113:
                    ii = dx.loc[dx.MomID == mm, "Person"].unique()
                    if len(ii) == 2:
                        n0 = np.sum(dx.MomID == ii[0])
                        n1 = np.sum(dx.MomID == ii[1])
                        idx = ii[0] if n1 > n0 else ii[1]
                        m = dx.shape[0]
                        dx = dx.loc[dx.Person != idx, :]
                        m -= dx.shape[0]
                        log.write("Dropped %d rows for momid %d (dropping subject %d from sibs %s %s)\n" %
                                  (m, mm, idx, ii[0], ii[1]))
                    else:
                        log.write("No multiple sibs for momid=%d " % mm)
                        log.write("(%d kid found for this momid)\n" % len(ii))

            fml = "Imprinted ~ AvgNonAltFreq + C(Lib) + Batch + Boy + KidRank + BirthLength_cen + RIN + X_chrM_TPMsum + C(GeneClass_c1_lnc2_nc3)"

            if kp:
                # Add placenta weight to some models
                fml += " + PlacentaWeight_cen"

            # Variance terms
            vc_fml = OrderedDict([
                ("Person", "0 + C(Person)"),
                ("Sample", "0 + C(Sample)"),
                ("Gene", "0 + C(Gene)"),
                ("Exon", "0 + C(Exon)"),
            ])

            if kc == 3:
                # Extra terms for MIG+PIG model
                fml += " + Pat"
                fml = fml.replace("Pat", "Pat01")
                fml = fml.replace("C(Lib)", "C(Lib)*Pat01")

            if not kp:
                dy = dx.drop("PlacentaWeight", axis=1)
            else:
                dy = dx

            # This is the final data set
            m = dy.shape[0]
            idm = pd.isnull(dy).sum(0)
            dy = dy.dropna()
            m -= dy.shape[0]
            if m > 0:
                idm = idm[idm.values > 0]
                log.write("Lost %d rows due to missing values\n" % m)
                log.write("Missing values:\n")
                log.write("%s\n" % idm.to_string())

            log.write("Final counts:\n")
            log.write("%d distinct people\n" % dy.Person.unique().size)
            log.write("%d distinct mothers\n" % dy.MomID.unique().size)
            log.write("%d distinct samples\n" % dy.Sample.unique().size)
            log.write("%d imprinting calls\n" % dy.shape[0])
            log.write("%d distinct genes\n" % dy.Gene.unique().size)
            log.write("%d distinct exons\n" % dy.Exon.unique().size)
            log.write("\n")

            # Create some centered variables
            dy["BirthLength_cen"] = dy.BirthLength - dy.BirthLength.mean()
            dy["BirthWeight_cen"] = dy.BirthWeight - dy.BirthWeight.mean()

            # Write out gene-level info
            info.write("Genes:\n")
            for ky, va in dy.groupby("Gene"):
                info.write("%s,%d\n" % (ky, va.shape[0]))

            # Write out sample-level info
            info.write("\nSamples:\n")
            for ky, va in dy.groupby("Sample"):
                info.write("%s,%d\n" % (ky, va.shape[0]))

            if kp:
                # Center placenta weight if we are using it
                dy["PlacentaWeight_cen"] = dy.PlacentaWeight - dy.PlacentaWeight.mean()

            if kc != 3:
                model = BinomialBayesMixedGLM.from_formula(
                    fml, vc_fml, dy, vcp_p=3, fe_p=3)

            else:
                # Design matrices for MIG+PIG model can't be constructed with
                # formulas, need to do it manually
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

                endog, exog = patsy.dmatrices(
                    fml, data=dy, return_type='dataframe')
                vcp_names = [
                    "Gene(Mat)", "Gene(Pat)", "Exon(Mat)", "Exon(Pat)",
                    "Person", "Sample"
                ]
                model = BinomialBayesMixedGLM(
                    endog,
                    exog,
                    exog_vc,
                    ident,
                    vcp_p=3,
                    fe_p=3,
                    vcp_names=vcp_names)

            if kc != 3:
                model2 = BinomialBayesMixedGLM.from_formula(
                    fml, vc_fml, dy, vcp_p=3, fe_p=3)
            else:
                model2 = BinomialBayesMixedGLM(
                    endog,
                    exog,
                    exog_vc,
                    ident,
                    vcp_p=3,
                    fe_p=3,
                    vcp_names=vcp_names)

            rslt2 = model2.fit_vb(verbose=False, scale_fe=True)

            out.write(["Maternal", "Paternal", "Complex", "Maternal+Paternal"][kc] + " imprinted genes:\n")
            if ks:
                out.write("Retaining one kid per sibship\n")
            out.write("%d people\n" % dy.Person.unique().size)
            out.write("%d mothers\n" % dy.MomID.unique().size)
            out.write("%d samples\n" % dy.Sample.unique().size)
            out.write("%d imprinting calls\n" % dy.shape[0])
            out.write("%d genes\n" % dy.Gene.unique().size)
            out.write("%d exons\n" % dy.Exon.unique().size)
            out.write(rslt2.summary().as_text() + "\n")

            # Standardized parameters
            ss = model2.exog.std(0)
            ss[0] = 1
            sp = pd.DataFrame({"Slopes": rslt2.fe_mean * ss,
                              "SE": rslt2.fe_sd * ss})
            sp.index = model2.exog_names
            sp = sp.loc[:, ["Slopes", "SE"]]
            out.write("\nStandardized parameters\n")
            out.write(sp.to_string(float_format="%5.3f") + "\n\n")

            if kc > 2 or kp == False or ks == False:
                # Don't need predicted gene effects for these models
                continue

            xx = model2.data.frame
            xx = xx[["Gene", "GeneClass_c1_lnc2_nc3"]]
            xx = xx.set_index("Gene")
            xx = xx.to_dict()["GeneClass_c1_lnc2_nc3"]

            rr = rslt2.random_effects("Gene")
            rr = rr.sort_values(by="SD")
            rr["N"] = [np.sum(dx.Gene == x[8:-1]) for x in rr.index]
            rr["Crude"] = [
                dx.loc[dx.Gene == x[8:-1], "Imprinted"].mean()
                for x in rr.index
            ]

            rr.index = [x[8:-1] for x in rr.index]
            pa = pd.Series(rslt2.fe_mean, index=model2.fep_names).to_dict()
            rr["GeneClass"] = [xx.get(k) for k in rr.index]
            rr.GeneClass = rr.GeneClass.astype(np.int)
            rr["GeneClassEffect"] = 0
            for k in (1, 2, 3):
                a = "C(GeneClass_c1_lnc2_nc3)[T.%2.1f]" % k
                if a in pa:
                    ii = rr.GeneClass == k
                    rr.GeneClassEffect[ii] = pa[a]

            rr["TotalEffect"] = rr.Mean + rr.GeneClassEffect
            rr = rr.sort_values(by="TotalEffect")
            rr.to_csv("posterior_genes_%d_%d.csv" % (kc, method))

            sape = {}
            for i in range(dx.shape[0]):
                x = dx.iloc[i, :]
                sape[x.Sample] = [x.Person, x.MomID]

            sa = rslt2.random_effects("Sample")
            pr = rslt2.random_effects("Person")
            sa = sa.reset_index()
            sa["index"] = sa["index"].apply(lambda x: x[10:-1])
            sa = sa.rename(columns={
                "index": "Sample",
                "Mean": "Sample_mean",
                "SD": "Sample_SD"
            })
            sa["Person"] = [
                sape[sa.iloc[i, :].Sample][0] for i in range(sa.shape[0])
            ]

            sa["MomID"] = [
                sape[sa.iloc[i, :].Sample][1] for i in range(sa.shape[0])
            ]
            sa["MomID"] = sa.MomID.apply(lambda x: str(x))

            pr = pr.reset_index()
            pr["index"] = pr["index"].apply(lambda x: np.int64(x[10:-1]))
            pr = pr.rename(columns={
                "index": "Person",
                "Mean": "Person_mean",
                "SD": "Person_SD"
            })
            pf = pd.merge(pr, sa, left_on="Person", right_on="Person")

            pf = pf[[
                "Person", "Sample", "MomID", "Person_mean", "Person_SD",
                "Sample_mean", "Sample_SD"
            ]]
            pf["Total"] = pf.Person_mean + pf.Sample_mean

            vn = [
                "Person", "BirthLength_cen", "RIN", "X_chrM_TPMsum"
            ]
            if kp:
                vn.append("PlacentaWeight_cen")
            dxx = dy[vn]
            dxx = dxx.groupby("Person").head(1)
            dxx = dxx.reset_index()
            pf = pd.merge(
                pf, dxx, left_on="Person", right_on="Person", how="left")
            pf.to_csv(
                "person_sample_%d_%d.csv" % (kc, method),
                index=None,
                float_format="%.5f")

out.write("```\n")
log.write("```\n")

out.close()
log.close()
info.close()
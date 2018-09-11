import pandas as pd
import os, sys, re
import numpy as np

# ASE file name
fname = "ASE_totalRNA_hetSNP_V2targets_500bpPadding_annotated.txt.gz"

# Path to data
dpr = "/nfs/kshedden/Beverly_Strassmann"

# For an exon to be considered imprinted, at least half of the SNPs
# must have at most this percentage of reads from one parent.
pct = 10

# 0=gene/sample masking, 1=sample masking
method = int(sys.argv[1])

# Imprinting code variable
code = "Comb_Imp_Code_PlacPolGene"

df = pd.read_csv(os.path.join(dpr, fname), delimiter="\t")

df = df.rename(columns={"%chrM_TPMsum": "PctchrM_TPMsum"})

print("Initial data set size:")
print(df.shape)

df = df.loc[df.totalCount >= 10, :]
print("Size after requiring at least 10 reads:")
print(df.shape)

df = df.loc[df[code] <= 2, :]
print("Size after requiring imprinting code <= 2:")
print(df.shape)

if method == 0:
    dd = df.DiscardIf1_method4.astype(str)
    ii = ~(dd.str.contains("1") | dd.str.contains("nan"))
    df = df.loc[ii, :]
else:
    df = df.loc[df.Removed==0, :]

print("Size after masking for maternal contamination:")
print(df.shape)

df["icode"] = df[code]
df["exon"] = df["mergedExons_PolishGeneSelection"] + df.strand

da = []
for run,df1 in df.groupby("RNAid"):
    for exon,df2 in df1.groupby("exon"):

        person = df2["DNAid"].unique()
        if len(person) > 1:
            1/0
        person = person[0]

        sample = df2["Sample_source"].unique()
        if len(sample) > 1:
            1/0
        sample = sample[0]

        x = np.asarray(df2.refCount)
        y = np.asarray(df2.altCount)
        z = x + y
        t = np.where(x > y, y, x)
        imp = 1*(np.mean(t <= z*pct/100.0) >= 0.5)

        # MOM estimate of ASE
        ase = np.mean(x * y) / np.mean(x + y)**2
        if ase > 0.25:
            ase = 0.5
        else:
            ase = (1 - np.sqrt(1 - 4*ase)) / 2
        ase = 0.5 - ase

        # Concatenate the gene ids for each SNP
        gene = df2["Gene_Symbol"].iloc[0]

        # Prepare an output record
        anaf = df2.aveNonAltFreq.iloc[0]
        icr = df2.ICR.iloc[0]
        pweight = df2["Placenta_Weight"].iloc[0]
        blength = df2["Birth_length"].iloc[0]
        bweight = df2["Birth_Weight"].iloc[0]
        rin = df2["RIN"].iloc[0]
        batch = df2["Submission_date"].iloc[0].replace("-", "_")
        pctchrM_TPMsum = df2["PctchrM_TPMsum"].iloc[0]
        GeneClass_c1_lnc2_nc3 = df2["GeneClass_c1_lnc2_nc3"].iloc[0]

        xc = df2["icode"].value_counts()
        if xc.size != 1:
            1/0
        xc = xc.index[0]

        # The longitudinal study ID
        idf2 = df2.ID_F2.iloc[0]

        da.append([person, idf2, sample, run, exon, gene, xc, imp, len(z),
                   sum(z), df2.Sex.iloc[0], df2.Kid_Rank.iloc[0],
                   anaf, df2.cDNA_Library_type.iloc[0],
                   df2.MalariaInfectionElBashirCriteria.iloc[0],
                   df2.ID_mere.iloc[0], pweight, blength, bweight,
                   rin, icr, ase, batch, pctchrM_TPMsum,
                   GeneClass_c1_lnc2_nc3])

da = pd.DataFrame(da)
da.columns = ["Person", "ID_F2", "Sample", "Run", "Exon", "Gene", "Icode", "Imprinted",
              "Nsnp", "Reads", "Sex", "KidRank", "AvgNonAltFreq", "Lib", "Malaria", "MomID",
              "PlacentaWeight", "BirthLength", "BirthWeight", "RIN", "ICR", "ASE", "Batch",
              "PctchrM_TPMsum", "GeneClass_c1_lnc2_nc3"]

#
# Merge in some additional mother information
#

dm = pd.read_csv("F2_OneTime_Shedden.txt.gz", delimiter="\t")
dm = dm.loc[:, ["ID_mere", "ID_F2", "DOB_F1", "DateNaissance"]]
da = pd.merge(da, dm, left_on="ID_F2", right_on="ID_F2", how="left")
da = da.rename(columns={"ID_mere": "ID_F1"})
for x in "DateNaissance", "DOB_F1":
    da[x] = pd.to_datetime(da[x])
da["MomAge"] = da.DateNaissance - da.DOB_F1
da.MomAge = da.MomAge.dt.days.astype(np.float64)/365.25

#
# Merge in some longitudinal study variables
#

cf = pd.read_csv("/nfs/kshedden/Beverly_Strassmann/Cohort_2018.csv.gz")
# Nothing here yet

# Save the file
da.to_csv("imprint_full_%dpct_%d.csv" % (pct, method), index=None)


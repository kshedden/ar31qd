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

# Imprinting code variable name
code = "Comb_Imp_Code_PlacPolGene"

# Gene symbol variable name, can contain multiple genes joined with semi-colon
gene_symbol = "Gene_Symbol_PlacPolGene"

# Require gene type to be 1, 2, 3
gene_type = "GeneClass_c1_lnc2_nc3"

# Exon symbol variable name, can contain multiple names joined with semi-colon
exon_symbol = "mergedExons_PolishGeneSelection"

df = pd.read_csv(os.path.join(dpr, fname), delimiter="\t")

# Exclude SNPs mapping to multiple genes
df = df.loc[pd.notnull(df[gene_symbol]), :]
df = df.loc[~df[gene_symbol].str.contains("[,;]"), :]

df = df.rename(columns={"X.chrM_TPMsum": "X_chrM_TPMsum"})
df[gene_type] = pd.to_numeric(df[gene_type], errors='coerce')

def nmsg(df):
    print("  %d data points" % df.shape[0])
    print("  %d people" % df.DNAid.unique().size)
    print("  %d samples" % df.RNAid.unique().size)

print("Initial data set size:")
nmsg(df)

df = df.loc[df.totalCount >= 10, :]
print("Size after requiring at least 10 reads:")
nmsg(df)

df = df.loc[df[code] <= 2, :]
print("Size after requiring imprinting code <= 2:")
nmsg(df)

df = df.loc[df[gene_type].isin([1, 2, 3]), :]
print("Size after requiring gene_type in 1, 2, 3:")
nmsg(df)

if method == 0:
    dd = df.DiscardIf1_method4.astype(str)
    ii = ~(dd.str.contains("1", regex=False) | dd.str.contains("nan", regex=False) | dd.str.contains("NA", regex=False))
    df = df.loc[ii, :]
else:
    df = df.loc[df.Removed==0, :]

print("Size after masking for maternal contamination:")
nmsg(df)

df["icode"] = df[code]

# Add the strand identifier to the exon name
df["exon"] = df[exon_symbol] + df.strand

# Drop of exon name is missing
df = df.loc[pd.notnull(df.exon), :]

# Get all the distinct exons from the combined exon names
exons_unique = set([])
for ex in df.exon:
    if pd.isnull(ex):
        continue
    for u in ex.split(":"):
        exons_unique.add(u)
exons_unique = list(exons_unique)

da = []
for run,df1 in df.groupby("RNAid"):
    for exon in exons_unique:

        # Get the data for one sample, one exon
        df2 = df1.loc[df1.exon.str.contains(exon, regex=False), :]
        if df2.shape[0] == 0:
            continue

        # Get the person ID for this RNAid value.  There should only be one
        # person per RNAid.
        person = df2["DNAid"].unique()
        if len(person) > 1:
            1/0
        person = person[0]

        # Get the sample source for this RNAid value.  There should only be one
        # sample source per RNAid.
        sample = df2["Sample_source"].unique()
        if len(sample) > 1:
            1/0
        sample = sample[0]

        # Calculate the imprinting status.
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

        genes = df2[gene_symbol].dropna().unique()
        if len(genes) > 1 | any([":" in y for y in genes]):
            1/0
        gene = genes[0]

        # Prepare an output record
        anaf = df2.aveNonAltFreq.iloc[0]
        icr = df2.ICR.iloc[0]
        pweight = df2["Placenta_Weight"].iloc[0]
        blength = df2["Birth_length"].iloc[0]
        bweight = df2["Birth_Weight"].iloc[0]
        rin = df2["RIN"].iloc[0]
        batch = df2["Submission_date"].iloc[0].replace("-", "_")
        X_chrM_TPMsum = df2["X_chrM_TPMsum"].iloc[0]
        GeneClass_c1_lnc2_nc3 = df2["GeneClass_c1_lnc2_nc3"].iloc[0]

        if len(df2.ID_mere.unique()) > 1:
            1/0
        momid = df2.ID_mere.iloc[0]

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
                   momid, pweight, blength, bweight,
                   rin, icr, ase, batch, X_chrM_TPMsum,
                   GeneClass_c1_lnc2_nc3])

da = pd.DataFrame(da)
da.columns = ["Person", "ID_F2", "Sample", "Run", "Exon", "Gene", "Icode", "Imprinted",
              "Nsnp", "Reads", "Sex", "KidRank", "AvgNonAltFreq", "Lib", "Malaria", "MomID",
              "PlacentaWeight", "BirthLength", "BirthWeight", "RIN", "ICR", "ASE", "Batch",
              "X_chrM_TPMsum", "GeneClass_c1_lnc2_nc3"]

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

# Save the file
da.to_csv("imprint_full_%dpct_%d.csv" % (pct, method), index=None)

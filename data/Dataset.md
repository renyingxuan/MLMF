* Dataset is available at [TACG](https://portal.gdc.cancer.gov/). 
* This paper conducts experiments on 10 cancer data sets of AML, BIC, COAD, GBM, KIRC, LIHC, LUSC, OV, SKCM and SARC of TCGA, and each data set includes
mRNA expression, DNA methylation and miRNA expression data.
* Perpare data files. The format is as follows:
```
data/
├── LSCC/
│   ├── partial/
|   |     ├──LUNG_Gene_0.1.txt
│   |     ├──LUNG_Gene_0.3.txt
|   |     ├──LUNG_Methy_0.1.txt
|   |     ...  
│   ├── LUNG_Gene_Expression.txt
│   ├── LUNG_Methy_Expression.txt
│   └── LUNG_Mirna_Expression.txt
├── ...
```
# MLMF：Multi-layer matrix factorization for cancer subtyping using full and partial multi-omics dataset


## Introduction

Cancer, with its inherent heterogeneity, is commonly categorized into distinct subtypes based on unique traits, cellular origins, and molecular markers specific to each type. However, current studies primarily rely on complete multi-omics datasets for predicting cancer subtypes, often overlooking predictive performance in cases where some omics data may be missing and neglecting implicit relationships across multiple layers of omics data integration. This paper introduces Multi-Layer Matrix Factorization (MLMF), a novel approach for cancer subtyping that employs multi-omics data clustering. MLMF initially processes multi-omics feature matrices by performing multi-layer linear or nonlinear factorization, decomposing the original data into latent feature representations unique to each omics type. These latent representations are subsequently fused into a consensus form, on which spectral clustering is performed to determine subtypes. Additionally, MLMF incorporates a class indicator matrix to handle missing omics data, creating a unified framework that can manage both complete and incomplete multi-omics data. Extensive experiments conducted on 10 multi-omics cancer datasets, both complete and with missing values, demonstrate that MLMF achieves results that are comparable to or surpass the performance of several state-of-the-art approaches.


## Dataset
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
## Running
* This paper contains the linear algorithm part and the nonlinear algorithm part of MLMF, which you can run separately
* For Linear MLMF:
```
#run Linear_MLMF.m
#the costs are saved in ./MDMF_linear
```
* For Nonlinear MLMF:
```
#run Nonlinear_MLMF.m
#the costs are saved in ./MLMF_nonlinear
```


## Comparison
the program of the comparison methods:
```
./Comparison
```
MLMF is based on the MATLAB and R language.

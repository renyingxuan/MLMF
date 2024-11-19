# MLMF：Multi-layer matrix factorization for cancer subtyping using full and partial multi-omics dataset

Contact: Yingxuan Ren, Bo Yang
Email: renyx@connect.hku.hk 

---

## Introduction

Cancer, with its inherent heterogeneity, is commonly categorized into distinct subtypes based on unique traits, cellular origins, and molecular markers specific to each type. However, current studies primarily rely on complete multi-omics datasets for predicting cancer subtypes, often overlooking predictive performance in cases where some omics data may be missing and neglecting implicit relationships across multiple layers of omics data integration. This paper introduces Multi-Layer Matrix Factorization (MLMF), a novel approach for cancer subtyping that employs multi-omics data clustering. MLMF initially processes multi-omics feature matrices by performing multi-layer linear or nonlinear factorization, decomposing the original data into latent feature representations unique to each omics type. These latent representations are subsequently fused into a consensus form, on which spectral clustering is performed to determine subtypes. Additionally, MLMF incorporates a class indicator matrix to handle missing omics data, creating a unified framework that can manage both complete and incomplete multi-omics data. Extensive experiments conducted on 10 multi-omics cancer datasets, both complete and with missing values, demonstrate that MLMF achieves results that are comparable to or surpass the performance of several state-of-the-art approaches.

----
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
|   |       
├── LSCC/
│   ├── module1/
│   ├── module2/
│   └── utils/
```
### Option 2. Docker Dockerfile

This is the same as option 1 except that you are building a docker image yourself. Please refer to option 1 for usage. 

```bash
# clone the repo
git clone https://github.com/zhengzhenxian/Repun.git
cd Repun

# build a docker image named hkubal/repun:latest
# might require docker authentication to build docker image 
docker build -f ./Dockerfile -t hkubal/repun:latest .

# run the docker image like option 1
docker run -it hkubal/repun:latest /opt/bin/repun --help
```


Check [Usage](#Usage) for more options.

----

## Usage

### General Usage

```bash
./repun \
  --bam_fn ${INPUT_DIR}/sample.bam \       ## use your bam file name here
  --ref_fn ${INPUT_DIR}/ref.fa \           ## use your reference file name here
  --truth_vcf_fn ${INPUT_DIR}/truth.vcf \  ## use your truth VCF file name here
  --threads ${THREADS} \                   ## maximum threads to be used
  --platform ${PLATFORM} \                 ## options: {ont, hifi, ilmn}
  --output_dir ${OUTPUT_DIR}               ## output path prefix 

## Final output file: ${OUTPUT_DIR}/unified.vcf.gz
```

### Options

**Required parameters:**

```bash

  -b BAM_FN, --bam_fn BAM_FN
                        BAM file input. The input file must be samtools indexed.
  -r REF_FN, --ref_fn REF_FN
                        FASTA reference file input. The input file must be samtools indexed.
  --truth_vcf_fn TRUTH_VCF_FN
                        Truth VCF file input.
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory.
  -t THREADS, --threads THREADS
                        Max threads to be used.
  -p PLATFORM, --platform PLATFORM
                        Select the sequencing platform of the input. Possible options: {ont, hifi,
                        ilmn}.

```

**Miscellaneous parameters:**

```bash

  -c CTG_NAME, --ctg_name CTG_NAME
                        The name of the contigs to be processed. Split by ',' for multiple contigs.
                        Default: all contigs will be processed.
  --bed_fn BED_FN       Path to a BED file. Execute Repun only in the provided BED regions.
  --region REGION       A region to be processed. Format: `ctg_name:start-end` (start is 1-based).
  --min_af MIN_AF       Minimal AF required for a variant to be called. Default: 0.08.
  --min_coverage MIN_COVERAGE
                        Minimal coverage required for a variant to be called. Default: 4.
  -s SAMPLE_NAME, --sample_name SAMPLE_NAME
                        Define the sample name to be shown in the VCF file. Default: SAMPLE.
  --output_prefix OUTPUT_PREFIX
                        Prefix for output VCF filename. Default: output.
  --remove_intermediate_dir
                        Remove the intermediate directory before finishing to save disk space.
  --include_all_ctgs    Execute Repun on all contigs, otherwise call in chr{1..22,X,Y} and {1..22,X,Y}.
  -d, --dry_run         Print the commands that will be run.
  --python PYTHON       Absolute path of python, python3 >= 3.9 is required.
  --pypy PYPY           Absolute path of pypy3, pypy3 >= 3.6 is required.
  --samtools SAMTOOLS   Absolute path of samtools, samtools version >= 1.10 is required.
  --whatshap WHATSHAP   Absolute path of whatshap, whatshap >= 1.0 is required.
  --parallel PARALLEL   Absolute path of parallel, parallel >= 20191122 is required.
  --disable_phasing     Disable phasing with whatshap.

```

----

## Disclaimer

NOTE: the content of this research code repository (i) is not intended to be a medical device; and (ii) is not intended for clinical use of any kind, including but not limited to diagnosis or prognosis.

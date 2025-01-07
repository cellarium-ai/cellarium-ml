library("biomaRt")
library("readxl")
exit <- function() { invokeRestart("abort") }
print("sucessfully running R script")
inputs <- commandArgs(trailingOnly = TRUE)
inputpath <- inputs[1]
outputpath <- inputs[2]
#inputpath <- "/home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/pseudobulk/ensembl_ids.xlsx"
#outputpath <- "/home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/pseudobulk/ensembl_hgcn_ids.tsv"
ensembsIDS <- read_excel(inputpath)
ensembl_list <- ensembsIDS$ensembl

ensembl <- useMart(biomart="ENSEMBL_MART_ENSEMBL", host="https://useast.ensembl.org",dataset="hsapiens_gene_ensembl") #useMart(biomart="ENSEMBL_MART_ENSEMBL", host="https://useast.ensembl.org")

results <- getBM(attributes=c('hgnc_symbol','uniprotswissprot','uniprotsptrembl','ensembl_gene_id'),
      filters = 'ensembl_gene_id',
      values = c(ensembl_list),
      uniqueRows = TRUE,
      mart = ensembl)
print("Sucessfully retrieved Biomart information")

results <- sapply(results, as.character)

write.table(results,file=outputpath,row.names=FALSE, sep="\t")

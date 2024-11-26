#!/bin/bash
#
#list1=($(seq 0 10 4442)) #start step stop
#list2=($(seq 10 10 4500))

#gcloud storage du gs://cellarium-human-primary-data/curriculum/human_all_primary_20241108/extract_files --summarize
#gsutil du --recursive gs://cellarium-human-primary-data/curriculum/human_all_primary_20241108/extract_files | wc -l

list1=($(seq 150 20 220)) #start step stop
list2=($(seq 170 20 230))

for index in "${!list1[@]}"; do
  i="${list1[$index]}"
  j="${list2[$index]}"
  name=$(eval gs://cellarium-human-primary-data/curriculum/human_all_primary_20241108/extract_files/extract_{$i..$j}.h5ad)
  echo $name
  eval "printf '%s\n' gs://cellarium-human-primary-data/curriculum/human_all_primary_20241108/extract_files/extract_{$i..$j}.h5ad" > paths.txt;
  xargs -a paths.txt -I {} gsutil cp {} /home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/extracts;
  eval "printf '%s\n' /home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/extracts/extract_{$i..$j}.h5ad" > paths_upload.txt;
  echo "STARTED UPLOAD TO ATLAS"
  xargs -a paths_upload.txt -I {} scp {} lys@atlas.icmm-joshi-group.lan.ku.dk:/storage/Lys/cellxgene;
  echo "Removing current chunk of files"
  xargs -a paths_upload.txt rm
done


list1=($(seq 3450 20 3520)) #start step stop
list2=($(seq 3470 20 3540))

for index in "${!list1[@]}"; do
  i="${list1[$index]}"
  j="${list2[$index]}"
  name=$(eval gs://cellarium-human-primary-data/curriculum/human_all_primary_20241108/extract_files/extract_{$i..$j}.h5ad)
  echo $name
  eval "printf '%s\n' gs://cellarium-human-primary-data/curriculum/human_all_primary_20241108/extract_files/extract_{$i..$j}.h5ad" > paths.txt;
  xargs -a paths.txt -I {} gsutil cp {} /home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/extracts;
  eval "printf '%s\n' /home/lys/Dropbox/PostDoc_Glycomics/cellarium-ml/data/extracts/extract_{$i..$j}.h5ad" > paths_upload.txt;
  echo "STARTED UPLOAD TO ATLAS"
  xargs -a paths_upload.txt -I {} scp {} lys@atlas.icmm-joshi-group.lan.ku.dk:/storage/Lys/cellxgene;
  echo "Removing current chunk of files"
  xargs -a paths_upload.txt rm
done

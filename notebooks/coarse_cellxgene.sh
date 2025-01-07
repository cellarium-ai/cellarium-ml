#!/bin/bash
input=$1
output=$2
#MAX_SIZE=$((2,7 * 1024 * 1024 * 1024)) #5GB
#MAX_SIZE=$(echo "2.7 * 1024 * 1024 * 1024" | bc )
#MAX_SIZE=2899102924
MAX_SIZE=2799102924
echo $MAX_SIZE


for filename in $(ls $input/*.h5ad | sort -V); do
    filesize=$(stat --format="%s" "$filename")
    if [ "$filesize" -le "$MAX_SIZE" ]; then
      echo $filename
      python coarse_cellxgene.py --input "$filename" --output "$output" --optiona;
      #true
    else
      echo "Skipping $filename (size: $((filesize / 1024 / 1024 / 1024)) GB)"
    fi
done

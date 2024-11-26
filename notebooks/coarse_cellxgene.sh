#!/bin/bash
directory=$1
for filename in $directory/*.h5ad; do
    python coarse_cellxgene.py --storage "$filename";
done

#/bin/bash

# python metadata_prediction_analysis.py --output_path ./test/10M --checkpoint_path /home/mehrtash/data/mb_checkpoints/10M_001_bs1536/epoch=1-step=29161__updated.ckpt --n_cells 10000
# python metadata_prediction_analysis.py --output_path ./test/19M --checkpoint_path /home/mehrtash/data/mb_checkpoints/19M_001_bs2048/epoch=1-step=28244__updated.ckpt --n_cells 10000
# python metadata_prediction_analysis.py --output_path ./test/30M --checkpoint_path /home/mehrtash/data/mb_checkpoints/30M_001_bs2560/epoch=2-step=43129__updated.ckpt --n_cells 10000
python metadata_prediction_analysis.py --output_path ./test/59M --checkpoint_path /home/mehrtash/data/mb_checkpoints/59M_001_bs3072/epoch=3-step=53770__updated.ckpt --n_cells 10000
python metadata_prediction_analysis.py --output_path ./test/100M --n_cells 10000

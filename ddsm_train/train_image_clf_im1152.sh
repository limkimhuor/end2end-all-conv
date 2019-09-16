#!/bin/bash

TRAIN_DIR="/Volumes/Transcend/gen/proceed_images/832x1152_train"
VAL_DIR="/Volumes/Transcend/gen/proceed_images/832x1152_valid"
TEST_DIR="/Volumes/Transcend/gen/proceed_images/832x1152_test"
PATCH_STATE="Combined_patches_im1152_224_s10/s10_YaroslavNet.h5"
BEST_MODEL="Combined_full_images/ddsm_YaroslavNet_s10.h5"
FINAL_MODEL="NOSAVE"

export NUM_CPU_CORES=2

# 255/65535 = 0.003891.
python3 image_clf_train.py \
	--patch-model-state $PATCH_STATE \
	--no-resume-from \
    --img-size 1152 832 \
    --no-img-scale \
    --rescale-factor 0.003891 \
	--featurewise-center \
    --featurewise-mean 52.18 \
    --no-equalize-hist \
    --top-depths 512 512 \
    --top-repetitions 3 3 \
    --batch-size 20 \
    --train-bs-multiplier 0.5 \
	--augmentation \
	--class-list bi_rads0 bi_rads1 bi_rads2 bi_rads3 bi_rads4 bi_rads5 \
	--nb-epoch 1 \
    --all-layer-epochs 2 \
    --no-load-val-ram \
    --no-load-train-ram \
    --optimizer adam \
    --weight-decay 0.0001 \
    --weight-decay2 0.0001 \
    --hidden-dropout 0.0 \
    --hidden-dropout2 0.0 \
    --init-learningrate 0.001 \
    --all-layer-multiplier 0.1 \
	--lr-patience 2 \
	--es-patience 10 \
	--auto-batch-balance \
    --pos-cls-weight 1.0 \
	--neg-cls-weight 1.0 \
	--best-model $BEST_MODEL \
	--final-model $FINAL_MODEL \
    --patch-net yaroslav \
	$TRAIN_DIR $VAL_DIR $TEST_DIR	

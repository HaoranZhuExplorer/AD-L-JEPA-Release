#!/bin/bash

# Define the list of experiment IDs and corresponding epoch IDs
declare -a exp_ids=("ad_l_jepa_exp_13_4" "ad_l_jepa_exp_13_5" "ad_l_jepa_exp_13_6" "ad_l_jepa_exp_13_7")
declare -a epoch_ids=(6 30 30 30)

# Loop through each index of the arrays
for i in "${!exp_ids[@]}"; do
    # Get the current experiment ID and epoch ID from arrays
    exp_id="${exp_ids[i]}"
    epoch_id="${epoch_ids[i]}"

    # Run the command with the current experiment ID and epoch ID substituted in
    bash ./scripts/dist_evaluate_knn.sh 4 \
         --cfg_file cfgs/kitti_models/second.yaml \
         --batch_size 4 \
         --pretrained_model "../output/kitti_models/ad_l_jepa_kitti/${exp_id}/ckpt/checkpoint_epoch_${epoch_id}.pth" \
         --extra_tag "knn_${exp_id}"
done

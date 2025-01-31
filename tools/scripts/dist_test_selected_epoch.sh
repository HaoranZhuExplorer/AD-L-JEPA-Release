#!/usr/bin/env bash

# Define the arrays of run_id and epoch values
run_ids=("second_ad_l_jepa_exp_14_2_13_run_1_best_in_3_runs")
epochs=("5" "10" "15" "20" "25" "30" "35" "40" "45" "50" "55" "60" "65" "70" "75" "80")

#run_ids=("second_ad_l_jepa_exp_14_2_new_epoch_30_run_3")
#epochs=("80")

# Loop through each run_id
for run_id in "${run_ids[@]}"; do
    # Loop through each epoch
    for epoch in "${epochs[@]}"; do
        # Execute the command with current run_id and epoch
        bash ./scripts/dist_test.sh 4 \
            --cfg_file cfgs/kitti_models/second.yaml \
            --batch_size 32 \
            --ckpt ../output/kitti_models/second/${run_id}/ckpt/checkpoint_epoch_${epoch}.pth \
            --extra_tag ${run_id}
    done
done
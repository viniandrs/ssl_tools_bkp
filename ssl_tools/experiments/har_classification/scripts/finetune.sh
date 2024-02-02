#/bin/bash

cd ..

#for dset in  "KuHar" "MotionSense" "RealWorld_thigh" "RealWorld_waist" "UCI";
for pretrain_dset in "KuHar" "MotionSense" "RealWorld_thigh" "RealWorld_waist" "UCI";
do
    for finetune_dset in  "KuHar" "MotionSense" "UCI" "RealWorld_thigh" "RealWorld_waist" "UCI";
    do
        
        ./tnc.py fit \
            --data ./../../../data/standartized_balanced/KuHar \
            --epochs 100 \
            --batch_size 128 \
            --accelerator gpu \
            --devices 1 \
            --load_backbone logs/pretrain/TNC/KuHar/checkpoints/last.ckpt \
            --training_mode finetune \
            --repeat 5 \
            --mc_sample_size 20 \
            --window_size 60 \
            --encoding_size 10 \
            --w 0.05 \
            --update_backbone False

        # renaming results folder
        new_result_folder_name=${pretrain_dset}_${finetune_dset}
        mv $(find logs/finetune/TNC -name "2024*" -type d) logs/finetune/TNC/$new_result_folder_name

        # updating symbolic link
        cd logs/finetune/TNC/$new_result_folder_name/checkpoints
        results_file_name=$(find -name "epoch*" -type "f" -printf "%f\n")
        ln -vsf $results_file_name last.ckpt
        
    done
done
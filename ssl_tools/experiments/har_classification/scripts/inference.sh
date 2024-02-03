#/bin/bash

cd ..

#for dset in  "KuHar" "MotionSense" "RealWorld_thigh" "RealWorld_waist" "UCI";
for pretrain_dset in "KuHar" "MotionSense" "RealWorld_thigh" "RealWorld_waist" "UCI";
do
    for finetune_dset in  "KuHar" "MotionSense" "UCI" "RealWorld_thigh" "RealWorld_waist" "UCI";
    do
        for test_dset in  "KuHar" "MotionSense" "UCI" "RealWorld_thigh" "RealWorld_waist" "UCI";
        do 
            ./tnc.py test \
                --data ./../../../data/standartized_balanced/${test_dset} \
                --load logs/finetune/TNC/${pretrain_dset}_${finetune_dset}/checkpoints/last.ckpt \
                --batch_size 128 \
                --accelerator gpu \
                --devices 1 \
                --mc_sample_size 20 \
                --window_size 60 \
                --encoding_size 10 \
                --w 0.05

            # renaming results folder
            mv $(find logs/test/TNC -name "2024*" -type d) logs/test/TNC/${test_set}
        done
    done
done
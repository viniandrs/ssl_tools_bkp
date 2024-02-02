#/bin/bash

cd ..

for dset in  "KuHar" "MotionSense" "UCI"; 
do 
    ./tnc.py fit \
        --data ./../../../data/view_concatenated/${dset} \
        --epochs 100 \
        --batch_size 16 \
        --accelerator gpu \
        --devices 1 \
        --training_mode pretrain \
        --checkpoint_metric train_loss \
        --repeat 5 \
        --mc_sample_size 20 \
        --window_size 60 \
        --encoding_size 10 \
        --w 0.05 \

        # renaming results folder
        mv $(find logs/pretrain/TNC -name "2024*" -type d) logs/pretrain/TNC/${dset}

        # updating symbolic link
        cd logs/pretrain/TNC/${dset}/checkpoints
        results_file_name=$(find -name "epoch*" -type "f" -printf "%f\n")
        ln -vsf $results_file_name last.ckpt
done


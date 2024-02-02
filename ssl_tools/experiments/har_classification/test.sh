# renaming results folder
new_result_folder_name=uci_kuhar

# updating symbolic link
cd logs/finetune/TNC/$new_result_folder_name/checkpoints
results_file_name=$(find -name "epoch*" -type "f" -printf "%f\n")
ln -vsf $results_file_name test.ckpt
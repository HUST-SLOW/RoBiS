original_data_path="./data/mvtec_ad_2" # the original mvtec_ad_2 dataset 
data_path="./mvtec_ad_2_processed" # the pre-processed dataset
save_dir="./saved_weights" # weights
amap_savedir="./anomaly_map_results" # the output path of the anomaly maps
bin_savedir="./binary_map_results" # the output path of the binary results

python ./examples/robis_main.py --device 0 \
--dataset_name MVTec-AD-2 --original_data_path "$original_data_path"  --data_path "$data_path" --mvtecad2_class_list all --test_type test_public \
--window_size 1024 --desired_overlap 0.1 \
--save_dir "$save_dir" --amap_savedir "$amap_savedir" --bin_savedir "$bin_savedir" \
--encoder dinov2reg_vit_base_14 --input_size 518 --crop_size 518 --INP_num 6 \
--total_epochs 200 --batch_size 16 --train_state True \
--test_state True --eval_pub True
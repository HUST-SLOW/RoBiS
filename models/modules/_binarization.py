import os
import cv2
import sys
sys.path.append(os.getcwd())
from models.modules._MEBin import MEBin
    
def _mvtec_bin(amap_path, bin_path, classname_list, test_type='test_public'):
    '''
    Binarize anomaly maps for the MVTec AD 2 dataset and save the results.
    This function processes each class in the MVTec AD 2 dataset, binarizes the anomaly maps using the MEBin algorithm,
    and saves the binarized maps to the specified output path.

    Args:
        amap_path(str): The directory that saves the anomaly prediction maps.
        bin_path(str): The output binary maps path.
        classname_list(list): Categories in the MVTec AD 2 dataset.
        test_type(str): The test subdataset you choose.
    '''
    anomaly_map_dir = amap_path
    binarization_dir = bin_path
    os.makedirs(amap_path, exist_ok=True)
    os.makedirs(bin_path, exist_ok=True)
    if classname_list == ['all']:
        classname_list = ['sheet_metal', 'vial', 'wallplugs', 'walnuts', 'can', 'fabric', 'fruit_jelly', 'rice']
    else:
        classname_list = classname_list

    for class_name in classname_list:
        print(f'Binarizing {class_name}...')
        class_output_path = os.path.join(binarization_dir, class_name, test_type)
        os.makedirs(class_output_path, exist_ok=True)
        
        # Collect anomaly map paths
        anomaly_map_path_list = []
        anomaly_num_list = []
        anomaly_types = sorted(os.listdir(os.path.join(anomaly_map_dir, class_name, test_type)))
        
        if test_type == 'test_public':
            for anomaly_type in anomaly_types:
                anomaly_type_anomaly_map_paths = sorted(os.listdir(os.path.join(anomaly_map_dir, class_name, test_type, anomaly_type)))
                anomaly_map_path_list.extend([os.path.join(anomaly_map_dir, class_name, test_type, anomaly_type, path) for path in anomaly_type_anomaly_map_paths])
                anomaly_num = len(anomaly_type_anomaly_map_paths)
                anomaly_num_list.append(anomaly_num)
        else:
            anomaly_types = ['unknown']
            anomaly_type_anomaly_map_paths = sorted(os.listdir(os.path.join(anomaly_map_dir, class_name, test_type)))
            anomaly_map_path_list.extend([os.path.join(anomaly_map_dir, class_name, test_type, path) for path in anomaly_type_anomaly_map_paths])
            anomaly_num = len(anomaly_type_anomaly_map_paths)
            anomaly_num_list.append(anomaly_num)
        
        # instantiate the binarization method
        bin = MEBin(anomaly_map_path_list, class_name=class_name)
        
        # Use the selected binarization method to binarize the anomaly maps
        binarized_maps, threshold_list = bin.binarize_anomaly_maps()
        
        # Save the binarization result
        start = 0
        class_threshold_dict = {}
        for i, anomaly_type in enumerate(anomaly_types):
            if test_type == 'test_public':
                anomaly_type_out_path = os.path.join(class_output_path, anomaly_type)
            else:
                anomaly_type_out_path = os.path.join(class_output_path)
            os.makedirs(anomaly_type_out_path, exist_ok=True)
            end = start + anomaly_num_list[i]
            anomaly_type_binarized_maps = binarized_maps[start:end]
            anomaly_type_thresholds = threshold_list[start:end]
            
            # Iterate over the binarized maps and thresholds for the current anomaly type
            class_threshold_dict[anomaly_type] = {}
            for j, threshold in enumerate(anomaly_type_thresholds):
                map_name = os.path.basename(anomaly_map_path_list[start + j])
                class_threshold_dict[anomaly_type][map_name] = threshold
            
            # Save the binarized maps for the current anomaly type
            for j, binarized_map in enumerate(anomaly_type_binarized_maps):
                map_path = os.path.join(anomaly_type_out_path, os.path.basename(anomaly_map_path_list[start + j]))
                map_path = map_path.replace('.tiff', '.png')
                cv2.imwrite(map_path, binarized_map)
            start = end

def mvtec_bin(cfg: dict):
    _mvtec_bin(
        amap_path=cfg['save']['amap_savedir'],
        bin_path=cfg['save']['bin_savedir'],
        classname_list=cfg['dataset']['mvtecad2_class_list'],
        test_type=cfg['dataset']['test_type']
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--amap_savedir",  required=True)
    parser.add_argument("--bin_savedir",  required=True)
    parser.add_argument("--mvtecad2_class_list", nargs="+", default=["all"])
    parser.add_argument("--test_type", default="test_public")
    args = parser.parse_args()

    cfg = {
        "save": {
            "amap_savedir": args.amap_savedir,
            "bin_savedir":  args.bin_savedir
        },
        "dataset": {
            "mvtecad2_class_list": args.mvtecad2_class_list,
            "test_type":           args.test_type
        }
    }
    mvtec_bin(cfg)
import os
import numpy as np
import math
import tifffile as tiff
from tqdm import tqdm

#Original sizes of all categories 
resolution_dict = {'sheet_metal':(4224, 1056), 'vial':(1400, 1900), 'wallplugs':(2448, 2048), 'walnuts':(2448, 2048),
                    'can':(2232, 1024), 'fabric':(2448, 2048), 'fruit_jelly':(2100, 1520), 'rice':(2448, 2048)}

def reconstruct_image(crop_path, ct, window_size, desired_overlap):
    """
    Reconstruct the image from its sliding-window crops.
    The reconstructed image is saved as {prefix}.tiff in the same folder,
    and the individual crop images are deleted.

    Args:
        crop_path (str): Folder containing the .tiff crop images.
        ct (str): Category key used to lookup original width and height in resolution_dict.
        window_size (int): Side length of the square cropping window.
        desired_overlap (float): Overlap ratio between consecutive windows(0 ≤ desired_overlap < 1).
    """
    step_size = int(window_size * (1 - desired_overlap))
    crop_files = sorted([f for f in os.listdir(crop_path) if f.endswith(".tiff")])

    original_width, original_height = resolution_dict[ct]
    original_width = int(original_width/4)
    original_height = int(original_height/4)
    
    grouped_files = {}
    for crop_file in crop_files:
        prefix = "_".join(crop_file.split("_")[:-1])
        if prefix not in grouped_files:
            grouped_files[prefix] = []
        grouped_files[prefix].append(crop_file)

    for prefix, files in tqdm(grouped_files.items()):
        padded_height = math.ceil(original_height / window_size) * window_size
        padded_width = math.ceil(original_width / window_size) * window_size

        reconstructed_image = np.zeros((padded_height, padded_width), dtype=np.float16)
        weight_matrix = np.zeros((padded_height, padded_width), dtype=np.float16)

        for crop_file in files:
            parts1 = crop_file.split(".")
            parts = parts1[0].split("_")
            index = [int(a) for a in str(parts[-1])]
            row_idx = int(index[0])
            col_idx = int(index[1])
            y1 = row_idx * step_size
            x1 = col_idx * step_size
            y2 = y1 + window_size
            x2 = x1 + window_size

            crop_img = tiff.imread(os.path.join(crop_path, crop_file))
            h, w = crop_img.shape

            if y2 > padded_height:
                h = padded_height - y1
                y2 = padded_height
            if x2 > padded_width:
                w = padded_width - x1
                x2 = padded_width
                crop_img = crop_img[:h, :w]

            reconstructed_image[y1:y2, x1:x2] += crop_img[:h, :w]
            weight_matrix[y1:y2, x1:x2] += 1
            os.remove(os.path.join(crop_path, crop_file))

        weight_matrix[weight_matrix == 0] = 1
        reconstructed_image = (reconstructed_image / weight_matrix[:, :]).astype(np.float16)
        reconstructed_image = reconstructed_image[:original_height, :original_width]
        save_path = os.path.join(crop_path, f"{prefix}.tiff")
        tiff.imwrite(save_path, reconstructed_image)

def _merge(crop_path, classname_list, window_size=1024, desired_overlap=0.1, test_type_filter=None):
    """
    Reconstruct full images for all categories and splits you given.
    If test_type_filter is given, only that split is processed.

    Args:
        crop_path (str): Folder containing the .tiff crop images.
        classname_list (list): List of categories to process.
        window_size (int): Side length of the square cropping window.
        desired_overlap (float): Overlap ratio between consecutive windows.
        test_type_filter (str): Split test type to restrict processing.
    """
    if classname_list == ['all']:
        classname_list = ['sheet_metal', 'vial', 'wallplugs', 'walnuts', 'can', 'fabric', 'fruit_jelly', 'rice']
    else:
        classname_list = classname_list
    window_size = int(window_size/4)
    for classname in classname_list:
        print(f"{classname} processing...")
        for test_type in os.listdir(os.path.join(crop_path,classname)):
            if test_type_filter is not None and test_type != test_type_filter:
                continue
            crop_path_1 = os.path.join(crop_path,classname,test_type)
            if test_type !='test_public':
                reconstruct_image(crop_path_1, classname, window_size, desired_overlap)
            else:
                for label in os.listdir(crop_path_1):
                    crop_path_2 = os.path.join(crop_path_1,label)
                    reconstruct_image(crop_path_2, classname, window_size, desired_overlap)
        print(f"{classname} finished.")

def merge(cfg: dict):
    _merge(
        crop_path      = cfg['save']['amap_savedir'],
        classname_list = cfg['dataset']['mvtecad2_class_list'],
        window_size    = cfg['crop']['window_size'],
        desired_overlap= cfg['crop']['desired_overlap'],
        test_type_filter= cfg['dataset']['test_type']
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--amap_savedir", required=True)
    parser.add_argument("--mvtecad2_class_list", nargs="+", default=["all"])
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--desired_overlap", type=float, default=0.1)
    parser.add_argument("--test_type", default="test_public")
    args = parser.parse_args()

    cfg = {
        "save": {
            "amap_savedir": args.amap_savedir
        },
        "dataset": {
            "mvtecad2_class_list": args.mvtecad2_class_list,
            "test_type": args.test_type
        },
        "crop": {
            "window_size": args.window_size,
            "desired_overlap": args.desired_overlap
        }
    }
    merge(cfg)
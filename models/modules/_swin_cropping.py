import os
import cv2
import numpy as np
import math

def generate_sliding_window_images(img_path, img_crop_path, window_size, desired_overlap):
    """
    Generate overlapping sliding-window crops from a single image.

    Args:
        img_path (str): Path to the source image.
        img_crop_path (str): Folder where the window crops will be written.
        window_size (int): Side length of the square window.
        desired_overlap (float): Overlap ratio between successive windows.
    """
    window_size = window_size
    desired_overlap = desired_overlap
    step_size = int(window_size * (1 - desired_overlap))
    
    img_name = os.path.basename(img_path).split(".")[0]
    img = cv2.imread(img_path)
    
    height, width, _ = img.shape
    
    num_steps_y = int(np.ceil((height - window_size) / step_size)) + 1
    num_steps_x = int(np.ceil((width - window_size) / step_size)) + 1
    
    y_steps = [i * step_size for i in range(num_steps_y)]
    x_steps = [i * step_size for i in range(num_steps_x)]
    
    count_y = 0
    for y in y_steps:
        count_x = 0
        for x in x_steps:
            window_x1 = x
            window_y1 = y
            window_x2 = x + window_size
            window_y2 = y + window_size
            window = img[window_y1:window_y2, window_x1:window_x2]
            
            if window.shape[0] < window_size or window.shape[1] < window_size:
                padded_window = np.zeros((window_size, window_size, 3), dtype=np.uint8)
                padded_window[:window.shape[0], :window.shape[1]] = window
                window = padded_window

            file_path = os.path.join(img_crop_path, f"{img_name}_{count_y}{count_x}.png")
            cv2.imwrite(file_path, window)
            count_x += 1
        count_y += 1

def process_images(img_files, img_dir, crop_dir, window_size, desired_overlap):
    """
    Generate sliding-window crops for a list of images.

    Args:
        img_files (list): Filenames of the images to process.
        img_dir (str): Directory containing the original images.
        crop_dir (str): Output directory for the window crops.
        window_size (int): Side length of the square window.
        desired_overlap (float): Overlap ratio between successive windows.
    """
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    for img_file in img_files:
        img_name = img_file.split(".")[0]
        img_path = os.path.join(img_dir, img_file)
        generate_sliding_window_images(img_path, crop_dir, window_size, desired_overlap)

def _crop(path, classname_list, crop_path, window_size, desired_overlap):
    """
    Generate sliding-window crops for an entire dataset.

    Args:
        path (str): Root directory of the original dataset.
        classname_list (list): Categories to process.
        crop_path (str): Root directory where crops will be stored.
        window_size (int): Side length of the square window.
        desired_overlap (float): Overlap ratio between successive windows.
    """
    if classname_list == ['all']:
        classname_list = ['sheet_metal', 'vial', 'wallplugs', 'walnuts', 'can', 'fabric', 'fruit_jelly', 'rice']
    else:
        classname_list = classname_list
    for ct in classname_list:
        print(f"{ct} processing...")
        if not os.path.isdir(os.path.join(path, ct)):
            continue
        ct_path = os.path.join(path, ct)
        cp_path = os.path.join(crop_path, ct)
        for category in os.listdir(ct_path):
            category_path = os.path.join(ct_path, category)
            crop_path_1 = os.path.join(cp_path, category)
            if os.path.isdir(crop_path_1) and os.listdir(crop_path_1):
                continue
            if category in ['test_private', 'test_private_mixed']:
                img_files = os.listdir(category_path)
                process_images(img_files, category_path, crop_path_1, window_size, desired_overlap)
            else:
                for label in os.listdir(category_path):
                    label_path = os.path.join(category_path, label)
                    crop_path_2 = os.path.join(crop_path_1, label)

                    if label == 'ground_truth':
                        for gt in os.listdir(label_path):
                            gt_path = os.path.join(label_path, gt)
                            crop_path_3 = os.path.join(crop_path_2, gt)
                            img_files = os.listdir(gt_path)
                            process_images(img_files, gt_path, crop_path_3, window_size, desired_overlap)
                    else:
                        img_files = os.listdir(label_path)
                        process_images(img_files, label_path, crop_path_2, window_size, desired_overlap)
        print(f"{ct} finished.")

def crop(cfg: dict):
    _crop(
        path=cfg['dataset']['original_data_path'],
        classname_list=cfg['dataset']['mvtecad2_class_list'],
        crop_path=cfg['dataset']['data_path'],
        window_size=cfg['crop']['window_size'],
        desired_overlap=cfg['crop']['desired_overlap']
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_data_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--desired_overlap", type=float, required=True)
    parser.add_argument("--mvtecad2_class_list", nargs="+", required=True)
    args = parser.parse_args()

    cfg = {
        "dataset": {
            "original_data_path": args.original_data_path,
            "data_path": args.data_path,
            "mvtecad2_class_list": args.mvtecad2_class_list
        },
        "crop": {
            "window_size": args.window_size,
            "desired_overlap": args.desired_overlap
        }
    }
    crop(cfg)
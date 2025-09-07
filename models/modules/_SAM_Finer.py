import os
import torch
import numpy as np
from tqdm import tqdm
import cv2
from scipy.ndimage import zoom
import sys
sys.path.append(os.getcwd())
from third_party_library.segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from utils.utils import setup_seed

def get_topn_bounding_boxes(binary_image, top_n=3):
    """
    Extract the top-N largest connected-component bounding boxes from a binary mask.
    Uses cv2 to label all white regions, sorts them by pixel area,
    and returns the axis-aligned bounding boxes of the largest N regions.

    Args:
        binary_image (np.ndarray): Single-channel binary mask.
        top_n (int): Maximum number of boxes to return.
    Returns:
        bounding_boxes[list]: Bounding boxes of the top-N largest components.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        top_indices = np.argsort(areas)[-top_n:] + 1
    else:
        return []
    bounding_boxes = []
    for idx in top_indices:
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        area = stats[idx, cv2.CC_STAT_AREA]
        category_id = 1
        is_crowd = 0
        bounding_boxes.append([x, y, x + w, y + h])
    return bounding_boxes

def fill_holes(image):
    """
    Fill all holes inside any foreground region of a binary mask.

    Args:
        image (np.ndarray): Single-channel binary mask.
    Returns:
        filled_image(np.ndarray): Binary mask with holes filled.
    """
    num_labels, labels = cv2.connectedComponents(image)
    mask = np.zeros_like(image)
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    filled_image = cv2.bitwise_or(image, mask)
    return filled_image

def _samfiner(path, classname_list, bin_path, test_type, sam_device):
    """
    Refine coarse binary masks with Segment-Anything (SAM).
    A lightweight ViT-B SAM is used for test_private_mixed,
    and ViT-H is used otherwise.
    Fabric and walnuts enable multimask output; rice is skipped.

    Args:
        path (str): Root directory of the original images.
        classname_list (list): Categories to process.
        bin_path (str): Directory containing coarse binary masks (where the refined masks will be saved).
        test_type (str): Dataset split to process.
        sam_device (int): CUDA device id.
    """
    if classname_list == ['all']:
        classname_list = ['sheet_metal', 'vial', 'wallplugs', 'walnuts', 'can', 'fabric', 'fruit_jelly', 'rice']
    else:
        classname_list = classname_list
    device = torch.device("cuda:{}".format(sam_device) if torch.cuda.is_available() else "cpu")

    if test_type == 'test_private_mixed':
        sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    else:
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam.to(device))
    dataset_dir = path
    save_dir = bin_path
    for class_name in classname_list:
        print(class_name)
        if class_name in ['rice']:
            continue
        elif class_name in ['fabric', 'walnuts']:
            tag = True
        else:
            tag = False
        setup_seed(1)
        dataset_class_dir = os.path.join(dataset_dir, class_name, test_type)
        image_path_list = []
        if test_type == 'test_public':
            image_name_list = os.listdir(os.path.join(dataset_class_dir, 'bad'))
            image_path_list.extend([os.path.join(dataset_class_dir, 'bad', x) for x in image_name_list])
            image_name_list = os.listdir(os.path.join(dataset_class_dir, 'good'))
            image_path_list.extend([os.path.join(dataset_class_dir, 'good', x) for x in image_name_list])
        elif test_type == 'validation':
            image_name_list = os.listdir(os.path.join(dataset_class_dir, 'good'))
            image_path_list.extend([os.path.join(dataset_class_dir, 'good', x) for x in image_name_list])
        else:
            image_name_list = os.listdir(dataset_class_dir)
            image_path_list.extend([os.path.join(dataset_class_dir, x) for x in image_name_list])
        image_path_list = sorted(image_path_list)
        for image_path in tqdm(image_path_list):
            image = cv2.imread(image_path)
            binary_mask_path = image_path.replace(dataset_dir, save_dir)
            parts = binary_mask_path.split(os.sep)          
            if test_type == 'validation':
                new_parts = parts[:-2] + parts[-1:]
                binary_mask_path = os.sep.join(new_parts)
            binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
            zoom_factors = (binary_mask.shape[0] / image.shape[0], binary_mask.shape[1] / image.shape[1], 1)
            image = zoom(image, zoom_factors, order=1)
            top_n = 3
            bbox_list = get_topn_bounding_boxes(binary_mask, top_n=top_n)
            if len(bbox_list) == 0:
                masks = np.zeros_like(binary_mask)
            else:
                predictor.set_image(image)
                mask_list = []
                for bbox in bbox_list:
                    box_prompt = np.array(bbox)
                    masks, _, _ = predictor.predict(box=box_prompt, return_logits=False, multimask_output=tag)
                    masks = masks.sum(0)
                    masks[masks > 0] = 255
                    masks = masks.astype(np.uint8)
                    mask_list.append(masks)
                masks = np.array(mask_list).max(0)
                masks = fill_holes(masks)
            save_path = image_path.replace(dataset_dir, save_dir)
            part = save_path.split(os.sep)          
            if test_type == 'validation':
                new_part = part[:-2] + part[-1:]
                save_path = os.sep.join(new_part)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, masks)

def samfiner(cfg: dict):
    _samfiner(
        path=cfg['dataset']['original_data_path'],
        classname_list=cfg['dataset']['mvtecad2_class_list'],
        bin_path=cfg['save']['bin_savedir'],
        test_type=cfg['dataset']['test_type'],
        sam_device=cfg['device']
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_data_path", required=True)
    parser.add_argument("--mvtecad2_class_list", nargs="+", default=["all"])
    parser.add_argument("--test_type", default="test_public")
    parser.add_argument("--bin_savedir", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    cfg = {
        "dataset": {
            "original_data_path": args.original_data_path,
            "mvtecad2_class_list": args.mvtecad2_class_list,
            "test_type": args.test_type
        },
        "save": {
            "bin_savedir": args.bin_savedir
        },
        "device": args.device
    }
    samfiner(cfg)

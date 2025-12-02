import numpy as np
import cv2
import os
from tqdm import tqdm
import itertools
import argparse

def compute_iou(mask1, mask2):
    """
    Compute the Intersection over Union (IoU) of two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0  # Avoid division by zero

    return intersection / union

def remove_overlapping_masks(mask_path, level, out_path, image_path, iou_threshold=0.5):
    
    try:
        mask_list = [os.path.join(mask_path, file) for file in sorted(os.listdir(mask_path)) if file.endswith('.npy')]
    except FileNotFoundError:
        if level == "small":
            print(f"Skipping small level as mask path does not exist.")
            os.makedirs(out_path, exist_ok=True)
        else:
            print(f"Mask path {mask_path} does not exist.")
        return

    # find overlapping masks, and remove the smaller one
    pairs_to_merge = []
    for i in tqdm(range(len(mask_list[::10])), desc="Finding overlapping masks"):
        mask_for_overlap_detection = np.load(mask_list[i])
        for (i, mask1), (j, mask2) in itertools.combinations(enumerate(mask_for_overlap_detection), 2):
            iou = compute_iou(mask1, mask2)
            if iou > iou_threshold:
                smaller_mask_index = i if mask1.sum() < mask2.sum() else j
                larger_mask_index = i if mask1.sum() >= mask2.sum() else j
                # print(f"Found overlapping masks: indices ({i}, {j}), smaller mask index: {smaller_mask_index}")
                pairs_to_merge.append((smaller_mask_index, larger_mask_index))

    # remove duplicates
    pairs_to_merge = list(set(pairs_to_merge))
    # if (i, j) is in pairs_to_merge, then (j, i) should not be in pairs_to_merge
    for i, j in pairs_to_merge:
        if (j, i) in pairs_to_merge:
            pairs_to_merge.remove((j, i))
    def is_index_in_position0(index, tuple_list):
        return any(index == tup[0] for tup in tuple_list)

    def find_position0(index, tuple_list):
        for tup in tuple_list:
            if index == tup[1]:  # Check if index is in position 1
                return True, tup[0]  # Return True and corresponding position 0
        return False, None  # Return False if not found

    image_names = sorted(os.listdir(image_path))
    for idx, mask_path in tqdm(enumerate(mask_list), total=len(mask_list)):
        masks = np.load(mask_path)
        obj_count = 0
        for i in range(masks.shape[0]):
            if is_index_in_position0(i, pairs_to_merge):
                continue
            obj_id = i
            mask = masks[i]

            # switch channel
            mask = mask.transpose(1, 2, 0)
            if not os.path.exists(os.path.join(out_path, f"{obj_count:03d}")):
                os.makedirs(os.path.join(out_path, f"{obj_count:03d}"))
            cv2.imwrite(os.path.join(out_path, f"{obj_count:03d}", image_names[idx]), mask*255)
            obj_count += 1
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_root", type=str, required=True, help="Root directory of the masks.")
    parser.add_argument("--out_root", type=str, required=True, help="Output root directory.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the images.")
    args = parser.parse_args()

    mask_path = os.path.join(args.mask_root, "large", "final-output")
    out_path = os.path.join(args.out_root, "multiview_masks_default")
    remove_overlapping_masks(mask_path, level="large", out_path=out_path, image_path=args.image_path)
    mask_path = os.path.join(args.mask_root, "middle", "final-output")
    out_path = os.path.join(args.out_root, "multiview_masks_middle")
    remove_overlapping_masks(mask_path, level="middle", out_path=out_path, image_path=args.image_path)
    mask_path = os.path.join(args.mask_root, "small", "final-output")
    out_path = os.path.join(args.out_root, "multiview_masks_small")
    remove_overlapping_masks(mask_path, level="small", out_path=out_path, image_path=args.image_path)
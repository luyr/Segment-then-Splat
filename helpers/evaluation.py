import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import argparse

device = "cuda"

def load_images_masks(scene_path):
    
    image_dir = os.path.join(scene_path, "images")
    default_mask_dir = os.path.join(scene_path, "multiview_masks_default_merged")
    train_txt = os.path.join(scene_path, "train.txt")
    with open(train_txt, "r") as f:
        train_list = f.readlines()
    train_list = [x.strip() for x in train_list]
    image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f in train_list]

    # first load large object images
    num_large_objects = len(os.listdir(default_mask_dir))
    large_object_images = [[] for _ in range(num_large_objects)]
    for object_idx, object_mask_dir in tqdm(enumerate(sorted(os.listdir(default_mask_dir))), total=num_large_objects, desc="Loading large object images"):
        object_mask_path = os.path.join(default_mask_dir, object_mask_dir)
        object_masks = [os.path.join(object_mask_path, f) for f in sorted(os.listdir(object_mask_path)) if f in train_list]
        for i in range(len(image_paths)):
            image = Image.open(image_paths[i])
            object_mask = Image.open(object_masks[i])
            
            masked_image = Image.composite(image, Image.new("RGBA", image.size, (0, 0, 0, 255)), object_mask)
            # find a bounding box around the mask and crop the image
            bbox = object_mask.getbbox()
            masked_image = masked_image.crop(bbox)
            large_object_images[object_idx].append(masked_image)
    # then load middle object images
    middle_mask_dir = os.path.join(scene_path, "multiview_masks_middle_merged")
    num_middle_objects = len(os.listdir(middle_mask_dir))
    middle_object_images = [[] for _ in range(num_middle_objects)]
    for object_idx, object_mask_dir in tqdm(enumerate(sorted(os.listdir(middle_mask_dir))), total=num_middle_objects, desc="Loading middle object images"):
        object_mask_path = os.path.join(middle_mask_dir, object_mask_dir)
        object_masks = [os.path.join(object_mask_path, f) for f in sorted(os.listdir(object_mask_path)) if f in train_list]
        for i in range(len(image_paths)):
            image = Image.open(image_paths[i])
            object_mask = Image.open(object_masks[i])    
            
            masked_image = Image.composite(image, Image.new("RGBA", image.size, (0, 0, 0, 255)), object_mask)
            # find a bounding box around the mask and crop the image
            bbox = object_mask.getbbox()
            masked_image = masked_image.crop(bbox)
            middle_object_images[object_idx].append(masked_image)
    # the load small object images
    small_mask_dir = os.path.join(scene_path, "multiview_masks_small_merged")
    num_small_objects = len(os.listdir(small_mask_dir))
    small_object_images = [[] for _ in range(num_small_objects)]
    for object_idx, object_mask_dir in tqdm(enumerate(sorted(os.listdir(small_mask_dir))), total=num_small_objects, desc="Loading small object images"):
        object_mask_path = os.path.join(small_mask_dir, object_mask_dir)
        object_masks = [os.path.join(object_mask_path, f) for f in sorted(os.listdir(object_mask_path)) if f in train_list]
        for i in range(len(image_paths)):
            image = Image.open(image_paths[i])
            object_mask = Image.open(object_masks[i])   
            
            masked_image = Image.composite(image, Image.new("RGBA", image.size, (0, 0, 0, 255)), object_mask)
            # find a bounding box around the mask and crop the image
            bbox = object_mask.getbbox()
            masked_image = masked_image.crop(bbox)
            small_object_images[object_idx].append(masked_image)

    return large_object_images, middle_object_images, small_object_images

def extract_CLIP_embeddings(model, preprocess, large_object_images, middle_object_images, small_object_images):
    # first calculate the features for the large objects
    large_object_features = []
    for object_idx in tqdm(range(len(large_object_images)), desc="Extracting large object features"):
        object_features = []
        for object_img in large_object_images[object_idx]:
            image = preprocess(object_img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
            object_features.append(image_features)
        large_object_features.append(torch.stack(object_features).mean(dim=0))
    mean_large_object_features = torch.stack(large_object_features)
    # then calculate the features for the middle objects
    middle_object_features = []
    for object_idx in tqdm(range(len(middle_object_images)), desc="Extracting middle object features"):
        object_features = []
        for object_img in middle_object_images[object_idx]:
            image = preprocess(object_img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
            object_features.append(image_features)
        middle_object_features.append(torch.stack(object_features).mean(dim=0))
    mean_middle_object_features = torch.stack(middle_object_features)
    # then calculate the features for the small objects
    small_object_features = []
    for object_idx in tqdm(range(len(small_object_images)), desc="Extracting small object features"):
        object_features = []
        for object_img in small_object_images[object_idx]:
            image = preprocess(object_img).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
            object_features.append(image_features)
        small_object_features.append(torch.stack(object_features).mean(dim=0))
    mean_small_object_features = torch.stack(small_object_features) if small_object_features else torch.empty(0)
    
    return mean_large_object_features, mean_middle_object_features, mean_small_object_features

def load_results(results_dir):

    transform = transforms.ToTensor()
    # load predicted large object images
    large_object_result_dirs = [os.path.join(results_dir, f, 'ours_40000/renders') for f in sorted(os.listdir(results_dir)) if f.startswith("test_default")]
    predicted_large_objects = []
    for object_result_dir in tqdm(large_object_result_dirs, desc="Loading rendered large objects"):
        object_images = [transform(Image.open(os.path.join(object_result_dir, f))) for f in sorted(os.listdir(object_result_dir))]
        predicted_large_objects.append(object_images)
    # load predicted middle object images
    middle_object_result_dirs = [os.path.join(results_dir, f, 'ours_40000/renders') for f in sorted(os.listdir(results_dir)) if f.startswith("test_middle")]
    predicted_middle_objects = []
    for object_result_dir in tqdm(middle_object_result_dirs, desc="Loading rendered middle objects"):
        # print(object_result_dir)
        object_images = [transform(Image.open(os.path.join(object_result_dir, f))) for f in sorted(os.listdir(object_result_dir))]
        predicted_middle_objects.append(object_images)
    # load predicted small object images
    small_object_result_dirs = [os.path.join(results_dir, f, 'ours_40000/renders') for f in sorted(os.listdir(results_dir)) if f.startswith("test_small")]
    predicted_small_objects = []
    for object_result_dir in tqdm(small_object_result_dirs, desc="Loading rendered small objects"):
        object_images = [transform(Image.open(os.path.join(object_result_dir, f))) for f in sorted(os.listdir(object_result_dir))]
        predicted_small_objects.append(object_images)
        
    return predicted_large_objects, predicted_middle_objects, predicted_small_objects

def calculate_IoU(label_path, mean_large_object_features, mean_middle_object_features, mean_small_object_features, predicted_large_objects, predicted_middle_objects, predicted_small_objects, model):
    gt_dirs = sorted(os.listdir(label_path))
    transform = transforms.ToTensor()
    all_ious = []
    for gt_idx, gt_dir in enumerate(gt_dirs):
        gt_names = sorted(os.listdir(os.path.join(label_path, gt_dir)))
        queries = [name.replace(".jpg", "") for name in gt_names]
        text_queries = clip.tokenize(queries).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_queries)
            
        gt_masks = [transform(Image.open(os.path.join(label_path, gt_dir, f))) for f in gt_names]
            
        # find the most similar object, use cosine similarity
        large_similarity = (mean_large_object_features @ text_features.T).squeeze(1)
        middle_similarity = (mean_middle_object_features @ text_features.T).squeeze(1)
        # small_similarity = (mean_small_object_features @ text_features.T).squeeze(1) if mean_small_object_features != torch.empty(0) else None
        
        topk_value_large, topk_indice_large = torch.topk(large_similarity, 1, largest=True, dim=0)
        # print(topk_value_large, topk_indice_large)
        topk_value_middle, topk_indice_middle = torch.topk(middle_similarity, 1, largest=True, dim=0)
        # print(topk_value_middle, topk_indice_middle)
        # if small_similarity is not None:
        #     topk_value_small, topk_indice_small = torch.topk(small_similarity, 1, largest=True, dim=0)
            # print(topk_value_small, topk_indice_small)
        
        ious = []
        large_object_ids = []
        middle_object_ids = []
        small_object_ids = []
        for j in range(len(queries)):
            if topk_value_large[..., j] - topk_value_middle[..., j] > 0:
                intersection = torch.logical_and(predicted_large_objects[topk_indice_large[...,j]][gt_idx], gt_masks[j]).sum()
                union = torch.logical_or(predicted_large_objects[topk_indice_large[...,j]][gt_idx], gt_masks[j]).sum()
            else:
                intersection = torch.logical_and(predicted_middle_objects[topk_indice_middle[...,j]][gt_idx], gt_masks[j]).sum()
                union = torch.logical_or(predicted_middle_objects[topk_indice_middle[...,j]][gt_idx], gt_masks[j]).sum()
        
            iou = intersection / union
            ious.append(iou)

        print(gt_dir)
        print("queries:", queries)
        print("mean IoU of current frame:", torch.tensor(ious).mean())
        all_ious.append(torch.tensor(ious).mean())
    print("final mIoU:", torch.tensor(all_ious).mean())

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, help="Scene path")
    parser.add_argument("--render_dir", type=str, required=True, help="Results directory path")
    parser.add_argument("--label_dir", type=str, required=True, help="Ground truth label directory path")
    args = parser.parse_args()
    
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    large_object_images, middle_object_images, small_object_images = load_images_masks(args.scene)
    print("==================== CLIP Embedding Association ====================")
    mean_large_object_features, mean_middle_object_features, mean_small_object_features = extract_CLIP_embeddings(
        model, preprocess, large_object_images, middle_object_images, small_object_images
    )
    
    predicted_large_objects, predicted_middle_objects, predicted_small_objects = load_results(args.render_dir)
    print("==================== mIoU Evaluation ====================")
    calculate_IoU(args.label_dir, mean_large_object_features, mean_middle_object_features, mean_small_object_features, \
        predicted_large_objects, predicted_middle_objects, predicted_small_objects, model)
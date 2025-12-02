import os
import cv2
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from plyfile import PlyData, PlyElement
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from nerfstudio.process_data.colmap_utils import parse_colmap_camera_params
import argparse

def load_colmap_info(colmap_dir):
    if os.path.exists(os.path.join(colmap_dir, "cameras.txt")):
        cam_id_to_camera = colmap_utils.read_cameras_text(os.path.join(colmap_dir, "cameras.txt"))
        image_infos = colmap_utils.read_images_text(os.path.join(colmap_dir, "images.txt"))
    elif os.path.exists(os.path.join(colmap_dir, "cameras.bin")):
        cam_id_to_camera = colmap_utils.read_cameras_binary(os.path.join(colmap_dir, "cameras.bin"))
        image_infos = colmap_utils.read_images_binary(os.path.join(colmap_dir, "images.bin"))
    else:
        raise ValueError(f"Could not find cameras.txt or cameras.bin in {colmap_dir}")

    if os.path.exists(os.path.join(colmap_dir, "points3D.bin")):
        colmap_points = colmap_utils.read_points3D_binary(os.path.join(colmap_dir, "points3D.bin"))
    elif os.path.exists(os.path.join(colmap_dir, "points3D.txt")):
        colmap_points = colmap_utils.read_points3D_text(os.path.join(colmap_dir, "points3D.txt"))
        
        
    sorted_image_infos = dict(sorted(image_infos.items(), key=lambda item: item[1].name))
    sorted_colmap_points = dict(sorted(colmap_points.items(), key=lambda item: item[1].id))
    
    return sorted_image_infos, sorted_colmap_points, cam_id_to_camera

def load_train_test_split(scene_root):
    train_txt = os.path.join(scene_root, "train.txt")
    with open(train_txt, "r") as f:
        train_list = f.readlines()
    train_list = [x.strip() for x in train_list]

    test_txt = os.path.join(scene_root, "test.txt")
    with open(test_txt, "r") as f:
        test_list = f.readlines()
    test_list = [x.strip() for x in test_list]
    
    return train_list, test_list
    
def load_multiview_masks(scene_root):
    # load large mask
    mask_dir = os.path.join(scene_root, "multiview_masks_large")
    object_dirs = sorted([os.path.join(mask_dir, d) for d in os.listdir(mask_dir)])
    all_masks_large = [] # num_objects x num_masks x H x W
    for object_dir in tqdm(object_dirs, desc="Loading large masks"):
        masks = []
        mask_list = sorted(os.listdir(object_dir))
        masks = [cv2.imread(os.path.join(object_dir, mask), cv2.IMREAD_GRAYSCALE) for mask in mask_list]
        all_masks_large.append(masks)

    # load middle mask
    mask_dir = os.path.join(scene_root, "multiview_masks_middle")
    object_dirs = sorted([os.path.join(mask_dir, d) for d in os.listdir(mask_dir)])
    all_masks_middle = [] # num_objects x num_masks x H x W
    for object_dir in tqdm(object_dirs, desc="Loading middle masks"):
        masks = []
        mask_list = sorted(os.listdir(object_dir))
        masks = [cv2.imread(os.path.join(object_dir, mask), cv2.IMREAD_GRAYSCALE) for mask in mask_list]
        all_masks_middle.append(masks)
        
    # load small mask
    mask_dir = os.path.join(scene_root, "multiview_masks_small")
    object_dirs = sorted([os.path.join(mask_dir, d) for d in os.listdir(mask_dir)])
    all_masks_small = [] # num_objects x num_masks x H x W
    for object_dir in tqdm(object_dirs, desc="Loading small masks"):
        masks = []
        mask_list = sorted(os.listdir(object_dir))
        masks = [cv2.imread(os.path.join(object_dir, mask), cv2.IMREAD_GRAYSCALE) for mask in mask_list]
        all_masks_small.append(masks)
        
    return all_masks_large, all_masks_middle, all_masks_small

def check_object_ids(sorted_colmap_points, sorted_image_infos, train_list, all_masks_large, all_masks_middle, all_masks_small):
    points_xyz = dict()
    points_rgb = dict()
    for point_id, point in sorted_colmap_points.items():
        points_xyz[point_id] = point.xyz
        points_rgb[point_id] = point.rgb

    # use the keys from points_xyz to create the large, middle, and small ids
    points_large_ids = dict()
    points_middle_ids = dict()
    points_small_ids = dict()
    for point_id in points_xyz.keys():
        points_large_ids[point_id] = 255
        points_middle_ids[point_id] = 255
        points_small_ids[point_id] = 255
        
    large_objects_pointIDs = defaultdict(list)
    middle_objects_pointIDs = defaultdict(list)
    small_objects_pointIDs = defaultdict(list)

    count = 0
    for img_idx, (image_id, _, _, _, name, xys, point3D_ids) in tqdm(enumerate(sorted_image_infos.values()), total=len(sorted_image_infos)):
        if name not in train_list:
            continue
        for xy_coord, point3D_id in zip(xys, point3D_ids):
            if point3D_id == -1:
                continue
            # for each 2D points, check if it is in the mask_large
            for obj_id, object_masks in enumerate(all_masks_large):
                if xy_coord[1] >= object_masks[count].shape[0] or xy_coord[0] >= object_masks[count].shape[1]:
                    continue
                if object_masks[count][int(xy_coord[1]), int(xy_coord[0])] == 255:
                    points_large_ids[point3D_id] = obj_id
                    large_objects_pointIDs[obj_id].append(point3D_id)
            # for each 2D points, check if it is in the mask_middle
            for obj_id, object_masks in enumerate(all_masks_middle):
                if xy_coord[1] >= object_masks[count].shape[0] or xy_coord[0] >= object_masks[count].shape[1]:
                    continue
                if object_masks[count][int(xy_coord[1]), int(xy_coord[0])] == 255:
                    points_middle_ids[point3D_id] = obj_id
                    middle_objects_pointIDs[obj_id].append(point3D_id)
            # for each 2D points, check if it is in the mask_small
            for obj_id, object_masks in enumerate(all_masks_small):
                if xy_coord[1] >= object_masks[count].shape[0] or xy_coord[0] >= object_masks[count].shape[1]:
                    continue
                if object_masks[count][int(xy_coord[1]), int(xy_coord[0])] == 255:
                    points_small_ids[point3D_id] = obj_id
                    small_objects_pointIDs[obj_id].append(point3D_id)
        count += 1
    
    # convert the points_xyz to a list
    points_xyz_list = []
    points_rgb_list = []
    points_large_ids_list = []
    points_middle_ids_list = []
    points_small_ids_list = []
    for point_id in points_xyz.keys():
        points_xyz_list.append(points_xyz[point_id])
        points_rgb_list.append(points_rgb[point_id])
        points_large_ids_list.append(points_large_ids[point_id])
        points_middle_ids_list.append(points_middle_ids[point_id])
        points_small_ids_list.append(points_small_ids[point_id])
    points_xyz_list = np.array(points_xyz_list)
    points_rgb_list = np.array(points_rgb_list)
    points_large_ids_list = np.array(points_large_ids_list)
    points_middle_ids_list = np.array(points_middle_ids_list)
    points_small_ids_list = np.array(points_small_ids_list)
    
    return points_xyz_list, points_rgb_list, points_large_ids_list, points_middle_ids_list, points_small_ids_list

def compensate_no_point_objs(points_xyz_list, points_rgb_list, points_large_ids_list, points_middle_ids_list, points_small_ids_list, all_masks_large, all_masks_middle, all_masks_small):
    # first handle large level
    unique_ids = np.array(range(len(all_masks_large)))
    for id in unique_ids:
        id_mask = points_large_ids_list == id
        object_points_xyz = points_xyz_list[id_mask]
        if object_points_xyz.shape[0] == 0:
            print(f"Large object {id} has no points, {object_points_xyz.shape[0]}")
            points_xyz_list = np.concatenate([points_xyz_list, np.zeros((1000, 3))], axis=0)
            points_rgb_list = np.concatenate([points_rgb_list, np.zeros((1000, 3))], axis=0)
            points_large_ids_list = np.concatenate([points_large_ids_list, id*np.ones((1000,))], axis=0)
            points_middle_ids_list = np.concatenate([points_middle_ids_list, np.zeros((1000,))+255], axis=0)
            points_small_ids_list = np.concatenate([points_small_ids_list, np.zeros((1000,))+255], axis=0)

    # then handle middle level
    unique_ids = np.array(range(len(all_masks_middle)))
    for id in unique_ids:
        id_mask = points_middle_ids_list == id
        object_points_xyz = points_xyz_list[id_mask]
        if object_points_xyz.shape[0] == 0:
            print(f"Middle object {id} has no points, {object_points_xyz.shape[0]}")
            points_xyz_list = np.concatenate([points_xyz_list, np.zeros((100, 3))], axis=0)
            points_rgb_list = np.concatenate([points_rgb_list, np.zeros((100, 3))], axis=0)
            points_large_ids_list = np.concatenate([points_large_ids_list, np.zeros((100,))+255], axis=0)
            points_middle_ids_list = np.concatenate([points_middle_ids_list, id*np.ones((100,))], axis=0)
            points_small_ids_list = np.concatenate([points_small_ids_list, np.zeros((100,))+255], axis=0)

    # then handle small level
    if len(all_masks_small) != 0:
        unique_ids = np.array(range(len(all_masks_small)))
        for id in unique_ids:
            id_mask = points_small_ids_list == id
            object_points_xyz = points_xyz_list[id_mask]
            if object_points_xyz.shape[0] == 0:
                print(f"Small object {id} has no points, {object_points_xyz.shape[0]}")
                points_xyz_list = np.concatenate([points_xyz_list, np.zeros((10, 3))], axis=0)
                points_rgb_list = np.concatenate([points_rgb_list, np.zeros((10, 3))], axis=0)
                points_large_ids_list = np.concatenate([points_large_ids_list, id*np.ones((10,))], axis=0)
                points_middle_ids_list = np.concatenate([points_middle_ids_list, id*np.ones((10,))], axis=0)
                points_small_ids_list = np.concatenate([points_small_ids_list, id*np.ones((10,))], axis=0)
            
    # add some points for the background
    background_points_xyz = np.zeros((1000, 3))
    background_points_rgb = np.zeros((1000, 3))
    background_points_large_ids = np.zeros((1000,))+255
    background_points_middle_ids = np.zeros((1000,))+255
    background_points_small_ids = np.zeros((1000,))+255
    points_xyz_list = np.concatenate([points_xyz_list, background_points_xyz], axis=0)
    points_rgb_list = np.concatenate([points_rgb_list, background_points_rgb], axis=0)
    points_large_ids_list = np.concatenate([points_large_ids_list, background_points_large_ids], axis=0)
    points_middle_ids_list = np.concatenate([points_middle_ids_list, background_points_middle_ids], axis=0)
    points_small_ids_list = np.concatenate([points_small_ids_list, background_points_small_ids], axis=0)
    
    return points_xyz_list, points_rgb_list, points_large_ids_list, points_middle_ids_list, points_small_ids_list

def find_similar_objects(points_xyz_list, points_rgb_list, points_large_ids_list, points_middle_ids_list, points_small_ids_list, all_masks_small):    
    # first handle large level
    unique_ids = np.unique(points_large_ids_list)
    object_xyz_means, object_rgb_means = [], []
    for id in unique_ids:
        id_mask = points_large_ids_list == id
        object_points_xyz = points_xyz_list[id_mask]
        object_xyz_means.append(np.mean(object_points_xyz, axis=0))
        object_points_rgb = points_rgb_list[id_mask]
        object_rgb_means.append(np.mean(object_points_rgb, axis=0))
    object_rgb_means = np.array(object_rgb_means)
    object_xyz_means = np.array(object_xyz_means)    

    # check paired xyz_distances between objects using numpy
    xyz_distances = cdist(object_xyz_means, object_xyz_means)
    # check paired rgb_distances between objects using numpy
    rgb_distances = cdist(object_rgb_means, object_rgb_means)/255.
    print("Mean XYZ distances:\n", xyz_distances.mean())
    print("Mean RGB distances:\n", rgb_distances.mean())
    # find xyz_distances < 0.6 and return index pairs
    idxs = np.where((xyz_distances + 20 * rgb_distances < 0.5) & (xyz_distances > 0.) & (rgb_distances > 0.))
    large_idx_pairs = list(zip(idxs[0], idxs[1]))
    # remove pairs with same index
    large_idx_pairs = [pair for pair in large_idx_pairs if pair[0] != pair[1]]
    # remove duplicates e.g. (1, 2) and (2, 1)
    large_idx_pairs = list(set([tuple(sorted(pair)) for pair in large_idx_pairs]))  
    if large_idx_pairs != []:
        print("Large level", large_idx_pairs)
        
    # then handle middle level
    unique_ids = np.unique(points_middle_ids_list)
    object_xyz_means, object_rgb_means = [], []
    for id in unique_ids:
        id_mask = points_middle_ids_list == id
        object_points_xyz = points_xyz_list[id_mask]
        object_xyz_means.append(np.mean(object_points_xyz, axis=0))
        object_points_rgb = points_rgb_list[id_mask]
        object_rgb_means.append(np.mean(object_points_rgb, axis=0))
    object_rgb_means = np.array(object_rgb_means)
    object_xyz_means = np.array(object_xyz_means)   

    xyz_distances = cdist(object_xyz_means, object_xyz_means)
    rgb_distances = cdist(object_rgb_means, object_rgb_means)/255.
    print("Mean XYZ distances:\n", xyz_distances.mean())
    print("Mean RGB distances:\n", rgb_distances.mean())
    idxs = np.where((xyz_distances + 20 * rgb_distances < 0.5) & (xyz_distances > 0.) & (rgb_distances > 0.))
    middle_idx_pairs = list(zip(idxs[0], idxs[1]))
    middle_idx_pairs = [pair for pair in middle_idx_pairs if pair[0] != pair[1]]
    middle_idx_pairs = list(set([tuple(sorted(pair)) for pair in middle_idx_pairs]))
    if middle_idx_pairs != []:
        print("Middle level", middle_idx_pairs)

    # then handle small level
    small_idx_pairs = []
    if len(all_masks_small) != 0:
        unique_ids = np.unique(points_small_ids_list)
        object_xyz_means, object_rgb_means = [], []
        for id in unique_ids:
            id_mask = points_middle_ids_list == id
            object_points_xyz = points_xyz_list[id_mask]
            object_xyz_means.append(np.mean(object_points_xyz, axis=0))
            object_points_rgb = points_rgb_list[id_mask]
            object_rgb_means.append(np.mean(object_points_rgb, axis=0))
        object_rgb_means = np.array(object_rgb_means)
        object_xyz_means = np.array(object_xyz_means)
        
        xyz_distances = cdist(object_xyz_means, object_xyz_means)
        rgb_distances = cdist(object_rgb_means, object_rgb_means)/255.
        print("Mean XYZ distances:\n", xyz_distances.mean())
        print("Mean RGB distances:\n", rgb_distances.mean())
        idxs = np.where((xyz_distances + 20 * rgb_distances < 0.5) & (xyz_distances > 0.) & (rgb_distances > 0.))
        small_idx_pairs = list(zip(idxs[0], idxs[1]))
        small_idx_pairs = [pair for pair in small_idx_pairs if pair[0] != pair[1]]
        small_idx_pairs = list(set([tuple(sorted(pair)) for pair in small_idx_pairs]))
        if small_idx_pairs != []:
            print("Small level", small_idx_pairs)
        
    return large_idx_pairs, middle_idx_pairs, small_idx_pairs

def merge_points(large_idx_pairs, middle_idx_pairs, small_idx_pairs):
    # merge 3D points based on the similar object ids
    # first handle large level
    for pair in large_idx_pairs:
        print(f"Large level: {pair}")
        id1, id2 = pair
        points_large_ids_list[points_large_ids_list == id2] = id1

    # then handle middle level
    for pair in middle_idx_pairs:
        print(f"Middle level: {pair}")
        id1, id2 = pair
        points_middle_ids_list[points_middle_ids_list == id2] = id1
        
    # then handle small level
    for pair in small_idx_pairs:
        print(f"Small level: {pair}")
        id1, id2 = pair
        points_small_ids_list[points_small_ids_list == id2] = id1
    
    return points_large_ids_list, points_middle_ids_list, points_small_ids_list

def merge_masks(large_idx_pairs, middle_idx_pairs, small_idx_pairs, all_masks_large, all_masks_middle, all_masks_small, mask_path):
    # merge masks by boolen OR
    # first handle default level
    for idx_pair in large_idx_pairs:
        for i in range(len(all_masks_large[idx_pair[0]])):
            all_masks_large[idx_pair[0]][i] = cv2.bitwise_or(all_masks_large[idx_pair[0]][i], all_masks_large[idx_pair[1]][i])
            
    all_masks_large = [all_masks_large[i] for i in range(len(all_masks_large)) if i not in [pair[1] for pair in large_idx_pairs]]

    # then handle middle level
    for idx_pair in middle_idx_pairs:
        for i in range(len(all_masks_middle[idx_pair[0]])):
            all_masks_middle[idx_pair[0]][i] = cv2.bitwise_or(all_masks_middle[idx_pair[0]][i], all_masks_middle[idx_pair[1]][i])
    all_masks_middle = [all_masks_middle[i] for i in range(len(all_masks_middle)) if i not in [pair[1] for pair in middle_idx_pairs]]

    # then handle small level
    for idx_pair in small_idx_pairs:
        for i in range(len(all_masks_small[idx_pair[0]])):
            all_masks_small[idx_pair[0]][i] = cv2.bitwise_or(all_masks_small[idx_pair[0]][i], all_masks_small[idx_pair[1]][i])
    all_masks_small = [all_masks_small[i] for i in range(len(all_masks_small)) if i not in [pair[1] for pair in small_idx_pairs]]
    
    # save all merged large masks
    image_names = sorted(os.listdir(os.path.join(mask_path, "images")))
    mask_dir_merged = os.path.join(mask_path, "multiview_masks_default_merged")
    os.makedirs(mask_dir_merged, exist_ok=True)
    for idx, object_masks in enumerate(all_masks_large):
        object_dir = os.path.join(mask_dir_merged, f"{idx:03d}")
        os.makedirs(object_dir, exist_ok=True)
        for i, mask in enumerate(object_masks):
            cv2.imwrite(os.path.join(object_dir, f"{image_names[i]}"), mask)
    # save all merged middle masks
    mask_dir_merged = os.path.join(mask_path, "multiview_masks_middle_merged")
    os.makedirs(mask_dir_merged, exist_ok=True)
    for idx, object_masks in enumerate(all_masks_middle):
        object_dir = os.path.join(mask_dir_merged, f"{idx:03d}")
        os.makedirs(object_dir, exist_ok=True)
        for i, mask in enumerate(object_masks):
            cv2.imwrite(os.path.join(object_dir, f"{image_names[i]}"), mask)
    # save all merged small masks
    mask_dir_merged = os.path.join(mask_path, "multiview_masks_small_merged")
    os.makedirs(mask_dir_merged, exist_ok=True)
    for idx, object_masks in enumerate(all_masks_small):
        object_dir = os.path.join(mask_dir_merged, f"{idx:03d}")
        os.makedirs(object_dir, exist_ok=True)
        for i, mask in enumerate(object_masks):
            cv2.imwrite(os.path.join(object_dir, f"{image_names[i]}"), mask)
            

def storePly(path, xyz, rgb, obj_id_large, obj_id_middle, obj_id_small):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('obj_id_default', 'u1'), ('obj_id_middle', 'u1'), ('obj_id_small', 'u1')]
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, obj_id_large, obj_id_middle, obj_id_small), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_root", type=str, required=True, help="Path to the scene root directory.")
    args = parser.parse_args()
    
    scene_root = args.scene_root
    colmap_dir = os.path.join(scene_root, "sparse/0/") 
    image_infos, colmap_points, cam_id_to_camera = load_colmap_info(colmap_dir)
    train_list, test_list = load_train_test_split(scene_root)
    print("Loaded COLMAP info and train/test split.")
    all_masks_large, all_masks_middle, all_masks_small = load_multiview_masks(scene_root)
    print("Loaded multiview masks.")
    points_xyz_list, points_rgb_list, points_large_ids_list, points_middle_ids_list, points_small_ids_list = \
        check_object_ids(colmap_points, image_infos, train_list, all_masks_large, all_masks_middle, all_masks_small)
    print("Initialized object ids for each 3D point.")
    points_xyz_list, points_rgb_list, points_large_ids_list, points_middle_ids_list, points_small_ids_list = \
        compensate_no_point_objs(points_xyz_list, points_rgb_list, points_large_ids_list, points_middle_ids_list, points_small_ids_list, all_masks_large, all_masks_middle, all_masks_small)
    print("Compensated objects with no associated 3D points.")
    large_idx_pairs, middle_idx_pairs, small_idx_pairs = \
        find_similar_objects(points_xyz_list, points_rgb_list, points_large_ids_list, points_middle_ids_list, points_small_ids_list, all_masks_small)
    print("Found similar objects (geometry + appearance) to merge.")
    points_large_ids_list, points_middle_ids_list, points_small_ids_list = \
        merge_points(large_idx_pairs, middle_idx_pairs, small_idx_pairs)
    merge_masks(large_idx_pairs, middle_idx_pairs, small_idx_pairs, all_masks_large, all_masks_middle, all_masks_small, scene_root)
    print("Merged similar objects in both 3D points and multiview masks.")
    storePly(os.path.join(scene_root, "sparse/0/points3D.ply"), \
        points_xyz_list, points_rgb_list, points_large_ids_list.reshape(-1,1), points_middle_ids_list.reshape(-1,1), points_small_ids_list.reshape(-1,1))
    print("Stored merged 3D points to PLY file.")

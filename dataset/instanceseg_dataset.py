'''
use uois3D(TRO2021) in point cloud instance segmentation - dataset
author: Haoxiang Ma
Date: 2021.04.24
'''

import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image

import torch
from torch._six import container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, \
    get_workspace_mask, remove_invisible_grasp_points


class InsSegDataset(Dataset):
    def __init__(self, root, camera='kinect', split='train', num_points=20000,
                 remove_outlier=False, remove_invisible=True, augment=False):
        assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.camera = camera
        self.augment = augment

        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for img_num in range(256):
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')


        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')


        return point_clouds

    def __getitem__(self, index):
        return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask

        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        if return_raw_cloud:
            return cloud_masked, color_masked

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]

        if self.augment:
            cloud_sampled = self.augment_data(cloud_sampled)

        # Compute object centers and directions
        offsets = np.zeros((len(seg_sampled), 3), dtype=np.float32)
        cf_3D_centers = np.zeros((100, 3), dtype=np.float32)  # 100 max object centers
        for i, k in enumerate(np.unique(seg_sampled)):
            mask = seg_sampled == k
            if k == 0:
                offsets[mask, ...] = 0
                continue
            # Compute 3D center
            center = np.average(cloud_sampled[mask],axis=0)
            cf_3D_centers[i-1] = center

            # Compute directions
            object_center_offsets = (center - cloud_sampled).astype(np.float32)
            offsets[mask, ...] = object_center_offsets[mask, ...]

        foreground_mask = (seg_sampled > 0).astype(np.int64)

        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['foreground_mask'] = foreground_mask
        ret_dict['instance_mask'] = seg_sampled
        ret_dict['cf_3D_centers'] = cf_3D_centers
        ret_dict['3D_offsets'] = offsets
        ret_dict['num_3D_centers'] = np.array(len(np.unique(seg_sampled)-1))
        return ret_dict


def collate_fn(batch):

    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]

    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


if __name__ == "__main__":
    import open3d as o3d
    root = "~\data\graspnet"
    train_dataset = InsSegDataset(root, split='train', remove_outlier=True,
                                    remove_invisible=True, num_points=20000,augment=True)
    print(len(train_dataset))
    ret = train_dataset[999]
    pcd = ret["point_clouds"]
    offset = ret["3D_offsets"]
    foremask = ret['foreground_mask']
    centers  = pcd+offset
    print(np.unique(centers,axis=0))
    scene = o3d.geometry.PointCloud()
    scene.points = o3d.utility.Vector3dVector(centers)
    o3d.io.write_point_cloud("center.ply", scene)
    for key in ret:
        print(key,ret[key].shape,ret[key].dtype)
    print(ret["num_3D_centers"])

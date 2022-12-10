
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
import open3d as o3d
from graspnetAPI.utils.utils import *
from graspnetAPI.utils.eval_utils import create_table_points, transform_points
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
parser.add_argument('--num_workers', type=int, default=50, help='Worker number[default: 50]')
cfgs = parser.parse_args()

def generate_scene_model(dataset_root, scene_name, anno_idx, return_poses=False, align=False, camera='realsense'):
    if align:
        camera_poses = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'camera_poses.npy'))
        camera_pose = camera_poses[anno_idx]
        align_mat = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'cam0_wrt_table.npy'))
        camera_pose = np.matmul(align_mat, camera_pose)
    scene_reader = xmlReader(
        os.path.join(dataset_root, 'scenes', scene_name, camera, 'annotations', '%04d.xml' % anno_idx))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    mat_list = []
    model_list = []
    pose_list = []
    for posevector in posevectors:
        obj_idx, pose = parse_posevector(posevector)
        obj_list.append(obj_idx)
        mat_list.append(pose)
    for obj_idx, pose in zip(obj_list, mat_list):
        plyfile = os.path.join(dataset_root, 'models', '%03d' % obj_idx, 'nontextured.ply')
        model = o3d.io.read_point_cloud(plyfile)
        points = np.array(model.points)
        if align:
            pose = np.dot(camera_pose, pose)
        points = transform_points(points, pose)
        model.points = o3d.utility.Vector3dVector(points)
        model_list.append(model)
        pose_list.append(pose)
    if return_poses:
        return model_list, obj_list, pose_list
    else:
        return model_list


class GraspNetDataset(Dataset):
    def __init__(self, root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=20000,
                 remove_outlier=False, remove_invisible=True, augment=False, load_label=True):
        assert (num_points <= 50000)
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}
        # self.step = 10

        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 190))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        elif split == 'all':
            self.sceneIds = list(range(0, 190))
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
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def save_data(self, index, return_raw_cloud=False):
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
        seg_mask = (seg > 0)
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
        cloud_masked, seg_masked = self.project_cad_to_camera_pcd(index=index,
                                                                  camera_pose=camera_poses[self.frameid[index]],
                                                                  align_mat=align_mat,
                                                                  scene_points=cloud_masked)
        scene_id = self.scenename[index]
        frame_id = self.frameid[index]
        father_path = os.path.join(self.root, 'clean_scenes', scene_id, self.camera, 'points')
        if not os.path.exists(father_path):
            os.makedirs(father_path)
            print(father_path)
        father_path = os.path.join(self.root, 'clean_scenes', scene_id, self.camera, 'seg')
        if not os.path.exists(father_path):
            os.makedirs(father_path)
        point_save_path = os.path.join(self.root, 'clean_scenes', scene_id, self.camera, 'points',
                                       str(frame_id).zfill(4) + '.npy')
        seg_save_path = os.path.join(self.root, 'clean_scenes', scene_id, self.camera, 'seg',
                                     str(frame_id).zfill(4) + '.npy')
        np.save(point_save_path, cloud_masked)
        np.save(seg_save_path, seg_masked)
        return

    def mix(self,pcd,pcd_seg,cpcd,cpcd_seg):
        object_idxs = np.unique(pcd_seg)
        mix_pcd = []
        mix_pcd_seg = []
        for i,object_id in enumerate(object_idxs):
            if np.random.random() > 0.5:
                mix_pcd.append(pcd[pcd_seg == object_id])
                mix_pcd_seg.append(pcd_seg[pcd_seg == object_id])
            else:
                mix_pcd.append(cpcd[cpcd_seg == object_id])
                mix_pcd_seg.append(cpcd_seg[cpcd_seg == object_id])
        mix_pcd = np.concatenate(mix_pcd)
        mix_pcd_seg = np.concatenate(mix_pcd_seg)
        return mix_pcd,mix_pcd_seg



    def create_table_points(self, lx, ly, lz, dx=0, dy=0, dz=0, grid_size=[0.01, 0.01, 0.01]):
        '''
        **Input:**
        - lx:
        - ly:
        - lz:
        **Output:**
        - numpy array of the points with shape (-1, 3).
        '''
        xmap = np.linspace(0, lx, int(lx / grid_size[0]))
        ymap = np.linspace(0, ly, int(ly / grid_size[1]))
        zmap = np.linspace(0, lz, int(lz / grid_size[2]))
        xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
        xmap += dx
        ymap += dy
        zmap += dz
        points = np.stack([xmap, ymap, zmap], axis=-1)
        points = points.reshape([-1, 3])
        return points

    def project_cad_to_camera_pcd(self, index, camera_pose, align_mat, scene_points):
        model_list, obj_list, pose_list = generate_scene_model(self.root, self.scenename[index], self.frameid[index],
                                                               return_poses=True,
                                                               align=False, camera="kinect")
        table = self.create_table_points(1.0, 1.0, 0.01, dx=-0.5, dy=-0.5, dz=0, grid_size=[0.002, 0.002, 0.008])
        table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
        t = o3d.geometry.PointCloud()
        t.points = o3d.utility.Vector3dVector(table_trans)
        pcd_combined = o3d.geometry.PointCloud()
        seg_id_list = []
        for i in range(len(model_list)):
            model = model_list[i].voxel_down_sample(0.005)
            pcd_combined += model
            seg_id_list.append(np.ones(len(model.points)) * (obj_list[i]+1))
        pcd_combined += t
        seg_id_list.append(np.zeros(len(t.points)))
        seg_mask = np.concatenate(seg_id_list, axis=0)
        scene_w_noise = o3d.geometry.PointCloud()
        scene_w_noise.points = o3d.utility.Vector3dVector(scene_points)

        dists = pcd_combined.compute_point_cloud_distance(scene_w_noise)
        dists = np.asarray(dists)
        ind = np.where(dists < 0.008)[0]
        pcd_combined_crop = pcd_combined.select_by_index(ind)
        seg_mask = seg_mask[ind]
        return np.asarray(pcd_combined_crop.points), seg_mask


import matplotlib.pyplot as plt


def get_color_mask(object_index, nc=None):
    """ Colors each index differently. Useful for visualizing semantic masks
        @param object_index: a [H x W] numpy array of ints from {0, ..., nc-1}
        @param nc: total number of colors. If None, this will be inferred by masks
        @return: a [H x W x 3] numpy array of dtype np.uint8
    """
    object_index = object_index.astype(int)

    if nc is None:
        NUM_COLORS = object_index.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]

    color_mask = np.zeros(object_index.shape + (3,)).astype(np.uint8)
    color_mask[:] = np.asarray([123,123,123])
    t = 0
    for i in np.unique(object_index):

        if i == 0 or i == -1:
            t += 1
            continue
        color_mask[object_index == i, :] = np.array(colors[t][:3]) * 255
        t += 1
    return color_mask


if __name__ == "__main__":
    dataset = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs=None, grasp_labels=None, split='train',
                        camera=cfgs.camera,
                        num_points=20000, remove_outlier=True, augment=False, load_label=False)
    print(len(dataset))
    pool = multiprocessing.Pool(cfgs.num_workers)
    for i in tqdm(range(len(dataset))):
        pool.apply_async(dataset.save_data,args=(i,))
    pool.close()
    pool.join()

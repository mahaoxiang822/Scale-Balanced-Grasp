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
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image,\
                            get_workspace_mask, remove_invisible_grasp_points
import open3d as o3d
from graspnetAPI.utils.utils import *
from graspnetAPI.utils.eval_utils import create_table_points,transform_points
import multiprocessing

def generate_scene_model(dataset_root, scene_name, anno_idx, return_poses=False, align=False, camera='realsense'):
    if align:
        camera_poses = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'camera_poses.npy'))
        camera_pose = camera_poses[anno_idx]
        align_mat = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'cam0_wrt_table.npy'))
        camera_pose = np.matmul(align_mat,camera_pose)
    scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', scene_name, camera, 'annotations', '%04d.xml'%anno_idx))
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
        plyfile = os.path.join(dataset_root, 'models', '%03d'%obj_idx, 'nontextured.ply')
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
                 remove_outlier=False, remove_invisible=True, augment=False, load_label=True, mode = "pure", val = False):
        assert(num_points<=50000)
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
        self.mode = mode
        self.val = val
        # self.step = 10

        if split == 'train':
            self.sceneIds = list( range(100) )
        elif split == 'test':
            self.sceneIds = list( range(100,190) )
        elif split == 'test_seen':
            self.sceneIds = list( range(100,130) )
        elif split == 'test_similar':
            self.sceneIds = list( range(130,190) )
        elif split == 'test_novel':
            self.sceneIds = list( range(160,190) )
        elif split == 'all':
            self.sceneIds = list(range(0,190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        
        self.pcdpath = []
        self.segpath = []

        self.colorpath = []
        self.depthpath = []
        self.labelpath = []

        self.metapath = []
        self.scenename = []
        self.frameid = []
        for x in tqdm(self.sceneIds, desc = 'Loading data path and collision labels...'):
            for img_num in range(256):
                self.pcdpath.append(os.path.join(root, 'clean_scenes', x, camera, 'points', str(img_num).zfill(4)+'.npy'))
                self.segpath.append(os.path.join(root, 'clean_scenes', x, camera, 'seg', str(img_num).zfill(4)+'.npy'))
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4)+'.mat'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(),  'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]


    def scene_list(self):
        return self.scenename

    def __len__(self):
        # return int(len(self.pcdpath) / self.step)
        return len(self.pcdpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        aug_trans = np.array([[1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 1]])
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)
            aug_trans = np.dot(aug_trans,flip_mat.T)


        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c,-s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)
        aug_trans = np.dot(aug_trans,rot_mat.T)

        return point_clouds, object_poses_list, aug_trans

    def __getitem__(self, index):
        if self.load_label:
            if self.mode == "mix":
                flag = np.random.randint(0,2)
                if flag == 0:
                    return self.get_data_label(index)
                else:
                    return self.get_data_label_noise(index)
            else:
                return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index):

        cloud_masked = np.load(self.pcdpath[index])
        seg_masked = np.load(self.segpath[index])
        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        #ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        return ret_dict

    def get_data_label(self, index):
        cloud_masked = np.load(self.pcdpath[index])
        seg_masked = np.load(self.segpath[index])
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)

        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        seg_sampled = seg_masked[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label>1] = 1

        # filter the collision point
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []

        #collision_list = np.load(os.path.join(root, 'collision_label', scene, 'collision_labels.npz'))

        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, offsets, scores, tolerance = self.grasp_labels[obj_idx]
            #collision = collision_list[i]
            collision = self.collision_labels[scene][i] #(Np, V, A, D)
            # remove invisible grasp points
            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled==obj_idx], points, poses[:,:,i], th=0.01)
                points = points[visible_mask]
                offsets = offsets[visible_mask]
                scores = scores[visible_mask]
                tolerance = tolerance[visible_mask]
                collision = collision[visible_mask]

            idxs = np.random.choice(len(points), min(max(int(len(points)/4),300),len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            tolerance = tolerance[idxs].copy()
            tolerance[collision] = 0
            grasp_tolerance_list.append(tolerance)

        ret_dict = {}
        if self.augment:
            cloud_sampled, object_poses_list, aug_trans = self.augment_data(cloud_sampled, object_poses_list)
            ret_dict['aug_trans'] = aug_trans

        # # transform to world coordinate in statistic angle
        # cloud_sampled = transform_point_cloud(cloud_sampled, trans[:3,:3], '3x3')
        # for i in range(len(object_poses_list)):
        #     object_poses_list[i] = np.dot(trans[:3,:3], object_poses_list[i]).astype(np.float32)

        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list
        ret_dict['trans'] = trans
        # ret_dict['instance_mask'] = seg_sampled
        return ret_dict

    def get_data_label_noise(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

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

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled = seg_masked[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label>1] = 1
        # filter the collision point
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []

        #collision_list = np.load(os.path.join(root, 'collision_label', scene, 'collision_labels.npz'))
        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, offsets, scores, tolerance = self.grasp_labels[obj_idx]

            #collision = collision_list[i]
            collision = self.collision_labels[scene][i] #(Np, V, A, D)

            # remove invisible grasp points
            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled==obj_idx], points, poses[:,:,i], th=0.01)
                points = points[visible_mask]
                offsets = offsets[visible_mask]
                scores = scores[visible_mask]
                tolerance = tolerance[visible_mask]
                collision = collision[visible_mask]

            idxs = np.random.choice(len(points), min(max(int(len(points)/4),300),len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            tolerance = tolerance[idxs].copy()
            tolerance[collision] = 0
            grasp_tolerance_list.append(tolerance)

        ret_dict = {}
        if self.augment:
            cloud_sampled, object_poses_list, aug_trans = self.augment_data(cloud_sampled, object_poses_list)
            ret_dict['aug_trans'] = aug_trans

        # # transform to world coordinate in statistic angle
        # cloud_sampled = transform_point_cloud(cloud_sampled, trans[:3,:3], '3x3')
        # for i in range(len(object_poses_list)):
        #     object_poses_list[i] = np.dot(trans[:3,:3], object_poses_list[i]).astype(np.float32)

        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        # ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list
        ret_dict['trans'] = trans

        return ret_dict

    def create_table_points(self, lx, ly, lz, dx=0, dy=0, dz=0, grid_size=[0.01,0.01,0.01]):
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

    def project_cad_to_camera_pcd(self,index,camera_pose,align_mat,scene_points):
        model_list, obj_list, pose_list = generate_scene_model(self.root, self.scenename[index], self.frameid[index], return_poses=True,
                                          align=False, camera="realsense")
        table = self.create_table_points(1.0, 1.0, 0.01, dx=-0.5, dy=-0.5, dz=0, grid_size=[0.002,0.002,0.008])
        table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
        t = o3d.geometry.PointCloud()
        t.points = o3d.utility.Vector3dVector(table_trans)
        pcd_combined = o3d.geometry.PointCloud()
        seg_id_list = []
        for i in range(len(model_list)):
            model = model_list[i].voxel_down_sample(0.002)
            pcd_combined += model
            seg_id_list.append(np.ones(len(model.points))*obj_list[i])
        pcd_combined += t
        seg_id_list.append(np.zeros(len(t.points)))
        seg_mask = np.concatenate(seg_id_list,axis=0)
        scene_w_noise = o3d.geometry.PointCloud()
        scene_w_noise.points = o3d.utility.Vector3dVector(scene_points)
        dists = pcd_combined.compute_point_cloud_distance(scene_w_noise)
        dists = np.asarray(dists)
        ind = np.where(dists < 0.008)[0]
        pcd_combined_crop = pcd_combined.select_by_index(ind)
        seg_mask = seg_mask[ind]
        # color_mask = get_color_mask(seg_mask,nc=len(obj_list)+1)/255
        #pcd_combined_crop.colors = o3d.utility.Vector3dVector(color_mask)
        # o3d.visualization.draw_geometries([pcd_combined_crop])
        # o3d.visualization.draw_geometries([scene_w_noise])
        return np.asarray(pcd_combined_crop.points),seg_mask


class GraspNetSegDataset(GraspNetDataset):
    def get_data_label(self, index):

        cloud_masked = np.load(self.pcdpath[index])
        seg_masked = np.load(self.segpath[index])
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)

        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        seg_sampled = seg_masked[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label>1] = 1

        # filter the collision point
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []

        #collision_list = np.load(os.path.join(root, 'collision_label', scene, 'collision_labels.npz'))
        if self.val:
            seg_sampled[seg_sampled>0] +=1
        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, offsets, scores, tolerance = self.grasp_labels[obj_idx]
            #collision = collision_list[i]
            collision = self.collision_labels[scene][i] #(Np, V, A, D)
            # remove invisible grasp points
            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled==obj_idx], points, poses[:,:,i], th=0.01)
                points = points[visible_mask]
                offsets = offsets[visible_mask]
                scores = scores[visible_mask]
                tolerance = tolerance[visible_mask]
                collision = collision[visible_mask]

            idxs = np.random.choice(len(points), min(max(int(len(points)/4),300),len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            tolerance = tolerance[idxs].copy()
            tolerance[collision] = 0
            grasp_tolerance_list.append(tolerance)

        ret_dict = {}
        if self.augment:
            cloud_sampled, object_poses_list, aug_trans = self.augment_data(cloud_sampled, object_poses_list)
            ret_dict['aug_trans'] = aug_trans
        offsets = np.zeros((len(seg_sampled), 3), dtype=np.float32)
        cf_3D_centers = np.zeros((100, 3), dtype=np.float32)  # 100 max object centers
        for i, k in enumerate(np.unique(seg_sampled)):
            mask = seg_sampled == k
            if k == 0:
                offsets[mask, ...] = 0
                continue
            # Compute 3D center
            center = np.average(cloud_sampled[mask], axis=0)
            cf_3D_centers[i - 1] = center

            # Compute directions
            object_center_offsets = (center - cloud_sampled).astype(np.float32)
            offsets[mask, ...] = object_center_offsets[mask, ...]

        foreground_mask = (seg_sampled > 0).astype(np.int64)
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list
        ret_dict['trans'] = trans
        ret_dict['foreground_mask'] = foreground_mask
        ret_dict['instance_mask'] = seg_sampled
        ret_dict['cf_3D_centers'] = cf_3D_centers
        ret_dict['3D_offsets'] = offsets
        ret_dict['num_3D_centers'] = np.array(len(np.unique(seg_sampled) - 1))
        return ret_dict


class GraspNetDataset_Align(Dataset):
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
        elif split == 'test_train':
            self.sceneIds = list(range(0, 10))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.colorpath = []
        self.depthpath = []
        self.pcdpath = []
        self.segpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for img_num in range(256):
                self.pcdpath.append(os.path.join(root, 'clear_scenes', x, camera, 'points', str(img_num).zfill(4) + '.npy'))
                self.segpath.append(
                    os.path.join(root, 'clear_scenes', x, camera, 'seg', str(img_num).zfill(4) + '.npy'))
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

    def augment_data(self, point_clouds,point_clouds_wonoise, object_poses_list):
        # Flipping along the YZ plane
        aug_trans = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            point_clouds_wonoise = transform_point_cloud(point_clouds_wonoise, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)
            aug_trans = np.dot(aug_trans, flip_mat.T)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        point_clouds_wonoise = transform_point_cloud(point_clouds_wonoise, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)
        aug_trans = np.dot(aug_trans, rot_mat.T)

        return point_clouds,point_clouds_wonoise, object_poses_list, aug_trans

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
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
        clear_cloud_masked = np.load(self.pcdpath[index])
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

        if len(clear_cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(clear_cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(clear_cloud_masked))
            idxs2 = np.random.choice(len(clear_cloud_masked), self.num_points-len(clear_cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        clear_cloud_sampled = clear_cloud_masked[idxs]

        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['clear_point_clouds'] = clear_cloud_sampled.astype(np.float32)

        return ret_dict

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
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
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label > 1] = 1


        # sample wonoise points
        clear_cloud_masked = np.load(self.pcdpath[index])
        clear_seg_masked = np.load(self.segpath[index])
        if len(clear_cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(clear_cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(clear_cloud_masked))
            idxs2 = np.random.choice(len(clear_cloud_masked), self.num_points-len(clear_cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        clear_cloud_sampled = clear_cloud_masked[idxs]
        clear_seg_sampled = clear_seg_masked[idxs]


        # filter the collision point
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []

        # collision_list = np.load(os.path.join(root, 'collision_label', scene, 'collision_labels.npz'))
        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, offsets, scores, tolerance = self.grasp_labels[obj_idx]

            # collision = collision_list[i]
            collision = self.collision_labels[scene][i]  # (Np, V, A, D)

            # remove invisible grasp points
            if self.remove_invisible:
                # todo: clear or no clear?
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled == obj_idx], points,
                                                             poses[:, :, i], th=0.01)
                points = points[visible_mask]
                offsets = offsets[visible_mask]
                scores = scores[visible_mask]
                tolerance = tolerance[visible_mask]
                collision = collision[visible_mask]

            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            tolerance = tolerance[idxs].copy()
            tolerance[collision] = 0
            grasp_tolerance_list.append(tolerance)

        ret_dict = {}
        if self.augment:
            cloud_sampled,clear_cloud_sampled, object_poses_list, aug_trans = self.augment_data(cloud_sampled,clear_cloud_sampled, object_poses_list)
            ret_dict['aug_trans'] = aug_trans

        # # transform to world coordinate in statistic angle
        # cloud_sampled = transform_point_cloud(cloud_sampled, trans[:3,:3], '3x3')
        # for i in range(len(object_poses_list)):
        #     object_poses_list[i] = np.dot(trans[:3,:3], object_poses_list[i]).astype(np.float32)

        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['clear_point_clouds'] = clear_cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list
        ret_dict['trans'] = trans
        return ret_dict

class GraspNetDataset_mix(GraspNetDataset_Align):

    def augment_data(self, point_clouds,npcd,cpcd, object_poses_list):
        # Flipping along the YZ plane
        aug_trans = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            npcd = transform_point_cloud(npcd, flip_mat, '3x3')
            cpcd = transform_point_cloud(cpcd, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)
            aug_trans = np.dot(aug_trans, flip_mat.T)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        npcd = transform_point_cloud(npcd, rot_mat, '3x3')
        cpcd = transform_point_cloud(cpcd, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)
        aug_trans = np.dot(aug_trans, rot_mat.T)

        return point_clouds,npcd,cpcd, object_poses_list, aug_trans

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
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

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        noise_cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        noise_seg_sampled = seg_masked[idxs]

        # sample wonoise points
        clear_cloud_masked = np.load(self.pcdpath[index])
        clear_seg_masked = np.load(self.segpath[index])
        if len(clear_cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(clear_cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(clear_cloud_masked))
            idxs2 = np.random.choice(len(clear_cloud_masked), self.num_points-len(clear_cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        clear_cloud_sampled = clear_cloud_masked[idxs]
        clear_seg_sampled = clear_seg_masked[idxs]

        mix_cloud_sampled,mix_seg_sampled = self.mix(noise_cloud_sampled,noise_seg_sampled,clear_cloud_sampled,clear_seg_sampled)


        if len(mix_cloud_sampled) >= self.num_points:
            idxs = np.random.choice(len(mix_cloud_sampled), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(mix_cloud_sampled))
            idxs2 = np.random.choice(len(mix_cloud_sampled), self.num_points - len(mix_cloud_sampled), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = mix_cloud_sampled[idxs]
        seg_sampled = mix_seg_sampled[idxs]
        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label > 1] = 1

        # filter the collision point
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []

        # collision_list = np.load(os.path.join(root, 'collision_label', scene, 'collision_labels.npz'))
        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, offsets, scores, tolerance = self.grasp_labels[obj_idx]

            # collision = collision_list[i]
            collision = self.collision_labels[scene][i]  # (Np, V, A, D)

            # remove invisible grasp points
            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled == obj_idx], points,
                                                             poses[:, :, i], th=0.01)
                points = points[visible_mask]
                offsets = offsets[visible_mask]
                scores = scores[visible_mask]
                tolerance = tolerance[visible_mask]
                collision = collision[visible_mask]

            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            tolerance = tolerance[idxs].copy()
            tolerance[collision] = 0
            grasp_tolerance_list.append(tolerance)

        ret_dict = {}
        if self.augment:
            cloud_sampled,noise_cloud_sampled,clear_cloud_sampled, object_poses_list, aug_trans = self.augment_data(cloud_sampled,noise_cloud_sampled,clear_cloud_sampled, object_poses_list)
            ret_dict['aug_trans'] = aug_trans

        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['noise_point_clouds'] = noise_cloud_sampled.astype(np.float32)
        ret_dict['clear_point_clouds'] = clear_cloud_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list
        ret_dict['trans'] = trans
        ret_dict['instance_mask'] = seg_sampled
        return ret_dict

    def mix(self,pcd,pcd_seg,cpcd,cpcd_seg):
        object_idxs = np.unique(pcd_seg)
        mix_pcd = []
        mix_pcd_seg = []
        for i,object_id in enumerate(object_idxs):
            if np.random.random() > 0.25:
                mix_pcd.append(pcd[pcd_seg == object_id])
                mix_pcd_seg.append(pcd_seg[pcd_seg == object_id])
            else:
                mix_pcd.append(cpcd[cpcd_seg == object_id])
                mix_pcd_seg.append(cpcd_seg[cpcd_seg == object_id])
        mix_pcd = np.concatenate(mix_pcd)
        mix_pcd_seg = np.concatenate(mix_pcd_seg)
        return mix_pcd,mix_pcd_seg


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
    print(color_mask.shape)
    t = 0
    for i in np.unique(object_index):

        if i == 0 or i == -1:
            t+=1
            continue
        color_mask[object_index == i, :] = np.array(colors[t][:3]) * 255
        t += 1
    return color_mask

def load_grasp_labels(root):
    obj_names = list(range(88))
    valid_obj_idxs = []
    grasp_labels = {}
    for i, obj_name in enumerate(tqdm(obj_names, desc='Loading grasping labels...')):
        if i == 18: continue
        valid_obj_idxs.append(i + 1) #here align with label png
        label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
        tolerance = np.load(os.path.join(BASE_DIR, 'tolerance', '{}_tolerance.npy'.format(str(i).zfill(3))))
        grasp_labels[i + 1] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
                                label['scores'].astype(np.float32), tolerance)

    return valid_obj_idxs, grasp_labels




def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key:collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))

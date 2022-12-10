import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from backbone import Pointnet2Backbone
from pct_zh import PointTransformerBackbone_light
from modules import *
from loss_utils import GRASP_MAX_WIDTH, GRASP_MAX_TOLERANCE, generate_grasp_views, batch_viewpoint_params_to_matrix
from pointnet2_utils import three_nn, three_interpolate
from pointnet2_util import index_points, square_distance

'''
GRASP_MAX_WIDTH 0.1
GRASP_MAX_TOLERANCE = 0.05
'''
from label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix


class GraspNetStage1(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300, obs=False, is_training=True):
        super().__init__()
        self.backbone = PointTransformerBackbone_light()
        self.vpmodule = ApproachNet(num_view, 256)
        self.obs = obs
        self.is_training = is_training

    def forward(self, end_points):
        pointcloud = end_points['point_clouds']
        seed_features, seed_xyz, end_points = self.backbone(pointcloud, end_points)

        if self.obs and (not self.is_training):
            dist, idx = three_nn(pointcloud, seed_xyz)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            up_sample_features = three_interpolate(seed_features, idx, weight)
            end_points['up_sample_features'] = up_sample_features
            end_points = ObjectBalanceSampling(end_points)
            # end_points = ForegroundSampling(end_points)
            seed_xyz = end_points['fp2_xyz']
            resample_features = end_points["fp2_features"]
            seed_features = resample_features

        end_points = self.vpmodule(seed_xyz, seed_features, end_points)
        return end_points


class GraspNetStage2(nn.Module):
    def __init__(self, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
                 is_training=True):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.is_training = is_training
        self.crop = CloudCrop(64, 3, cylinder_radius, hmin, hmax_list)
        self.operation = OperationNet(num_angle, num_depth)
        self.tolerance = ToleranceNet(num_angle, num_depth)

    def forward(self, end_points):
        pointcloud = end_points['input_xyz']
        if self.is_training:
            grasp_top_views_rot, _, _, _, end_points = match_grasp_view_and_label(end_points)
            seed_xyz = end_points['batch_grasp_point']
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']
            seed_xyz = end_points['fp2_xyz']
        vp_features = self.crop(seed_xyz, pointcloud, grasp_top_views_rot)
        end_points = self.operation(vp_features, end_points)
        end_points = self.tolerance(vp_features, end_points)
        return end_points


class GraspNetStage2_seed_features_multi_scale(nn.Module):
    def __init__(self, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
                 is_training=True):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.is_training = is_training

        self.crop1 = CloudCrop(64, 3, cylinder_radius * 0.25, hmin, hmax_list)
        self.crop2 = CloudCrop(64, 3, cylinder_radius * 0.5, hmin, hmax_list)
        self.crop3 = CloudCrop(64, 3, cylinder_radius * 0.75, hmin, hmax_list)
        self.crop4 = CloudCrop(64, 3, cylinder_radius, hmin, hmax_list)
        self.operation = OperationNet(num_angle, num_depth)
        self.tolerance = ToleranceNet(num_angle, num_depth)
        self.fuse_multi_scale = nn.Conv1d(256 * 4, 256, 1)
        self.gate_fusion = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.Sigmoid()
        )

    def forward(self, end_points):
        pointcloud = end_points['input_xyz']
        if self.is_training:
            grasp_top_views_rot, _, _, _, end_points = match_grasp_view_and_label(end_points)
            seed_xyz = end_points['batch_grasp_point']
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']
            seed_xyz = end_points['fp2_xyz']

        vp_features1 = self.crop1(seed_xyz, pointcloud, grasp_top_views_rot)
        vp_features2 = self.crop2(seed_xyz, pointcloud, grasp_top_views_rot)
        vp_features3 = self.crop3(seed_xyz, pointcloud, grasp_top_views_rot)
        vp_features4 = self.crop4(seed_xyz, pointcloud, grasp_top_views_rot)

        B, _, num_seed, num_depth = vp_features1.size()
        vp_features_concat = torch.cat([vp_features1, vp_features2, vp_features3, vp_features4], dim=1)
        vp_features_concat = vp_features_concat.view(B, -1, num_seed * num_depth)
        vp_features_concat = self.fuse_multi_scale(vp_features_concat)
        vp_features_concat = vp_features_concat.view(B, -1, num_seed, num_depth)
        seed_features = end_points['fp2_features']
        seed_features_gate = self.gate_fusion(seed_features) * seed_features
        seed_features_gate = seed_features_gate.unsqueeze(3).repeat(1, 1, 1, 4)
        vp_features = vp_features_concat + seed_features_gate
        end_points = self.operation(vp_features, end_points)
        end_points = self.tolerance(vp_features, end_points)
        return end_points


class GraspNet(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.08, hmin=-0.02,
                 hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=True, obs=False):
        super().__init__()
        self.is_training = is_training
        self.view_estimator = GraspNetStage1(input_feature_dim, num_view, obs, is_training)
        self.grasp_generator = GraspNetStage2_seed_features(num_angle, num_depth, cylinder_radius, hmin,
                                                            hmax_list, is_training)

    def forward(self, end_points):
        end_points = self.view_estimator(end_points)
        if self.is_training:
            end_points = process_grasp_labels(end_points)
        end_points = self.grasp_generator(end_points)
        return end_points


class GraspNet_MSCQ(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.08, hmin=-0.02,
                 hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=True, obs=False):
        super().__init__()
        self.is_training = is_training
        self.view_estimator = GraspNetStage1(input_feature_dim, num_view, is_training=is_training, obs=obs)
        self.grasp_generator = GraspNetStage2_seed_features_multi_scale(num_angle, num_depth, cylinder_radius, hmin,
                                                                        hmax_list, is_training)

    def forward(self, end_points):
        end_points = self.view_estimator(end_points)
        if self.is_training:
            end_points = process_grasp_labels(end_points)
        end_points = self.grasp_generator(end_points)
        return end_points


def pred_decode(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        ## load predictions
        objectness_score = end_points['objectness_score'][i].float()
        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_center = end_points['fp2_xyz'][i].float()
        # grasp_center = end_points['fp2_xyz'][i].float()+end_points['shift_pred'][i].float()
        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_angle_class_score = end_points['grasp_angle_cls_pred'][i]
        grasp_width = 1.2 * end_points['grasp_width_pred'][i]
        grasp_width = torch.clamp(grasp_width, min=0, max=GRASP_MAX_WIDTH)
        grasp_tolerance = end_points['grasp_tolerance_pred'][i]

        ## slice preds by angle
        # grasp angle
        grasp_angle_class = torch.argmax(grasp_angle_class_score, 0)
        grasp_angle = grasp_angle_class.float() / 12 * np.pi
        # grasp score & width & tolerance
        grasp_angle_class_ = grasp_angle_class.unsqueeze(0)
        grasp_score = torch.gather(grasp_score, 0, grasp_angle_class_).squeeze(0)
        grasp_width = torch.gather(grasp_width, 0, grasp_angle_class_).squeeze(0)
        grasp_tolerance = torch.gather(grasp_tolerance, 0, grasp_angle_class_).squeeze(0)

        ## slice preds by score/depth
        # grasp depth
        grasp_depth_class = torch.argmax(grasp_score, 1, keepdims=True)
        grasp_depth = (grasp_depth_class.float() + 1) * 0.01
        # grasp score & angle & width & tolerance
        grasp_score = torch.gather(grasp_score, 1, grasp_depth_class)
        grasp_angle = torch.gather(grasp_angle, 1, grasp_depth_class)
        grasp_width = torch.gather(grasp_width, 1, grasp_depth_class)
        grasp_tolerance = torch.gather(grasp_tolerance, 1, grasp_depth_class)

        ## slice preds by objectness
        objectness_pred = torch.argmax(objectness_score, 0)
        objectness_mask = (objectness_pred == 1)

        graspable_confident = torch.softmax(objectness_score, dim=0)[1, :].unsqueeze(1)
        grasp_score = grasp_score * graspable_confident

        grasp_score = grasp_score[objectness_mask]
        grasp_width = grasp_width[objectness_mask]
        grasp_depth = grasp_depth[objectness_mask]
        approaching = approaching[objectness_mask]
        grasp_angle = grasp_angle[objectness_mask]
        grasp_center = grasp_center[objectness_mask]
        grasp_tolerance = grasp_tolerance[objectness_mask]
        grasp_score = grasp_score * grasp_tolerance / GRASP_MAX_TOLERANCE

        ## convert to rotation matrix
        Ns = grasp_angle.size(0)
        approaching_ = approaching.view(Ns, 3)
        grasp_angle_ = grasp_angle.view(Ns)
        rotation_matrix = batch_viewpoint_params_to_matrix(approaching_, grasp_angle_)
        rotation_matrix = rotation_matrix.view(Ns, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids],
                      axis=-1))
    return grasp_preds

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import time
import copy
from torch.autograd import Variable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from loss_utils import GRASP_MAX_WIDTH, GRASP_MAX_TOLERANCE, THRESH_GOOD, THRESH_BAD, \
    transform_point_cloud, generate_grasp_views, \
    batch_viewpoint_params_to_matrix, huber_loss, FocalLoss_Ori , BinaryFocalLoss


# width distribution prior
width_distribution_prior = np.load("statistic/best_width_distribution_robust_32.npy", allow_pickle=True).item()
width_distribution_prior_num = width_distribution_prior['num']
intervals = width_distribution_prior['interval']
width_distribution_prior_ = torch.zeros(32)
for i in range(32):
    width_distribution_prior_[i] = width_distribution_prior_num[i]
width_max_prob = torch.max(width_distribution_prior_)
width_distribution_prior_ = - (width_distribution_prior_ / width_max_prob).log() + 1
width_distribution_prior = width_distribution_prior_.cuda()
print(width_distribution_prior)

def generate_reweight_mask(end_points):
    # load distribution prior
    batch_grasp_label = end_points['batch_grasp_label_all']  # (B, Ns,V, A, D)
    B, Ns, V, A, D = batch_grasp_label.size()
    batch_grasp_offset = end_points['batch_grasp_offset_all']
    top_view_grasp_widths = batch_grasp_offset[:, :, :, :,:, 2].reshape(B, Ns, -1)
    batch_grasp_label = batch_grasp_label.reshape(B, Ns, -1)
    target_labels_inds = torch.argmax(batch_grasp_label, dim=2, keepdim=True)  # (B, Ns, 1)
    target_widths = torch.gather(top_view_grasp_widths, 2, target_labels_inds).squeeze(2)  # (B, Ns)
    id_mask = torch.zeros(size=target_widths.size()).long()
    for idx in range(len(intervals) - 1):
        id_mask[(intervals[idx] < target_widths) * (intervals[idx + 1] > target_widths)] = idx
    weight_mask = width_distribution_prior[id_mask]
    return weight_mask


def get_loss(end_points):
    reweight_mask = generate_reweight_mask(end_points)
    objectness_loss, end_points = compute_robust_graspable_loss(end_points)
    view_loss, end_points = compute_weighted_view_loss(end_points, reweight_mask.clone())
    grasp_loss, end_points = compute_weighted_grasp_loss(end_points, reweight_mask.clone())
    loss = objectness_loss + view_loss + 0.2 * grasp_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_robust_graspable_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    fp2_inds = end_points['fp2_inds'].long()
    objectness_label = torch.gather(objectness_label, 1, fp2_inds)
    batch_grasp_label = end_points['batch_grasp_label_all']
    B, Ns, V, A, D = batch_grasp_label.size()
    target_labels = batch_grasp_label.view(B, Ns, V, -1)
    target_labels = target_labels.max(3)[0]
    graspable_cnt = torch.sum((target_labels > THRESH_BAD).long(), dim=2)
    graspable_label = (graspable_cnt > 10) * objectness_label
    end_points['graspable_mask'] = graspable_label

    loss = criterion(objectness_score, graspable_label)
    end_points['loss/stage1_graspable_loss'] = loss
    graspable_pred = torch.argmax(objectness_score, 1)
    end_points['stage1_graspable_acc'] = (graspable_pred == graspable_label.long()).float().mean()
    end_points['stage1_graspable_prec'] = (graspable_pred == graspable_label.long())[graspable_pred == 1].float().mean()
    end_points['stage1_graspable_recall'] = (graspable_pred == graspable_label.long())[
        graspable_label == 1].float().mean()
    return loss, end_points


def compute_weighted_view_loss(end_points, weight_mask, width_weight_mask=None):
    criterion = nn.MSELoss(reduction='none')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_label']
    objectness_label = end_points['objectness_label']
    fp2_inds = end_points['fp2_inds'].long()
    V = view_label.size(2)
    objectness_label = torch.gather(objectness_label, 1, fp2_inds)

    batch_grasp_label = end_points['batch_grasp_label_all']  # (B, Ns, A, D)
    B, Ns, V, A, D = batch_grasp_label.size()

    # target_labels = batch_grasp_label.view(B, Ns, -1)
    # target_labels = target_labels.max(2)[0]
    # graspable_label = (target_labels > THRESH_BAD).long()

    #robust
    target_labels = batch_grasp_label.view(B, Ns, V, -1)
    target_labels = target_labels.max(3)[0]
    graspable_cnt = torch.sum((target_labels > THRESH_BAD).long(), dim=2)
    graspable_label = (graspable_cnt > 10) * objectness_label

    graspable_label = graspable_label * objectness_label
    objectness_mask = (graspable_label > 0)
    objectness_mask = objectness_mask.unsqueeze(-1).repeat(1, 1, V)
    if not (width_weight_mask is None):
        weight_mask = (weight_mask * width_weight_mask)
    weight_mask = weight_mask.unsqueeze(-1).repeat(1, 1, V)
    loss_mask = objectness_mask.float() * weight_mask
    pos_view_pred_mask = ((view_score >= THRESH_GOOD) & objectness_mask)

    loss = criterion(view_score, view_label)

    loss = torch.sum(loss * loss_mask) / (loss_mask.sum() + 1e-6)

    end_points['loss/stage1_view_loss'] = loss
    end_points['stage1_pos_view_pred_count'] = pos_view_pred_mask.long().sum()

    return loss, end_points


def compute_weighted_grasp_loss(end_points, weight_mask):
    objectness_label = end_points['objectness_label']
    fp2_inds = end_points['fp2_inds'].long()
    objectness_mask = torch.gather(objectness_label, 1, fp2_inds).bool()  # (B, Ns)

    # process labels
    batch_grasp_label = end_points['batch_grasp_label']  # (B, Ns, A, D)
    batch_grasp_offset = end_points['batch_grasp_offset']  # (B, Ns, A, D, 3)
    batch_grasp_tolerance = end_points['batch_grasp_tolerance']  # (B, Ns, A, D)
    B, Ns, A, D = batch_grasp_label.size()

    # pick the one with the highest angle score
    top_view_grasp_angles = batch_grasp_offset[:, :, :, :, 0]  # (B, Ns, A, D)
    top_view_grasp_depths = batch_grasp_offset[:, :, :, :, 1]  # (B, Ns, A, D)
    top_view_grasp_widths = batch_grasp_offset[:, :, :, :, 2]  # (B, Ns, A, D)
    target_labels_inds = torch.argmax(batch_grasp_label, dim=2, keepdim=True)  # (B, Ns, 1, D)
    target_labels = torch.gather(batch_grasp_label, 2, target_labels_inds).squeeze(2)  # (B, Ns, D)
    target_widths = torch.gather(top_view_grasp_widths, 2, target_labels_inds).squeeze(2)  # (B, Ns, D)
    target_tolerance = torch.gather(batch_grasp_tolerance, 2, target_labels_inds).squeeze(2)  # (B, Ns, D)

    graspable_mask = (target_labels > THRESH_BAD)

    objectness_mask = objectness_mask.unsqueeze(-1).expand_as(graspable_mask)
    weight_mask = weight_mask.unsqueeze(-1).expand_as(graspable_mask)
    loss_mask = (objectness_mask & graspable_mask).float() * weight_mask

    # 1. grasp score loss
    depth_loss_mask = loss_mask.max(dim=2)[0].unsqueeze(-1).expand_as(loss_mask)
    target_labels_inds_ = target_labels_inds.transpose(1, 2)  # (B, 1, Ns, D)
    grasp_score = torch.gather(end_points['grasp_score_pred'], 1, target_labels_inds_).squeeze(1)
    grasp_score_loss = huber_loss(grasp_score - target_labels, delta=1.0)
    grasp_score_loss = torch.sum(grasp_score_loss * depth_loss_mask) / (depth_loss_mask.sum() + 1e-6)
    end_points['loss/stage2_grasp_score_loss'] = grasp_score_loss

    # 2. inplane rotation cls loss
    target_angles_cls = target_labels_inds.squeeze(2)  # (B, Ns, D)
    criterion_grasp_angle_class = nn.CrossEntropyLoss(reduction='none')
    grasp_angle_class_score = end_points['grasp_angle_cls_pred']
    grasp_angle_class_loss = criterion_grasp_angle_class(grasp_angle_class_score, target_angles_cls)
    grasp_angle_class_loss = torch.sum(grasp_angle_class_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    end_points['loss/stage2_grasp_angle_class_loss'] = grasp_angle_class_loss
    grasp_angle_class_pred = torch.argmax(grasp_angle_class_score, 1)
    end_points['stage2_grasp_angle_class_acc/0_degree'] = (grasp_angle_class_pred == target_angles_cls)[
        loss_mask.bool()].float().mean()
    acc_mask_15 = ((torch.abs(grasp_angle_class_pred - target_angles_cls) <= 1) | (
            torch.abs(grasp_angle_class_pred - target_angles_cls) >= A - 1))
    end_points['stage2_grasp_angle_class_acc/15_degree'] = acc_mask_15[loss_mask.bool()].float().mean()
    acc_mask_30 = ((torch.abs(grasp_angle_class_pred - target_angles_cls) <= 2) | (
            torch.abs(grasp_angle_class_pred - target_angles_cls) >= A - 2))
    end_points['stage2_grasp_angle_class_acc/30_degree'] = acc_mask_30[loss_mask.bool()].float().mean()

    # 3. width reg loss
    grasp_width_pred = torch.gather(end_points['grasp_width_pred'], 1, target_labels_inds_).squeeze(1)
    grasp_width_loss = huber_loss((grasp_width_pred - target_widths) / GRASP_MAX_WIDTH, delta=1)
    grasp_width_loss = torch.sum(grasp_width_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    end_points['loss/stage2_grasp_width_loss'] = grasp_width_loss

    # 4. tolerance reg loss
    grasp_tolerance_pred = torch.gather(end_points['grasp_tolerance_pred'], 1, target_labels_inds_).squeeze(1)
    grasp_tolerance_loss = huber_loss((grasp_tolerance_pred - target_tolerance) / GRASP_MAX_TOLERANCE, delta=1)
    grasp_tolerance_loss = torch.sum(grasp_tolerance_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    end_points['loss/stage2_grasp_tolerance_loss'] = grasp_tolerance_loss

    grasp_loss = grasp_score_loss + grasp_angle_class_loss \
                 + grasp_width_loss + grasp_tolerance_loss
    return grasp_loss, end_points


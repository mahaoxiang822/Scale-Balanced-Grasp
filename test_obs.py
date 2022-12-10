import os
import sys
import numpy as np
import argparse
import time

import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup, GraspNetEval

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet_MSCQ, pred_decode
import torch.nn.functional as F
from dsn import DSN,cluster
from graspnet_dataset import GraspNetDataset, collate_fn
from collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--seg_checkpoint_path', required=True, help='Segmentation Model checkpoint path')
parser.add_argument('--dump_dir', required=True, help='Dump dir to save outputs')
parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during inference [default: 1]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--num_workers', type=int, default=30, help='Number of workers used in evaluation [default: 30]')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir): os.mkdir(cfgs.dump_dir)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# Create Dataset and Dataloader
TEST_DATASET = GraspNetDataset(cfgs.dataset_root, None, None, split='test', camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, augment=False, load_label=False)
print(len(TEST_DATASET))
SCENE_LIST = TEST_DATASET.scene_list()
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
print(len(TEST_DATALOADER))
# Init the model
net = GraspNet_MSCQ(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                     cylinder_radius=0.08, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False, obs=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# Load checkpoint
checkpoint = torch.load(cfgs.checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))

#load seg network
seg_net = DSN(input_feature_dim=0)
seg_net.to(device)
# Load checkpoint
checkpoint = torch.load(cfgs.seg_checkpoint_path)
seg_net.load_state_dict(checkpoint['model_state_dict'])

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def inference():
    batch_interval = 100
    stat_dict = {} # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    tic = time.time()
    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
        
        # Forward pass
        with torch.no_grad():
            end_points = seg_net(batch_data)
            batch_xyz_img = end_points["point_clouds"]
            B, _, N = batch_xyz_img.shape
            batch_offsets = end_points["center_offsets"]
            batch_fg = end_points["foreground_logits"]
            batch_fg = F.softmax(batch_fg, dim=1)
            batch_fg = torch.argmax(batch_fg, dim=1)
            # end_points["instance_mask"] = batch_fg
            clustered_imgs = []
            for i in range(B):
                clustered_img, uniq_cluster_centers = cluster(batch_xyz_img[i], batch_offsets[i].permute(1, 0),
                                                              batch_fg[i])
                clustered_img = clustered_img.unsqueeze(0)
                clustered_imgs.append(clustered_img)
            end_points['seed_cluster'] = torch.cat(clustered_imgs,dim=0)
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        # Dump results for evaluation
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            gg = GraspGroup(preds)
            #gg.nms(translation_thresh=0.03,rotation_thresh=(30.0/180.0)*np.pi)

            # collision detection
            if cfgs.collision_thresh > 0:
                cloud, _ = TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(cfgs.dump_dir, SCENE_LIST[data_idx], cfgs.camera)
            save_path = os.path.join(save_dir, str(data_idx%256).zfill(4)+'.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

        if batch_idx % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs'%(batch_idx, (toc-tic)/batch_interval))
            tic = time.time()

def evaluate():
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test')
    res, ap = ge.eval_all(cfgs.dump_dir, proc=cfgs.num_workers)
    save_dir = os.path.join(cfgs.dump_dir, 'ap_{}.npy'.format(cfgs.camera))
    np.save(save_dir, res)

if __name__=='__main__':
    inference()
    #evaluate()

import os
import sys
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
from dsn import DSN,cluster
from instanceseg_dataset import InsSegDataset,collate_fn
import time


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--dump_dir', required=True, help='Dump dir to save outputs')
parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--max_epoch', type=int, default=18, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--num_workers', type=int, default=2, help='workers num during training [default: 2]')

cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir): os.mkdir(cfgs.dump_dir)

# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

TEST_DATASET = InsSegDataset(cfgs.dataset_root, camera=cfgs.camera, split='test_novel',
                               num_points=cfgs.num_point, remove_outlier=True, augment=False)

print(len(TEST_DATASET))
SCENE_LIST = TEST_DATASET.scene_list()
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
                             num_workers=cfgs.num_workers, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
print(len(TEST_DATALOADER))
# Init the model and optimzier
net = DSN(input_feature_dim=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Load checkpoint
checkpoint = torch.load(cfgs.checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))


TEST_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'test'))


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
    for i in np.unique(object_index):
        if i == 0 or i == -1:
            continue
        color_mask[object_index == i, :] = np.array(colors[i][:3]) * 255
    return color_mask

def inference():
    batch_interval = 100
    stat_dict = {}  # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    tic = time.time()
    objects_F_measure = []
    objects_precision = []
    objects_recall = []
    obj_detected_075_percentage = []
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

            end_points = net(batch_data)
            batch_xyz_img = end_points["point_clouds"]
            B,_,N = batch_xyz_img.shape
            batch_offsets = end_points["center_offsets"]
            batch_fg = end_points["foreground_logits"]
            batch_fg = F.softmax(batch_fg,dim=1)
            batch_fg = torch.argmax(batch_fg,dim=1)
            for i in range(B):
                data_idx = batch_idx * cfgs.batch_size + i
                clustered_img, uniq_cluster_centers = cluster(batch_xyz_img[i],batch_offsets[i].permute(1,0),batch_fg[i])
                seg_mask = clustered_img.cpu().numpy()
                res = multilabel_metrics(prediction=seg_mask,gt=end_points['instance_mask'][i].cpu().numpy(),obj_detect_threshold=0.75)
                objects_F_measure.append(res['Objects F-measure'])
                objects_precision.append(res['Objects Precision'])
                objects_recall.append(res['Objects Recall'])
                obj_detected_075_percentage.append(res['obj_detected_075_percentage'])
                color_mask = get_color_mask(seg_mask).astype(np.float32)/255
                xyz = batch_xyz_img[i].cpu().numpy()
                result = {
                    "xyz":xyz,
                    "color":color_mask
                }
                save_dir = os.path.join(cfgs.dump_dir, SCENE_LIST[data_idx], cfgs.camera)
                save_path_npy = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.npy')
                save_path_ply = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.ply')
                scene = o3d.geometry.PointCloud()
                scene.points = o3d.utility.Vector3dVector(xyz)
                scene.colors = o3d.utility.Vector3dVector(color_mask)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                #np.save(save_path_npy,result)
                o3d.io.write_point_cloud(save_path_ply, scene)
        if batch_idx % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs' % (batch_idx, (toc - tic) / batch_interval))
            tic = time.time()
    print("F_measure",sum(objects_F_measure)/len(objects_F_measure))
    print("Precision", sum(objects_precision) / len(objects_precision))
    print("Recall", sum(objects_recall) / len(objects_recall))
    print("75_per_detected", sum(obj_detected_075_percentage) / len(obj_detected_075_percentage))



if __name__ == '__main__':
    inference()

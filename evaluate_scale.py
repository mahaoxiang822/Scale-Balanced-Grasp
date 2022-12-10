
import numpy as np
from graspnetAPI.graspnet_eval import *
from graspnetAPI.utils.eval_utils import *

global SCALE

def eval_grasp(grasp_group, models, dexnet_models, poses, config, table=None, voxel_size=0.008, TOP_K=50):
    '''
    **Input:**

    - grasp_group: GraspGroup instance for evaluation.

    - models: in model coordinate

    - dexnet_models: models in dexnet format

    - poses: from model to camera coordinate

    - config: dexnet config.

    - table: in camera coordinate

    - voxel_size: float of the voxel size.

    - TOP_K: int of the number of top grasps to evaluate.
    '''
    num_models = len(models)
    ## grasp nms
    grasp_group = grasp_group.nms(0.03, 30.0 / 180 * np.pi)

    ## assign grasps to object
    # merge and sample scene
    model_trans_list = list()
    seg_mask = list()
    for i, model in enumerate(models):
        model_trans = transform_points(model, poses[i])
        seg = i * np.ones(model_trans.shape[0], dtype=np.int32)
        model_trans_list.append(model_trans)
        seg_mask.append(seg)
    seg_mask = np.concatenate(seg_mask, axis=0)
    scene = np.concatenate(model_trans_list, axis=0)

    # assign grasps
    indices = compute_closest_points(grasp_group.translations, scene)
    model_to_grasp = seg_mask[indices]
    pre_grasp_list = list()
    for i in range(num_models):
        grasp_i = grasp_group[model_to_grasp == i]
        grasp_i.sort_by_score()
        pre_grasp_list.append(grasp_i[:5].grasp_group_array)
    all_grasp_list = np.vstack(pre_grasp_list)
    remain_mask = np.argsort(all_grasp_list[:, 0])[::-1]
    if len(remain_mask) ==0:
        grasp_list = []
        score_list = []
        collision_mask_list = []
        for i in range(num_models):
            grasp_list.append([])
            score_list.append([])
            collision_mask_list.append([])
        return grasp_list, score_list, collision_mask_list

    min_score = all_grasp_list[remain_mask[min(49, len(remain_mask) - 1)], 0]

    grasp_list = []
    for i in range(num_models):
        remain_mask_i = pre_grasp_list[i][:, 0] >= min_score
        grasp_list.append(pre_grasp_list[i][remain_mask_i])
    # grasp_list = pre_grasp_list

    ## collision detection
    if table is not None:
        scene = np.concatenate([scene, table])

    collision_mask_list, empty_list, dexgrasp_list = collision_detection(
        grasp_list, model_trans_list, dexnet_models, poses, scene, outlier=0.05, return_dexgrasps=True)

    ## evaluate grasps
    # score configurations
    force_closure_quality_config = dict()
    fc_list = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    for value_fc in fc_list:
        value_fc = round(value_fc, 2)
        config['metrics']['force_closure']['friction_coef'] = value_fc
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            config['metrics']['force_closure'])
    # get grasp scores
    score_list = list()

    for i in range(num_models):
        dexnet_model = dexnet_models[i]
        collision_mask = collision_mask_list[i]
        dexgrasps = dexgrasp_list[i]
        scores = list()
        num_grasps = len(dexgrasps)
        for grasp_id in range(num_grasps):
            if collision_mask[grasp_id]:
                scores.append(-1.)
                continue
            if dexgrasps[grasp_id] is None:
                scores.append(-1.)
                continue
            grasp = dexgrasps[grasp_id]
            score = get_grasp_score(grasp, dexnet_model, fc_list, force_closure_quality_config)
            scores.append(score)
        score_list.append(np.array(scores))

    return grasp_list, score_list, collision_mask_list

class GraspNetEval_scale(GraspNetEval):

    def eval_scene(self, scene_id, dump_folder, TOP_K=20, return_list=False, vis=False, max_width=0.1):
        '''
        **Input:**

        - scene_id: int of the scene index.

        - dump_folder: string of the folder that saves the dumped npy files.

        - TOP_K: int of the top number of grasp to evaluate

        - return_list: bool of whether to return the result list.

        - vis: bool of whether to show the result

        - max_width: float of the maximum gripper width in evaluation

        **Output:**

        - scene_accuracy: np.array of shape (256, 50, 6) of the accuracy tensor.
        '''
        config = get_config()
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)

        list_coe_of_friction = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

        model_list, dexmodel_list, _ = self.get_scene_models(scene_id, ann_id=0)

        model_sampled_list = list()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)

        scene_accuracy = []
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []

        for ann_id in range(256):
            grasp_group = GraspGroup().from_npy(
                os.path.join(dump_folder, get_scene_name(scene_id), self.camera, '%04d.npy' % (ann_id,)))
            _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

            # clip width to [0,max_width]
            gg_array = grasp_group.grasp_group_array
            min_width_mask = (gg_array[:, 1] < 0)
            max_width_mask = (gg_array[:, 1] > max_width)
            gg_array[min_width_mask, 1] = 0
            gg_array[max_width_mask, 1] = max_width
            if SCALE == "small":
                width_mask = (gg_array[:, 1] < 0.04)
            elif SCALE == "medium":
                width_mask = (gg_array[:, 1] >= 0.04) * (gg_array[:, 1] < 0.07)
            elif SCALE == "large":
                width_mask = (gg_array[:, 1] >= 0.07)
            else:
                print("unknown scale")
                exit(0)
            gg_array = gg_array[width_mask]
            grasp_group.grasp_group_array = gg_array

            grasp_list, score_list, collision_mask_list = eval_grasp(grasp_group, model_sampled_list, dexmodel_list,
                                                                     pose_list, config, table=table_trans,
                                                                     voxel_size=0.008, TOP_K=TOP_K)

            # remove empty
            grasp_list = [x for x in grasp_list if len(x) != 0]
            score_list = [x for x in score_list if len(x) != 0]
            collision_mask_list = [x for x in collision_mask_list if len(x) != 0]

            if len(grasp_list) == 0:
                grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
                scene_accuracy.append(grasp_accuracy)
                grasp_list_list.append([])
                score_list_list.append([])
                collision_list_list.append([])
                print('\rMean Accuracy for scene:{} ann:{}='.format(scene_id, ann_id), np.mean(grasp_accuracy[:, :]),
                      end='')
                continue

            # concat into scene level
            grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(
                score_list), np.concatenate(collision_mask_list)

            if vis:
                t = o3d.geometry.PointCloud()
                t.points = o3d.utility.Vector3dVector(table_trans)
                model_list = generate_scene_model(self.root, 'scene_%04d' % scene_id, ann_id, return_poses=False,
                                                  align=False, camera=self.camera)
                import copy
                gg = GraspGroup(copy.deepcopy(grasp_list))
                scores = np.array(score_list)
                scores = scores / 2 + 0.5  # -1 -> 0, 0 -> 0.5, 1 -> 1
                scores[collision_mask_list] = 0.3
                gg.scores = scores
                gg.widths = 0.1 * np.ones((len(gg)), dtype=np.float32)
                grasps_geometry = gg.to_open3d_geometry_list()
                pcd = self.loadScenePointCloud(scene_id, self.camera, ann_id)

                o3d.visualization.draw_geometries([pcd, *grasps_geometry])
                o3d.visualization.draw_geometries([pcd, *grasps_geometry, *model_list])
                o3d.visualization.draw_geometries([*grasps_geometry, *model_list, t])
            # sort in scene level
            grasp_confidence = grasp_list[:, 0]
            indices = np.argsort(-grasp_confidence)
            grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[
                indices]

            grasp_list_list.append(grasp_list)
            score_list_list.append(score_list)
            collision_list_list.append(collision_mask_list)

            # calculate AP
            grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
            for fric_idx, fric in enumerate(list_coe_of_friction):
                for k in range(0, TOP_K):
                    if k + 1 > len(score_list):
                        grasp_accuracy[k, fric_idx] = np.sum(((score_list <= fric) & (score_list > 0)).astype(int)) / (
                                    k + 1)
                    else:
                        grasp_accuracy[k, fric_idx] = np.sum(
                            ((score_list[0:k + 1] <= fric) & (score_list[0:k + 1] > 0)).astype(int)) / (k + 1)

            print('\rMean Accuracy for scene:%04d ann:%04d = %.3f' % (
            scene_id, ann_id, 100.0 * np.mean(grasp_accuracy[:, :])), end='', flush=True)
            scene_accuracy.append(grasp_accuracy)
        if not return_list:
            return scene_accuracy
        else:
            return scene_accuracy, grasp_list_list, score_list_list, collision_list_list
    def parallel_eval_scenes(self, scene_ids, dump_folder, proc = 2):
        '''
        **Input:**

        - scene_ids: list of int of scene index.

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - scene_acc_list: list of the scene accuracy.
        '''
        from multiprocessing import Pool
        p = Pool(processes = proc)
        res_list = []
        for scene_id in scene_ids:
            res_list.append(p.apply_async(self.eval_scene, (scene_id, dump_folder)))
        p.close()
        p.join()
        scene_acc_list = []
        for res in res_list:
            scene_acc_list.append(res.get())
        return scene_acc_list

    pass
ge = GraspNetEval_scale(root="/data/mahaoxiang/graspnet", camera="realsense", split='test')
dump_folder = "logs/dump_full_model"
scales = ["small", "medium", "large"]
for scale in scales:

    SCALE = scale

    print("----- evaluating " + scale + "----------")
    res, ap = ge.eval_seen(dump_folder, proc=32)
    print("seen")
    print("AP",np.mean(res))
    res = res.transpose(3,0,1,2).reshape(6,-1)
    res = np.mean(res,axis=1)
    print("AP0.4",res[1])
    print("AP0.8",res[3])

    res, ap = ge.eval_similar(dump_folder, proc=32)
    print("similar")
    print("AP",np.mean(res))
    res = res.transpose(3,0,1,2).reshape(6,-1)
    res = np.mean(res,axis=1)
    print("AP0.4",res[1])
    print("AP0.8",res[3])

    res, ap = ge.eval_novel(dump_folder, proc=32)
    print("novel")
    print("AP",np.mean(res))
    res = res.transpose(3,0,1,2).reshape(6,-1)
    res = np.mean(res,axis=1)
    print("AP0.4",res[1])
    print("AP0.8",res[3])

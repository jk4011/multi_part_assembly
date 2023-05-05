import os
import random

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

from torch.utils.data import Dataset, DataLoader
import torch

from knn_cuda import KNN


class GeometryPartDataset(Dataset):
    """Geometry part assembly dataset.

    We follow the data prepared by Breaking Bad dataset:
        https://breaking-bad-dataset.github.io/
    """

    def __init__(
        self,
        data_dir,
        data_fn,
        data_keys,
        category='',
        total_points=50000,
        num_points=1000,
        min_num_part=2,
        max_num_part=20,
        shuffle_parts=False,
        rot_range=-1,
        overfit=-1,
        scale=1,
    ):
        # store parameters
        self.category = category if category.lower() != 'all' else ''
        self.data_dir = data_dir
        self.num_points = num_points
        self.total_points = total_points
        self.min_num_part = min_num_part
        self.max_num_part = max_num_part  # ignore shapes with more parts
        self.shuffle_parts = shuffle_parts  # shuffle part orders
        self.rot_range = rot_range  # rotation range in degree
        self.scale = scale
        
        # list of fracture folder path
        self.data_list = self._read_data(data_fn)
        if overfit > 0:
            self.data_list = self.data_list[:overfit]

        # additional data to load, e.g. ('part_ids', 'instance_label')
        if isinstance(data_keys, str):
            data_keys = [data_keys]
        self.data_keys = data_keys

    def _read_data(self, data_fn):
        """Filter out invalid number of parts."""
        with open(os.path.join(self.data_dir, data_fn), 'r') as f:
            mesh_list = [line.strip() for line in f.readlines()]
            if self.category:
                mesh_list = [
                    line for line in mesh_list
                    if self.category in line.split('/')
                ]
        data_list = []
        for mesh in mesh_list:
            mesh_dir = os.path.join(self.data_dir, mesh)
            if not os.path.isdir(mesh_dir):
                print(f'{mesh} does not exist')
                continue
            for frac in os.listdir(mesh_dir):
                # we take both fractures and modes for training
                if 'fractured' not in frac and 'mode' not in frac:
                    continue
                frac = os.path.join(mesh, frac)
                num_parts = len(os.listdir(os.path.join(self.data_dir, frac)))
                if self.min_num_part <= num_parts <= self.max_num_part:
                    data_list.append(frac)
                    
        return data_list

    @staticmethod
    def _recenter_pc(pc):
        """pc: [N, 3]"""
        centroid = pc.mean(axis=0)
        pc = pc - centroid[None]
        return pc, centroid

    def _rotate_pc(self, pc):
        """pc: [N, 3]"""
        if self.rot_range > 0.:
            rot_euler = (np.random.rand(3) - 0.5) * 2. * self.rot_range
            rot_mat = R.from_euler('xyz', rot_euler, degrees=True).as_matrix()
        else:
            rot_mat = R.random().as_matrix()
        rot_mat = torch.Tensor(rot_mat)
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc, quat_gt

    @staticmethod
    def _shuffle_pc(pc):
        """pc: [N, 3]"""
        order = np.arange(pc.shape[0])
        random.shuffle(order)
        pc = pc[order]
        return pc


    def _knn(self, src, dst, k=1, is_naive=False):
        """return k nearest neighbors using GPU"""
        if len(src) * len(dst) > 10e8:
            # TODO: optimize memory through recursion
            pass

        if not isinstance(src, torch.Tensor):
            src = torch.tensor(src)
        if not isinstance(dst, torch.Tensor):
            dst = torch.tensor(dst)
        
        assert(len(src.shape) == 2)
        assert(len(dst.shape) == 2)
        assert(src.shape[-1] == dst.shape[-1])
        
        src = src.cuda()
        dst = dst.cuda()
        
        if is_naive: 
            src = src.reshape(-1, 1, src.shape[-1])
            distance = torch.norm(src - dst, dim=-1)

            knn = distance.topk(k, largest=False)
            distance = knn.values
            indices = knn.indices
            
        else: # memory efficient
            knn = KNN(k=1, transpose_mode=True)
            distance, indices = knn(dst[None, :], src[None, :]) 
        
        distance = distance.ravel().cpu()
        indices = indices.ravel().cpu()

        return distance, indices

    def _get_broken_pcs_idxs(self, points, threshold=0.01):
        indices = []

        for i in range(len(points)):
            idx_i = torch.zeros(len(points[i]))
            idx_i = idx_i.to(torch.bool)

            for j in range(len(points)):
                if i == j:
                    continue
                distances, _ = self._knn(points[i], points[j])
                idx_i = torch.logical_or(idx_i, distances < threshold)
            indices.append(idx_i)
        
        return indices

    def _get_pcs(self, data_folder):
        """Read mesh and sample point cloud from a folder."""
        # `data_folder`: xxx/plate/1d4093ad2dfad9df24be2e4f911ee4af/fractured_0
        data_folder = os.path.join(self.data_dir, data_folder)
        file_names = os.listdir(data_folder)
        file_names.sort()
        if not self.min_num_part <= len(file_names) <= self.max_num_part:
            raise ValueError

        # shuffle part orders
        if self.shuffle_parts:
            random.shuffle(file_names)
        
        # read mesh and sample points
        meshes = [
            trimesh.load(os.path.join(data_folder, mesh_file))
            for mesh_file in file_names
        ]
        
        # calculate surface area and ratio
        surface_areas = [mesh.area for mesh in meshes]
        total_area = sum(surface_areas)
        pcs_ratios = [area / total_area for area in surface_areas]

        # sample
        pcs = []
        for mesh, ratio in zip(meshes, pcs_ratios):
            num_sample = int(self.total_points * ratio)
            if num_sample < 10:
                num_sample = 10
            samples = trimesh.sample.sample_surface(mesh, num_sample)[0]
            samples = self.scale * samples
            pcs.append(torch.Tensor(samples))
        
        return pcs, file_names

    def get_original_pcs(self, index):
        pcs, _ = self._get_pcs(self.data_list[index])
        return pcs
    
    def __getitem__(self, index):
            
        pcs, file_names = self._get_pcs(self.data_list[index])

        broken_indices = self._get_broken_pcs_idxs(pcs)
        
        # random rotate and translate
        num_parts = len(pcs)
        quat, trans = [], []
        for i in range(num_parts):
            pc = pcs[i]
            
            pc, gt_trans = self._recenter_pc(pc)
            pc, gt_quat = self._rotate_pc(pc)
            quat.append(gt_quat)
            trans.append(gt_trans)
            
            pcs[i] = pc
        
        broken_pcs = [pc[idx] for pc, idx in zip(pcs, broken_indices)]
        
        # shuffle points
        pcs = [self._shuffle_pc(pc) for pc in pcs]
        broken_pcs = [self._shuffle_pc(pc) for pc in broken_pcs]
        
        """
        data_dict = {
            'pcs': MAX_NUM x M_i x 3
                The points sampled from each part.
                
            'broken_pcs': MAX_NUM x N_i x 3
                The points sampled from broken surface.

            'trans': MAX_NUM x 3
                Translation vector

            'quat': MAX_NUM x 4
                Rotation as quaternion.

        'data_id': int
                ID of the data.

        }
        """

        data_dict = {
            'pcs': pcs,
            'broken_pcs': broken_pcs,
            'quat': quat,
            'trans': trans,
            'data_id': index,
            'dir_name': self.data_dir + "/" + self.data_list[index],
            'file_names': file_names,
        }
        
        return data_dict

    def __len__(self):
        return len(self.data_list)


def build_geometry_dataloader(cfg):
    train_set, val_set = build_geometry_dataset(cfg)
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.exp.batch_size,
        shuffle=True,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.exp.num_workers > 0),
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.exp.batch_size * 2,
        shuffle=False,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.exp.num_workers > 0),
    )
    return train_loader, val_loader


def build_geometry_dataset(cfg):
    data_dict = dict(
        data_dir=cfg.data.data_dir,
        data_fn=cfg.data.data_fn.format('train'),
        data_keys=cfg.data.data_keys,
        category=cfg.data.category,
        total_points=cfg.data.total_points,
        min_num_part=cfg.data.min_num_part,
        max_num_part=cfg.data.max_num_part,
        shuffle_parts=cfg.data.shuffle_parts,
        rot_range=cfg.data.rot_range,
        overfit=cfg.data.overfit,
        scale=cfg.data.scale,
    )
    train_set = GeometryPartDataset(**data_dict)

    data_dict['data_fn'] = cfg.data.data_fn.format('val')
    data_dict['shuffle_parts'] = False
    val_set = GeometryPartDataset(**data_dict)
    return train_set, val_set

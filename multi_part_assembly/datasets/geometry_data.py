import os
import random

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

from torch.utils.data import Dataset, DataLoader
import torch

from knn_cuda import KNN
from functools import lru_cache
import jhutil

from copy import copy


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
        self.scale = scale
        self.rot_range = rot_range  # rotation range in degree

        # list of fracture folder path
        self.data_list = self._read_data(data_fn)

        if overfit > 0:
            self.data_list = self.data_list[:overfit]
        # self.data_list = ['artifact/101902_sf/mode_4', 'artifact/101902_sf/mode_2'] + self.data_list

        # additional data to load, e.g. ('part_ids', 'instance_label')
        if isinstance(data_keys, str):
            data_keys = [data_keys]
        self.data_keys = data_keys

        # read rotation_cache
        cache_path = os.path.join(self.data_dir, 'rotation_cache.npy')
        if os.path.isfile(cache_path):
            rotation_cache = np.load(cache_path, allow_pickle=True)
            self.rotation_cache = dict(rotation_cache.item())
        else:
            self.rotation_cache = {}

    def get_length_list(self):
        data_lengs = []
        for data_folder in self.data_list:
            data_folder = os.path.join(self.data_dir, data_folder)
            file_names = os.listdir(data_folder)
            data_lengs.append(len(file_names))
        return data_lengs

    def __exit__(self):
        np.save(os.path.join(self.data_dir, 'rotation_cache.npy'),
                self.rotation_cache)
        # pass

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
                file_names = os.listdir(os.path.join(self.data_dir, frac))
                file_names = [fn for fn in file_names if fn.endswith('.obj')]
                num_parts = len(file_names)
                if self.min_num_part <= num_parts <= self.max_num_part:
                    data_list.append(frac)

        return data_list

    @staticmethod
    def _recenter_pc(pc):
        """pc: [N, 3]"""
        centroid = pc.mean(axis=0)
        pc = pc - centroid[None]
        return pc, centroid

    def _get_rotation_matrix(self, file_name):
        if file_name in self.rotation_cache:
            rot_mat = self.rotation_cache[file_name]
        else:
            if self.rot_range > 0.:
                rot_euler = (np.random.rand(3) - 0.5) * 2. * self.rot_range
                rot_mat = R.from_euler(
                    'xyz', rot_euler, degrees=True).as_matrix()
            else:
                rot_mat = R.random().as_matrix()
            self.rotation_cache[file_name] = rot_mat
        return rot_mat

    def _rotate_pc(self, pc, rot_mat):
        """pc: [N, 3]"""
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

    def _get_overlap_ratio(self, src_mesh, ref_mesh, threshold=0.01):

        src_v = torch.Tensor(src_mesh.vertices)  # (v, 3)
        ref_v = torch.Tensor(ref_mesh.vertices)  # (w, 3)

        distances, _ = jhutil.knn(src_v, ref_v, k=1)
        src_v_is_joint = (distances < threshold).ravel()  # (v, ) ∈ {0, 1}

        src_faces = torch.Tensor(src_mesh.faces)     # (f_i, 3) ∈ {0, ..., v-1}
        src_f_is_joint = torch.zeros(len(src_faces))  # (f_i, ) ∈ {0, 1}
        for i, face_v in enumerate(src_faces):
            face_v_is_joint = src_v_is_joint[face_v]  # (3, ) ∈ {0, 1}
            src_f_is_joint[i] = torch.all(face_v_is_joint)
            
        area_faces = torch.Tensor(src_mesh.area_faces)
        total_area = area_faces.sum()
        joint_area = (area_faces * src_f_is_joint).sum()
        
        overlap_ratio = (joint_area / total_area).item()

        return overlap_ratio

    def _get_broken_pcs_idxs(self, points, threshold=0.01):
        broken_pcs_idxs = []

        for i in range(len(points)):
            idx_i = torch.zeros(len(points[i]))
            idx_i = idx_i.to(torch.bool)

            for j in range(len(points)):
                if i == j:
                    continue
                if not self._box_overlap(points[i], points[j]):
                    continue
                distances, _ = jhutil.knn(points[i], points[j])
                idx_i = torch.logical_or(idx_i, distances < threshold)
            broken_pcs_idxs.append(idx_i)

        return broken_pcs_idxs

    def _get_adjacent_pair(self, points, threshold=0.005):
        """
        Args:
            points : (n, p_i)
        Returns:
            adjacent_pair: (N, 2)
        """
        adjacent_pair = []  # (N, 2)

        for i in range(len(points)):
            idx_i = torch.zeros(len(points[i]))  # (N)
            idx_i = idx_i.to(torch.bool)

            for j in range(len(points)):
                if i <= j:
                    continue
                if not self._box_overlap(points[i], points[j]):
                    continue
                distances, _ = jhutil.knn(points[i], points[j])

                if (distances < threshold).sum() > 20:
                    adjacent_pair.append((i, j))

        return adjacent_pair

    def _box_overlap(self, src, ref):
        # src : (N, 3)
        # ref : (M, 3)
        src_min = src.min(axis=0)[0]  # (3,)
        src_max = src.max(axis=0)[0]  # (3,)
        ref_min = ref.min(axis=0)[0]  # (3,)
        ref_max = ref.max(axis=0)[0]  # (3,)

        # Check x-axis overlap
        if src_max[0] < ref_min[0] or src_min[0] > ref_max[0]:
            return False

        # Check y-axis overlap
        if src_max[1] < ref_min[1] or src_min[1] > ref_max[1]:
            return False

        # Check z-axis overlap
        if src_max[2] < ref_min[2] or src_min[2] > ref_max[2]:
            return False

        return True

    def _get_pcs(self, data_folder):
        """Read mesh and sample point cloud from a folder."""
        # `data_folder`: xxx/plate/1d4093ad2dfad9df24be2e4f911ee4af/fractured_0
        data_folder = os.path.join(self.data_dir, data_folder)
        file_names = os.listdir(data_folder)
        file_names = [fn for fn in file_names if fn.endswith('.obj')]
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
            if num_sample < 100:
                num_sample = 100

            samples = trimesh.sample.sample_surface(mesh, num_sample)[0]
            samples = self.scale * samples
            pcs.append(torch.Tensor(samples))

        return pcs, file_names

    @lru_cache(maxsize=1000)
    def _get_broken_data(self, data_folder):
        """Read mesh and sample point cloud from a folder."""
        # `data_folder`: xxx/plate/1d4093ad2dfad9df24be2e4f911ee4af/fractured_0
        data_folder = os.path.join(self.data_dir, data_folder)
        file_names = os.listdir(data_folder)
        file_names = [fn for fn in file_names if fn.endswith('.obj')]
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
        
        overlap_ratios = torch.zeros(len(meshes), len(meshes))
        for i in range(len(meshes)):
            for j in range(len(meshes)):
                if i == j:
                    overlap_ratios[i][j] = -1
                    continue
                src_v = torch.Tensor(meshes[i].vertices)
                ref_v = torch.Tensor(meshes[j].vertices)
                if not self._box_overlap(src_v, ref_v):
                    continue
                overlap_ratios[i][j] = self._get_overlap_ratio(meshes[i], meshes[j])

        
        
        # calculate surface area and ratio
        surface_areas = [mesh.area for mesh in meshes]
        total_area = sum(surface_areas)
        pcs_ratios = [area / total_area for area in surface_areas]

        # parsing into list
        vertices_list = []  # (N, v_i, 3)
        faces_list = []  # (N, f_i, 3)
        area_faces_list = []  # (N, f_i)
        for mesh in meshes:
            faces = torch.Tensor(mesh.faces)  # (f_i, 3)
            vertices = torch.Tensor(mesh.vertices)  # (v_i, 3)
            area_faces = torch.Tensor(mesh.area_faces)  # (f_i)

            faces_list.append(faces)
            vertices_list.append(vertices)
            area_faces_list.append(area_faces)

        is_pts_broken_list = self._get_broken_pcs_idxs(vertices_list, 0.005)  # (N, v_i)
        adjacent_pair = self._get_adjacent_pair(vertices_list)  # (ap, 2)
        is_face_broken_list = []  # (N, f_i)
        for faces, is_pts_broken, vertices in zip(faces_list, is_pts_broken_list, vertices_list):

            is_face_broken = []  # (f_i, )
            for vertex_idx in faces:
                vertex_idx = vertex_idx.long()
                is_vertex_broken = is_pts_broken[vertex_idx]  # (3, )
                is_face_broken.append(torch.all(is_vertex_broken))
            is_face_broken_list.append(torch.tensor(is_face_broken))

        # if not broken surface, area = 0
        borken_surface_area_list = []  # (N, f_i)
        for is_face_broken, area_faces in zip(is_face_broken_list, area_faces_list):
            borken_surface_area = torch.zeros_like(area_faces)
            borken_surface_area[is_face_broken] = area_faces[is_face_broken]
            borken_surface_area_list.append(borken_surface_area)

        surface_areas = [torch.sum(area_faces)
                         for area_faces in borken_surface_area_list]

        total_area = sum(surface_areas)
        pcs_ratios = [area / total_area for area in surface_areas]

        # sample
        broken_pcs = []  # (N, p_i, 3)
        for mesh, ratio, face_weight in zip(meshes, pcs_ratios, borken_surface_area_list):
            num_sample = int(self.total_points * ratio)
            if num_sample < 10:
                num_sample = 10
            face_weight = face_weight.numpy()
            samples = trimesh.sample.sample_surface(
                mesh, num_sample, face_weight)[0]
            samples = self.scale * samples
            broken_pcs.append(torch.Tensor(samples))
        
                
                

        data = {
            'overlap_ratios': overlap_ratios,  # (N, N)
            'broken_pcs': broken_pcs,  # (N, p_i, 3)
            'file_names': file_names,  # (N, )
            'adjacent_pair': adjacent_pair,  # (ap, 2)
        }
        return data

    def get_original_pcs(self, index):
        pcs, _ = self._get_pcs(self.data_list[index])
        return pcs

    def __getitem__(self, index, sample_from_broken_face=True):
        data_folder = self.data_list[index]

        if sample_from_broken_face:
            pcs = None
            data = self._get_broken_data(data_folder)
            broken_pcs = copy(data["broken_pcs"])
            file_names = copy(data["file_names"])
            adjacent_pair = copy(data["adjacent_pair"])
            overlap_ratios = copy(data["overlap_ratios"])

            num_parts = len(broken_pcs)
            quat, trans = [], []
            for i in range(num_parts):

                file_name = os.path.join(data_folder, file_names[i])
                rot_mat = self._get_rotation_matrix(file_name)

                pc_origin = broken_pcs[i]
                pc, gt_trans = self._recenter_pc(pc_origin)
                pc, gt_quat = self._rotate_pc(pc, rot_mat)
                quat.append(gt_quat)
                trans.append(gt_trans)

                # check the rotation and translation are correct
                pc_recovered = jhutil.quat_trans_transform(gt_quat, gt_trans, pc.double())
                diff = torch.abs(pc_origin - pc_recovered)

                assert torch.all(diff < 1e-5), "all pcs must be recovered within 1e-5"
                assert diff.mean().item() < 1e-6, "mean of diff must be less than 1e-6"

                broken_pcs[i] = pc

        else:
            # TODO: 위에 있는 로직과 비슷하게 변경하기
            pcs, file_names = self._get_pcs(data_folder)

            broken_indices = self._get_broken_pcs_idxs(pcs)

            # random rotate and translate
            num_parts = len(pcs)
            quat, trans = [], []
            for i in range(num_parts):

                file_name = os.path.join(data_folder, file_names[i])
                rot_mat = self._get_rotation_matrix(file_name)

                pc = pcs[i]
                pc, gt_trans = self._recenter_pc(pc)
                pc, gt_quat = self._rotate_pc(pc, rot_mat)
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
            'adjacent_pair': adjacent_pair,
            'overlap_ratios': overlap_ratios,
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


def save_geometry_dataset(cfg):
    train_set, val_set = build_geometry_dataset(cfg)
    train_data, val_data = [], []

    for data in train_set:
        train_data.append(data)
    for data in val_set:
        val_data.append(data)

    train_name = cfg.data.data_fn.format('train').replace('.txt', '.pth')
    val_name = cfg.data.data_fn.format('val').replace('.txt', '.pth')

    train_file_path = os.path.join(cfg.data.data_dir, train_name)
    val_file_path = os.path.join(cfg.data.data_dir, val_name)

    torch.save(train_data, train_file_path)
    jhutil.jhprint(0000, f"data saved at {train_file_path}")
    torch.save(val_data, val_file_path)
    jhutil.jhprint(0000, f"data saved at {val_file_path}")

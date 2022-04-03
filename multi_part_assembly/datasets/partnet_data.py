import os
import numpy as np

from torch.utils.data import Dataset, DataLoader


class PartNetPartDataset(Dataset):
    """PartNet part assembly dataset."""

    def __init__(
        self,
        data_dir,
        data_fn,
        data_keys,
        min_num_part=2,
        max_num_part=20,
        overfit=-1,
    ):
        # store parameters
        self.data_dir = data_dir  # './data'
        self.data_fn = data_fn  # 'Chair.train.npy'
        self.min_num_part = min_num_part
        self.max_num_part = max_num_part  # ignore shapes with more parts
        self.level = 3  # fixed in the paper

        # array of data_idx, [43250,  3069, 37825, 43941, 40503, ...]
        self.shape_ids = np.load(os.path.join(self.data_dir, data_fn))
        if overfit > 0:
            self.shape_ids = self.shape_ids[:overfit]

        # additional data to load, e.g. ('part_ids', 'instance_label')
        self.data_keys = data_keys

    def _rand_another(self):
        """Randomly load another data when current shape has too much parts."""
        index = np.random.choice(len(self))
        return self.__getitem__(index)

    def _pad_data(self, data):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        data = np.array(data)
        pad_shape = (self.max_num_part, ) + tuple(data.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[:data.shape[0]] = data
        return pad_data

    def __getitem__(self, index):
        shape_id = self.shape_ids[index]
        cur_data_fn = os.path.join(self.data_dir, 'shape_data',
                                   f'{shape_id}_level{self.level}.npy')
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()

        # if current shape has too much parts, we randomly load another data
        num_parts = cur_data['part_pcs'].shape[0]  # let's call it `p`
        if num_parts > self.max_num_part or num_parts < self.min_num_part:
            return self._rand_another()
        """
        `cur_data` is dict stored in separate npz files with following keys:
            'part_pcs', 'part_poses', 'part_ids', 'geo_part_ids', 'sym'
        """
        """
        data_dict = {
            'part_pcs': MAX_NUM x N x 3
                The points sampled from each part.

            'part_trans': MAX_NUM x 3
                Translation vector

            'part_quat': MAX_NUM x 4
                Rotation as quaternion.

            'part_valids': MAX_NUM
                1 for shape parts, 0 for padded zeros.

            'data_id': int
                ID of the data.

            'shape_id': int
                ID of the shape.

            'part_ids': MAX_NUM
                Indicator of whether two parts are geometrically equivalent,
                    e.g. [0, 4, 4, 4, 1, 2, 3] means there are 3 same parts.
                If two parts belong to the same category (e.g. chair leg)
                    and have the same bbox size, then they are equivalent.
                This can be used to generate one-hot label to differentiate
                    parts with same geometry as model input

            'instance_label': MAX_NUM x MAX_NUM
                One-hot label to differentiate geometrically equivalent parts.
                If `part_ids` is [0, 4, 4, 4, 1, 2, 3], `instance_label` will be
                    [[1, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0],  # the first instance of `4`
                     [0, 1, 0, 0, 0, 0, 0],  # the second instance of `4`
                     [0, 0, 1, 0, 0, 0, 0],  # the third instance of `4`
                     [1, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0]].
                Say if we extracted per-part feature of shape MAX_NUM x C,
                    we can concat it with `instance_label` and input to final MLP.

            'match_ids': MAX_NUM
                Convert from `part_ids`, label parts that have other equivalent
                    parts in this shape.
                If `part_ids` is [0, 4, 4, 4, 1, 2, 3],
                    `match_ids` will be [0, 1, 1, 1, 0, 0, 0].
                If `part_ids` is [0, 1, 1, 2, 3, 4, 4, 4],
                    `match_ids` will be [0, 1, 1, 0, 0, 2, 2, 2].

            'contact_points': MAX_NUM x MAX_NUM x 4
                Pairwise contact matrix is of shape p x p x 4.
                First item is 1 --> two parts are connecting, 0 otherwise.
                Last three items are the contacting point coordinate.

            'sym': MAX_NUM x 3
                Symmetry label for a part.
                I.e. [0, 0, 0] means the part is not symmetric,
                    [1, 1, 0] means it's symmetric along x and y axis.

        }
        """

        data_dict = {}
        # part point clouds
        cur_pts = cur_data['part_pcs']  # p x N x 3
        data_dict['part_pcs'] = self._pad_data(cur_pts)
        # part poses
        cur_pose = cur_data['part_poses']  # p x (3 + 4)
        cur_pose = self._pad_data(cur_pose)
        data_dict['part_trans'] = cur_pose[:, :3]
        data_dict['part_quat'] = cur_pose[:, 3:]
        # valid part masks
        valids = np.zeros((self.max_num_part), dtype=np.float32)
        valids[:num_parts] = 1.
        data_dict['part_valids'] = valids
        # data_id and shape_id
        data_dict['data_id'] = index
        data_dict['shape_id'] = int(shape_id)

        for key in self.data_keys:
            if key == 'part_ids':
                cur_part_ids = cur_data['geo_part_ids']  # p
                data_dict['part_ids'] = self._pad_data(cur_part_ids)

            elif key == 'instance_label':
                instance_label = np.zeros(
                    (self.max_num_part, self.max_num_part), dtype=np.float32)
                cur_part_ids = cur_data['geo_part_ids']  # p
                num_per_class = [0 for _ in range(max(cur_part_ids) + 1)]
                for j in range(num_parts):
                    cur_class = int(cur_part_ids[j])
                    cur_instance = int(num_per_class[cur_class])
                    instance_label[j, cur_instance] = 1
                    num_per_class[int(cur_part_ids[j])] += 1
                data_dict['instance_label'] = instance_label

            elif key == 'match_ids':
                cur_part_ids = cur_data['geo_part_ids']  # p
                out = self._pad_data(cur_part_ids)
                index = 1
                for i in range(1, int(out.max() + 1)):
                    idx = np.where(out == i)[0]
                    if len(idx) == 0:
                        continue
                    elif len(idx) == 1:
                        out[idx] = 0
                    else:
                        out[idx] = index
                        index += 1
                data_dict['match_ids'] = out

            elif key == 'contact_points':
                # `cur_contacts` is a p x p x 4 contact matrix
                # The first item: 1 --> two parts are connecting, otherwise 0
                # The remaining three items: the contact point coordinate
                cur_contact_data_fn = os.path.join(
                    self.data_dir, 'contact_points',
                    f'pairs_with_contact_points_{shape_id}_level{self.level}.npy'
                )
                cur_contacts = np.load(cur_contact_data_fn, allow_pickle=True)
                out = np.zeros((self.max_num_part, self.max_num_part, 4),
                               dtype=np.float32)
                out[:num_parts, :num_parts] = cur_contacts
                data_dict['contact_points'] = out

            elif key == 'sym':
                cur_sym = cur_data['sym']  # p x 3
                data_dict['sym'] = self._pad_data(cur_sym)

            elif key == 'valid_matrix':
                out = np.zeros((self.max_num_part, self.max_num_part),
                               dtype=np.float32)
                out[:num_parts, :num_parts] = 1.
                data_dict['valid_matrix'] = out

            else:
                raise ValueError(f'ERROR: unknown data {key}!')

        return data_dict

    def __len__(self):
        return len(self.shape_ids)


def build_partnet_dataloader(cfg):
    train_set = PartNetPartDataset(
        data_dir=cfg.data.data_dir,
        data_fn=cfg.data.data_fn.format('train'),
        data_keys=cfg.data.data_keys,
        min_num_part=cfg.data.min_num_part,
        max_num_part=cfg.data.max_num_part,
        overfit=cfg.data.overfit,
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.exp.batch_size,
        shuffle=True,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.exp.num_workers > 0),
    )

    val_set = PartNetPartDataset(
        data_dir=cfg.data.data_dir,
        data_fn=cfg.data.data_fn.format('val'),
        data_keys=cfg.data.data_keys,
        min_num_part=cfg.data.min_num_part,
        max_num_part=cfg.data.max_num_part,
        overfit=cfg.data.overfit,
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

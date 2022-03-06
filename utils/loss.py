import torch

from .quaternion import qrot, qtransform
from .chamfer import chamfer_distance


def _valid_mean(loss_per_part, valids):
    """Average loss values according to the valid parts.

    Args:
        loss_per_part: [B, P]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch, averaged over valid parts
    """
    valids = valids.float().detach()
    loss_per_data = (loss_per_part * valids).sum(1) / valids.sum(1)
    return loss_per_data


def trans_l2_loss(trans1, trans2, valids):
    """L2 loss for transformation.

    Args:
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch
    """
    loss_per_data = (trans1 - trans2).pow(2).sum(dim=-1)  # [B, P]
    loss_per_data = _valid_mean(loss_per_data, valids)
    return loss_per_data


def rot_l2_loss(quat1, quat2, valids):
    """L2 loss for rotation.

    Args:
        quat1: [B, P, 4]
        quat2: [B, P, 4]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch
    """
    # since quat == -quat
    rot_l2_1 = (quat1 - quat2).pow(2).sum(dim=-1)  # [B, P]
    rot_l2_2 = (quat1 + quat2).pow(2).sum(dim=-1)
    loss_per_data = torch.minimum(rot_l2_1, rot_l2_2)
    loss_per_data = _valid_mean(loss_per_data, valids)
    return loss_per_data


def rot_cosine_loss(quat1, quat2, valids):
    """Cosine loss for rotation.

    Args:
        quat1: [B, P, 4]
        quat2: [B, P, 4]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch
    """
    # since quat == -quat
    loss_per_data = 1. - torch.abs(torch.sum(quat1 * quat2, dim=-1))  # [B, P]
    loss_per_data = _valid_mean(loss_per_data, valids)
    return loss_per_data


def rot_points_l2_loss(pts, quat1, quat2, valids):
    """L2 loss between point clouds transformed by quats.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        quat1: [B, P, 4]
        quat2: [B, P, 4]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch
    """
    pts1 = qrot(quat1, pts)
    pts2 = qrot(quat2, pts)

    loss_per_data = (pts1 - pts2).pow(2).sum(-1).mean(-1)  # [B, P]
    loss_per_data = _valid_mean(loss_per_data, valids)
    return loss_per_data


def rot_points_cd_loss(pts, quat1, quat2, valids):
    """Chamfer distance between point clouds transformed by quats.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        quat1: [B, P, 4]
        quat2: [B, P, 4]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch
    """
    batch_size = pts.shape[0]

    pts1 = qrot(quat1, pts)
    pts2 = qrot(quat2, pts)

    dist1, dist2 = chamfer_distance(pts1.flatten(0, 1), pts2.flatten(0, 1))
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    loss_per_data = loss_per_data.view(batch_size, -1).type_as(pts)  # [B, P]
    loss_per_data = _valid_mean(loss_per_data, valids)
    return loss_per_data


def shape_cd_loss(pts, trans1, trans2, quat1, quat2, valids):
    """Chamfer distance between point clouds after rotation and translation.

    Args:
        pts: [B, P, N, 3], model input point cloud to be transformed
        trans1: [B, P, 3]
        trans2: [B, P, 3]
        quat1: [B, P, 4]
        quat2: [B, P, 4]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch
    """
    batch_size = pts.shape[0]
    num_points = pts.shape[2]

    pts1 = qtransform(trans1, quat1, pts)
    pts2 = qtransform(trans2, quat2, pts)

    pts1 = pts1.flatten(1, 2)  # [B, P*N, 3]
    pts2 = pts2.flatten(1, 2)
    # TODO: the padded 0 points may break the chamfer here?
    # TODO: i.e. some points' corresponding points is the padded points
    dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)  # [B, P*N]

    valids = valids.float().detach()
    valids = valids.unsqueeze(2).repeat(1, 1, num_points).view(batch_size, -1)
    dist1 = dist1 * valids
    dist2 = dist2 * valids
    # TODO: should use `_valid_mean` instead of directly taking mean?
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    return loss_per_data
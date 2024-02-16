import torch

def quaternions_to_rotation_matrices(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Args:
        quaternions (torch.Tensor): Tensor with size Kx4, where K is the number of quaternions
            we want to transform to rotation matrices.

    Returns:
        torch.Tensor: Tensor with size Kx3x3, that contains the computed rotation matrices.
    """
    K = quaternions.shape[0]
    # Allocate memory for a Tensor of size Kx3x3 that will hold the rotation
    # matrix along the x-axis
    R = quaternions.new_zeros((K, 3, 3))

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[:, 1] ** 2
    yy = quaternions[:, 2] ** 2
    zz = quaternions[:, 3] ** 2
    ww = quaternions[:, 0] ** 2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = quaternions.new_zeros((K, 1))
    s[n != 0] = 2 / n[n != 0]

    xy = s[:, 0] * quaternions[:, 1] * quaternions[:, 2]
    xz = s[:, 0] * quaternions[:, 1] * quaternions[:, 3]
    yz = s[:, 0] * quaternions[:, 2] * quaternions[:, 3]
    xw = s[:, 0] * quaternions[:, 1] * quaternions[:, 0]
    yw = s[:, 0] * quaternions[:, 2] * quaternions[:, 0]
    zw = s[:, 0] * quaternions[:, 3] * quaternions[:, 0]

    xx = s[:, 0] * xx
    yy = s[:, 0] * yy
    zz = s[:, 0] * zz

    idxs = torch.arange(K).to(quaternions.device)
    R[idxs, 0, 0] = 1 - yy - zz
    R[idxs, 0, 1] = xy - zw
    R[idxs, 0, 2] = xz + yw

    R[idxs, 1, 0] = xy + zw
    R[idxs, 1, 1] = 1 - xx - zz
    R[idxs, 1, 2] = yz - xw

    R[idxs, 2, 0] = xz - yw
    R[idxs, 2, 1] = yz + xw
    R[idxs, 2, 2] = 1 - xx - yy

    return R


def rpy2mat(rpy):
    cosines = torch.cos(rpy)
    sines = torch.sin(rpy)
    cx, cy, cz = cosines.unbind(-1)
    sx, sy, sz = sines.unbind(-1)
    # pyformat: disable
    rotation = torch.stack(
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
         sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
         -sy, cy * sx, cy * cx], dim=-1)
    # pyformat: enable
    rotation = rotation.view(*rotation.shape[:-1], 3, 3)
    return rotation
import torch
import torch.nn as nn


class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def compute_geodesic_distance(self, m1, m2):
        """ Compute the geodesic distance between two rotation matrices.

        Args:
            m1, m2: Two rotation matrices with the shape (batch x 3 x 3).

        Returns:
            The minimal angular difference between two rotation matrices in radian form [0, pi].
        """
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        # cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
        # cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)

        theta = torch.acos(cos)

        return theta
    
    def rot_smooth_loss(self, rotmat):
    
        diff = self.compute_geodesic_distance(rotmat[:, 1:], rotmat[:, :-1])
        vel = diff * 30
        loss = vel ** 2
        loss = 0.5 * torch.mean(loss)
        return loss

    def __call__(self, m1, m2, reduction='mean'):
        loss = self.compute_geodesic_distance(m1, m2)

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        else:
            raise RuntimeError(f'unsupported reduction: {reduction}')


def pos_smooth_loss(pos):
    # minimize delta steps
    diff = (pos[:, 1:] - pos[:, :-1])
    vel = diff * 30
    loss = vel ** 2
    loss = 0.5 * torch.mean(loss)
    return loss

def compute_geodesic_distance(m1, m2):
    """ Compute the geodesic distance between two rotation matrices.

    Args:
        m1, m2: Two rotation matrices with the shape (batch x 3 x 3).

    Returns:
        The minimal angular difference between two rotation matrices in radian form [0, pi].
    """
    batch = m1.shape[:-2]
    m = torch.matmul(m1, m2.transpose(-2, -1))  # batch*3*3

    cos = (m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2] - 1) / 2
    cos = torch.min(cos, torch.ones(*batch).cuda())
    cos = torch.max(cos, torch.ones(*batch).cuda() * -1)
    
    eps = 1e-7
    cos = torch.clamp(cos, -1 + eps, 1 - eps)
        
    theta = torch.acos(cos)
    return theta


def rot_smooth_loss(rotmat):
    
    diff = compute_geodesic_distance(rotmat[:, 1:], rotmat[:, :-1])
    vel = diff * 30
    loss = vel ** 2
    loss = 0.5 * torch.mean(loss)
    return loss


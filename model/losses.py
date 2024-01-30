import torch

def angular_distance(r1, r2, reduction="mean"):
    """https://math.stackexchange.com/questions/90081/quaternion-distance
    https.

    ://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tool
    s.py.

    1 - <q1, q2>^2  <==> (1-cos(theta)) / 2
    """
    assert r1.shape == r2.shape
    if r1.shape[-1] == 4:
        return angular_distance_quat(r1, r2, reduction=reduction)
    if len(r1.shape) == 2 and r1.shape[-1] == 3:  # bs * 3
        return angular_distance_vec(r1, r2, reduction=reduction) 
    else:
        return angular_distance_rot(r1, r2, reduction=reduction) #!!!!!!!!!!!!!!!!!!!!!! 3*3


def angular_distance_quat(pred_q, gt_q, reduction="mean"):
    dist = 1 - torch.pow(torch.bmm(pred_q.view(-1, 1, 4), gt_q.view(-1, 4, 1)), 2)
    if reduction == "mean":
        return dist.mean()
    elif reduction == "sum":
        return dist.sum()
    else:
        return dist


def angular_distance_vec(vec_1, vec_2, reduction="mean"):
    cos = torch.bmm(vec_1.unsqueeze(1), vec_2.unsqueeze(2)).squeeze() / (
        torch.norm(vec_1, dim=1) * torch.norm(vec_2, dim=1)
    )  # [-1, 1]
    dist = (1 - cos) / 2  # [0, 1]
    if reduction == "mean":
        return dist.mean()
    elif reduction == "sum":
        return dist.sum()
    else:
        return dist


def angular_distance_rot(m1, m2, reduction="mean"): #!!!!!!!!!!!!!!!This
    m = torch.bmm(m1, m2.transpose(1, 2))  # b*3*3
    m_trace = torch.einsum("bii->b", m)  # batch trace
    cos = (m_trace - 1) / 2  # [-1, 1]
    # eps = 1e-6
    # cos = torch.clamp(cos, -1+eps, 1-eps)  # avoid nan
    # theta = torch.acos(cos)
    dist = (1 - cos) / 2  # [0, 1]
    if reduction == "mean":
        return dist.mean()
    elif reduction == "sum":
        return dist.sum()
    else:
        return dist

def SmoothL1Dis(p1, p2, threshold=0.1):
    '''
    p1: b*n*3
    p2: b*n*3
    '''
    diff = torch.abs(p1 - p2)
    less = torch.pow(diff, 2) / (2.0 * threshold)
    higher = diff - threshold / 2.0
    dis = torch.where(diff > threshold, higher, less)
    dis = torch.mean(torch.sum(dis, dim=2))
    return dis


def ChamferDis(p1, p2):
    '''
    p1: b*n1*3
    p2: b*n2*3
    '''
    dis = torch.norm(p1.unsqueeze(2) - p2.unsqueeze(1), dim=3)
    dis1 = torch.min(dis, 2)[0]
    dis2 = torch.min(dis, 1)[0]
    dis = 0.5*dis1.mean(1) + 0.5*dis2.mean(1)
    return dis.mean()


def PoseDis(r1, t1, s1, r2, t2, s2):
    '''
    r1, r2: b*3*3
    t1, t2: b*3
    s1, s2: b*3
    '''
    # dis_r = angular_distance(r1, r2, reduction="mean")
    dis_r = torch.mean(torch.norm(r1 - r2, dim=1))
    dis_t = torch.mean(torch.norm(t1 - t2, dim=1))
    dis_s = torch.mean(torch.norm(s1 - s2, dim=1))

    return dis_r + dis_t + dis_s

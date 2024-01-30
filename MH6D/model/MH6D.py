import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import ModifiedResnet, PointNet2MSG, PoseNet
from losses import SmoothL1Dis, ChamferDis, PoseDis
from model.backbone_module import SE
from rotation_utils import Ortho6d2Mat

class Net(nn.Module):
    def __init__(self, nclass=6, nprior=1024):
        super(Net, self).__init__()
        self.nclass = nclass
        self.nprior = nprior

        self.rgb_extractor = ModifiedResnet()
        self.channel_attention_rgb = SE(128)
        self.pts_extractor = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02,0.04], [0.04,0.08], [0.08,0.16]])
        self.channel_attention_pts = SE(128)
        self.prior_extractor = PointNet2MSG(radii_list=[[0.05,0.10], [0.10,0.20], [0.20,0.30], [0.30,0.40]])
        self.channel_attention_prior = SE(128)
        self.posenet = PoseNet(nclass, nprior)

    def forward(self, inputs):
        end_points = {}

        rgb = inputs['rgb']
        pts = inputs['pts']
        choose = inputs['choose']
        prior = inputs['prior']
        cls = inputs['category_label'].reshape(-1)

        c = torch.mean(pts, 1, keepdim=True)
        pts = pts - c

        b = pts.size(0)
        index = cls + torch.arange(b, dtype=torch.long).cuda() * self.nclass

        # rgb feat
        rgb_local = self.rgb_extractor(rgb)
        d = rgb_local.size(1)
        rgb_local = rgb_local.view(b, d, -1)
        choose = choose.unsqueeze(1).repeat(1, d, 1)
        rgb_local = torch.gather(rgb_local, 2, choose).contiguous()
        rgb_local = self.channel_attention_rgb(rgb_local)

        # prior feat
        prior_local = self.prior_extractor(prior)
        prior_local = self.channel_attention_prior(prior_local)

        if self.training:
            delta_t1 = torch.rand(b, 1, 3).cuda()
            delta_t1 = delta_t1.uniform_(-0.02, 0.02)
            delta_r1 = torch.rand(b, 6).cuda()
            delta_r1 = Ortho6d2Mat(delta_r1[:, :3].contiguous(), delta_r1[:, 3:].contiguous()).view(-1,3,3)
            delta_s1 = torch.rand(b, 1).cuda()
            delta_s1 = delta_s1.uniform_(0.8, 1.2)
            pts1 = (pts - delta_t1) / delta_s1.unsqueeze(2) @ delta_r1

            pts1_local = self.pts_extractor(pts1)
            pts1_local = self.channel_attention_pts(pts1_local)
            A1, Qv1, r1, t1, s1 = self.posenet(rgb_local, pts1_local, prior_local, pts1, index)
            end_points['pred_attention1'] = A1
            end_points['pred_qv1'] = Qv1
            end_points['pred_translation1'] = delta_t1.squeeze(1) + delta_s1 * torch.bmm(delta_r1, t1.unsqueeze(2)).squeeze(2) + c.squeeze(1)
            end_points['pred_rotation1'] = delta_r1 @ r1
            end_points['pred_size1'] = s1 * delta_s1

            delta_t2 = torch.rand(b, 1, 3).cuda()
            delta_t2 = delta_t2.uniform_(-0.02, 0.02)
            delta_r2 = torch.rand(b, 6).cuda()
            delta_r2 = Ortho6d2Mat(delta_r2[:, :3].contiguous(), delta_r2[:, 3:].contiguous()).view(-1,3,3)
            delta_s2 = torch.rand(b, 1).cuda()
            delta_s2 = delta_s2.uniform_(0.8, 1.2)
            pts2 = (pts - delta_t2) / delta_s2.unsqueeze(2) @ delta_r2

            pts2_local = self.pts_extractor(pts2)
            pts2_local = self.channel_attention_pts(pts2_local) 
            A2, Qv2, r2, t2, s2 = self.posenet(rgb_local, pts2_local, prior_local, pts2, index)
            end_points['pred_attention2'] = A2
            end_points['pred_qv2'] = Qv2
            end_points['pred_translation2'] = delta_t2.squeeze(1) + delta_s2 * torch.bmm(delta_r2, t2.unsqueeze(2)).squeeze(2) + c.squeeze(1)
            end_points['pred_rotation2'] = delta_r2 @ r2
            end_points['pred_size2'] = s2 * delta_s2

            delta_t3 = torch.rand(b, 1, 3).cuda()
            delta_t3 = delta_t3.uniform_(-0.02, 0.02)
            delta_r3 = torch.rand(b, 6).cuda()
            delta_r3 = Ortho6d2Mat(delta_r3[:, :3].contiguous(), delta_r3[:, 3:].contiguous()).view(-1,3,3)
            delta_s3 = torch.rand(b, 1).cuda()
            delta_s3 = delta_s3.uniform_(0.8, 1.2)
            pts3 = (pts - delta_t3) / delta_s3.unsqueeze(2) @ delta_r3

            pts3_local = self.pts_extractor(pts3)
            pts3_local = self.channel_attention_pts(pts3_local)
            A3, Qv3, r3, t3, s3 = self.posenet(rgb_local, pts3_local, prior_local, pts3, index)
            end_points['pred_attention3'] = A3
            end_points['pred_qv3'] = Qv3
            end_points['pred_translation3'] = delta_t3.squeeze(1) + delta_s3 * torch.bmm(delta_r3, t3.unsqueeze(2)).squeeze(2) + c.squeeze(1)
            end_points['pred_rotation3'] = delta_r3 @ r3
            end_points['pred_size3'] = s3 * delta_s3

        else:
            pts_local = self.pts_extractor(pts)
            pts_local = self.channel_attention_pts(pts_local)
            A, Qv, r, t, s = self.posenet(rgb_local, pts_local, prior_local, pts, index)
            end_points['pred_attention'] = A
            end_points['pred_qv'] = Qv
            end_points['pred_rotation'] = r
            end_points['pred_translation'] = t + c.squeeze(1)
            end_points['pred_size'] = s

        return end_points


class SupervisedLoss(nn.Module):
    def __init__(self, cfg):
        super(SupervisedLoss, self).__init__()
        self.cfg = cfg

    def forward(self, end_points):
        attention1 = end_points['pred_attention1']
        qv1 = end_points['pred_qv1']
        t1 = end_points['pred_translation1']
        r1 = end_points['pred_rotation1']
        s1 = end_points['pred_size1']
        loss1 = self._get_loss(attention1, qv1, r1, t1, s1, end_points)

        attention2 = end_points['pred_attention2']
        qv2 = end_points['pred_qv2']
        t2 = end_points['pred_translation2']
        r2 = end_points['pred_rotation2']
        s2 = end_points['pred_size2']
        loss2 = self._get_loss(attention2, qv2, r2, t2, s2, end_points)

        attention3 = end_points['pred_attention3']
        qv3 = end_points['pred_qv3']
        t3 = end_points['pred_translation3']
        r3 = end_points['pred_rotation3']
        s3 = end_points['pred_size3']
        loss3 = self._get_loss(attention3, qv3, r3, t3, s3, end_points)

        return loss1 + loss2 + loss3
  
    def _get_loss(self, attention, qv, r, t, s, end_points):
        qo = torch.bmm(F.softmax(attention, dim=2), qv)
        loss_qo = SmoothL1Dis(qo, end_points['qo'])
        loss_qv = ChamferDis(qv, end_points['model'])
        loss_pose = PoseDis(r,t,s,end_points['rotation_label'],end_points['translation_label'],end_points['size_label'])

        cfg = self.cfg
        loss = loss_pose + cfg.gamma1 * loss_qv + cfg.gamma2 * loss_qo
        return loss


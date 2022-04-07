import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from einops import rearrange, reduce, repeat
import torch.nn.init as init

from .stepwise_fusion import LSTMStepwiseFusion


class ListenerGenerator(nn.Module):
    def __init__(
        self,
        param,
    ):
        super().__init__()
        generator_cfg = param.model.generator
        self.generator_cfg = generator_cfg
        self.fusion = LSTMStepwiseFusion(
            init_signal_size=70 + 3,
            driven_signal_size=70 + 3 + 45,
            lstm_input_size=generator_cfg.fusion.lstm_input_dim,
            act_layer=nn.Tanh,
            hidden_size=generator_cfg.fusion.lstm.hidden_size,
            num_layers=generator_cfg.fusion.lstm.num_layers,
            bias=generator_cfg.fusion.lstm.bias,
            batch_first=True,
            dropout=generator_cfg.fusion.lstm.dropout,
        )
        self.dynam_3dmm_fc = nn.Linear(generator_cfg.fusion.lstm.hidden_size, generator_cfg.dynam_3dmm_dim + 3)
        self.loss_weights = param.loss_weights
        self.loss_weights['TOTAL_LOSS'] = 1
        self.loss_names = ['TOTAL_LOSS'] + sorted(list(self.loss_weights.keys()))
        self.dynam_3dmm_split = [3, 64, 3, 3]

    def predict(
        self,
        audio,
        driven,
        init,
        lengths,
    ):
        driven = torch.cat((audio, driven), dim=-1)
        output = self.fusion(
            driven,
            init,
            lengths,
        )
        output = self.dynam_3dmm_fc(output)
        return output

    def get_3dmm_loss(self, pred, gt):
        b, t, c = pred.shape
        xpred = pred.view(b * t, c)
        xgt = gt.view(b * t, c)
        pairwise_distance = F.pairwise_distance(xpred, xgt)
        loss = torch.mean(pairwise_distance)
        spiky_loss = self.get_spiky_loss(pred, gt)
        return loss, spiky_loss
    
    def get_spiky_loss(self, pred, gt):
        b, t, c = pred.shape
        pred_spiky = pred[:, 1:, :] - pred[:, :-1, :]
        gt_spiky = gt[:, 1:, :] - gt[:, :-1, :]
        pred_spiky = pred_spiky.view(b * (t - 1), c)
        gt_spiky = gt_spiky.view(b * (t - 1), c)
        pairwise_distance = F.pairwise_distance(pred_spiky, gt_spiky)
        return torch.mean(pairwise_distance)

    def get_loss(
        self,
        pred_3dmm_dynam,
        oth_listener_3dmm_dynam,
    ):
        bs = pred_3dmm_dynam.size(0)
        # angle / exp / trans loss
        pd_angle, pd_exp, pd_trans, pd_crop = torch.split(pred_3dmm_dynam, self.dynam_3dmm_split, dim=-1)
        gt_angle, gt_exp, gt_trans, gt_crop = torch.split(oth_listener_3dmm_dynam, self.dynam_3dmm_split, dim=-1)
        angle_loss, angle_spiky_loss = self.get_3dmm_loss(pd_angle, gt_angle)
        exp_loss, exp_spiky_loss = self.get_3dmm_loss(pd_exp, gt_exp)
        trans_loss, trans_spiky_loss = self.get_3dmm_loss(pd_trans, gt_trans)
        crop_loss, crop_spiky_loss = self.get_3dmm_loss(pd_crop, gt_crop)

        loss = angle_loss       * self.loss_weights['loss_angle'] + \
               angle_spiky_loss * self.loss_weights['loss_angle_spiky'] + \
               exp_loss         * self.loss_weights['loss_exp'] + \
               exp_spiky_loss   * self.loss_weights['loss_exp_spiky'] + \
               trans_loss       * self.loss_weights['loss_trans'] + \
               trans_spiky_loss * self.loss_weights['loss_trans_spiky'] + \
               crop_loss        * self.loss_weights['loss_crop'] + \
               crop_spiky_loss  * self.loss_weights['loss_crop_spiky']

        with torch.no_grad():
            loss_dict = {
                'TOTAL_LOSS': {
                    'val': loss.item(),
                    'n': bs,
                },
                'loss_angle': {
                    'val': angle_loss.item(),
                    'n': bs,
                },
                'loss_angle_spiky': {
                    'val': angle_spiky_loss.item(),
                    'n': bs,
                },
                'loss_exp': {
                    'val': exp_loss.item(),
                    'n': bs,
                },
                'loss_exp_spiky': {
                    'val': exp_spiky_loss.item(),
                    'n': bs,
                },
                'loss_trans': {
                    'val': trans_loss.item(),
                    'n': bs,
                },
                'loss_trans_spiky': {
                    'val': trans_spiky_loss.item(),
                    'n': bs,
                },
                'loss_crop': {
                    'val': crop_loss.item(),
                    'n': bs,
                },
                'loss_crop_spiky': {
                    'val': crop_spiky_loss.item(),
                    'n': bs,
                },
            }
            for key in loss_dict:
                loss_dict[key]['weight'] = self.loss_weights[key]
        return loss, loss_dict

    
    def forward(
        self,
        audio,
        driven,
        init,
        lengths,
        target=None,
    ):
        pred = self.predict(
            audio,
            driven,
            init,
            lengths,
        )

        if target is not None:
            for i in range(audio.size(0)):
                pred[i, lengths[i] - 1:, :] = 0.
            target_expand = torch.zeros_like(pred)
            target_expand[:, :-1, :] = target
            loss, loss_dict = self.get_loss(
                pred,
                target_expand,
            )
            return loss, loss_dict, pred
        return pred

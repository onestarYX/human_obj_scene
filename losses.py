import torch
from torch import nn


class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef * loss


class NerfWLoss(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        b_l: beta loss (2nd term in equation 13)
        s_l: sigma loss (3rd term in equation 13)
    """

    def __init__(self, coef=1, lambda_u=0.01):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u

    def forward(self, inputs, targets, ray_mask):
        ray_mask_sum = ray_mask.sum() + 1e-20
        # if ray_mask_sum < len(inputs['rgb_fine']):
        #     print(ray_mask_sum)

        # print(inputs["transient_accumulation"].shape)

        ret = {}
        ret['c_l'] = 0.5 * (((inputs['rgb_coarse'] - targets) ** 2) * ray_mask[:, None]).sum() / ray_mask_sum
        if 'rgb_fine' in inputs:
            if 'beta' not in inputs:  # no transient head, normal MSE loss
                ret['f_l'] = 0.5 * (((inputs['rgb_fine'] - targets) ** 2) * ray_mask[:, None]).sum() / ray_mask_sum
            else:
                ret['f_l'] = \
                    (((inputs['rgb_fine'] - targets) ** 2 / (2 * inputs['beta'].unsqueeze(1) ** 2)) * ray_mask[:,
                                                                                                      None]).sum() / ray_mask_sum
                ret['b_l'] = 3 + (torch.log(inputs['beta']) * ray_mask).sum() / ray_mask_sum
                ret['s_l'] = self.lambda_u * inputs['transient_sigmas'].mean()

        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret


class NerfletWLoss(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        b_l: beta loss (2nd term in equation 13)
        s_l: sigma loss (3rd term in equation 13)
    """

    def __init__(self, coef=1, lambda_u=0.01):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u

    def forward(self, pred, gt_rgbs, gt_labels, ray_mask, encode_t=True, predict_label=True, loss_pos_ray_ratio=1):
        if loss_pos_ray_ratio != 1:
            assert 0 < loss_pos_ray_ratio <= 1
            original_ones_count = ray_mask.sum()
            zeros_to_flip_count = int(original_ones_count / loss_pos_ray_ratio - original_ones_count)
            zero_indices = torch.nonzero(ray_mask == 0).squeeze()
            if zeros_to_flip_count > len(zero_indices):
                zeros_to_flip_count = len(zero_indices)
            # rand_selected_zero_indices = torch.randint(low=0, high=len(zero_indices), size=(zeros_to_flip_count,)).to(ray_mask.device)
            # rand_selected_zero_indices = zero_indices[rand_selected_zero_indices]
            rand_selected_zero_indices = torch.randperm(len(zero_indices))[:zeros_to_flip_count]
            rand_selected_zero_indices = zero_indices[rand_selected_zero_indices]
            ray_mask[rand_selected_zero_indices] = 1

        ray_mask_sum = ray_mask.sum() + 1e-20
        ret = {}
        if encode_t:
            ret['color_l'] = (((pred['combined_rgb_map'] - gt_rgbs) ** 2 / (2 * pred['beta'].unsqueeze(1) ** 2)) * ray_mask[:, None]).sum() / ray_mask_sum
            ret['beta_l'] = 3 + (torch.log(pred['beta']) * ray_mask).sum() / ray_mask_sum  # TODO: what's the difference between this line here and the paper eqn?
            ret['transient_reg'] = self.lambda_u * pred['transient_occ'].mean()
        else:
            ret['color_l'] = 0.5 * (((pred['static_rgb_map'] - gt_rgbs) ** 2) * ray_mask[:, None]).sum() / ray_mask_sum

        if predict_label:
            if encode_t:
                label_pred = pred['combined_label']
            else:
                label_pred = pred['static_label']
            ret['label_cce'] = torch.nn.functional.cross_entropy(label_pred, gt_labels.to(torch.long).squeeze())

        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret


loss_dict = {'color': ColorLoss,
             'nerfw': NerfWLoss,
             'nerfletw': NerfletWLoss}
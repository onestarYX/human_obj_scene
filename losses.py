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

    def __init__(self, lambda_u=0.01, min_num_rays_per_part=32, max_hitting_parts_per_ray=3,
                 weight_coverage_loss=0.01):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.lambda_u = lambda_u
        self.min_num_rays_per_part = min_num_rays_per_part
        self.max_hitting_parts_per_ray = max_hitting_parts_per_ray
        self.weights = {
            'color_l': 1,
            'beta_l': 1,
            'transient_reg': 1,
            'label_cce': 1,
            'mask_loss': 1,
            'occupancy_loss': 1,
            'occupancy_loss_ell': 0.0001,
            'coverage_loss': weight_coverage_loss,
            'overlap_loss': 0.01,
            # 'occupancy_loss_ell': 1,
            # 'coverage_loss': 1,
            # 'overlap_loss': 1
        }

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
            ret['label_cce'] = torch.nn.functional.cross_entropy(label_pred, gt_labels.to(torch.long))

        # Mask loss
        ret['mask_loss'] = torch.mean((pred['static_mask'] - ray_mask) ** 2)

        # Occupancy loss
        ray_max_occ = pred['static_occ'].max(-1)[0].max(-1)[0]
        ret['occupancy_loss'] = torch.nn.functional.binary_cross_entropy(ray_max_occ, ray_mask.to(torch.float32), reduction='mean')


        # Occupancy loss for ellipsoids
        ray_max_ell_occ = pred['static_ellipsoid_occ'].max(-1)[0].max(-1)[0]
        ret['occupancy_loss_ell'] = torch.nn.functional.binary_cross_entropy(ray_max_ell_occ, ray_mask.to(torch.float32), reduction='mean')

        # Coverage loss
        part_ray_max_occ = pred['static_occ'].max(1)[0]
        part_ray_max_occ_topk = torch.topk(part_ray_max_occ, self.min_num_rays_per_part, dim=0, sorted=False)[0]
        sm = part_ray_max_occ_topk.new_tensor(1e-6)
        ret['coverage_loss'] = -torch.log(part_ray_max_occ_topk + sm).mean()

        # Overlapping loss
        # ray_occ_sum = part_ray_max_occ.sum(-1)
        ray_occ_sum = pred['static_ellipsoid_occ'].max(1)[0].sum(-1)
        zero_tensor = torch.zeros_like(ray_occ_sum)
        ret['overlap_loss'] = torch.maximum(ray_occ_sum - self.max_hitting_parts_per_ray, zero_tensor).mean()


        for k, v in ret.items():
            ret[k] = self.weights[k] * v

        return ret


loss_dict = {'color': ColorLoss,
             'nerfw': NerfWLoss,
             'nerfletw': NerfletWLoss}
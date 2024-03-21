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

    def __init__(self, coef=None, predict_label=True, lambda_u=0.01):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u
        self.predict_label = predict_label

    def forward(self, inputs, gt_rgbs, gt_labels, ray_mask):
        ray_mask_sum = ray_mask.sum() + 1e-20
        # if ray_mask_sum < len(inputs['rgb_fine']):
        #     print(ray_mask_sum)

        # print(inputs["transient_accumulation"].shape)

        ret = {}
        ret['c_l'] = 0.5 * (((inputs['rgb_coarse'] - gt_rgbs) ** 2) * ray_mask).sum() / ray_mask_sum
        if 'rgb_fine' in inputs:
            if 'beta' not in inputs:  # no transient head, normal MSE loss
                ret['f_l'] = 0.5 * (((inputs['rgb_fine'] - gt_rgbs) ** 2) * ray_mask).sum() / ray_mask_sum
            else:
                ret['f_l'] = \
                    (((inputs['rgb_fine'] - gt_rgbs) ** 2 / (2 * inputs['beta'].unsqueeze(1) ** 2)) * ray_mask).sum() / ray_mask_sum
                ret['b_l'] = 3 + (torch.log(inputs['beta']) * ray_mask).sum() / ray_mask_sum
                ret['s_l'] = self.lambda_u * inputs['transient_sigmas'].mean()

        if self.predict_label:
            label_c = inputs['label_coarse']
            ret['cce_coarse'] = torch.nn.functional.cross_entropy(label_c, gt_labels)
            if 'label_fine' in inputs:
                label_f = inputs['label_fine']
                ret['cce_fine'] = torch.nn.functional.cross_entropy(label_f, gt_labels)

        for k, v in ret.items():
            ret[k] = self.coef[k] * v

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
                 loss_weights=None, label_only=False):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.lambda_u = lambda_u
        self.min_num_rays_per_part = min_num_rays_per_part
        self.max_hitting_parts_per_ray = max_hitting_parts_per_ray
        assert loss_weights is not None
        self.weights = loss_weights
        self.label_only = label_only

    def forward(self, pred, gt_rgbs, gt_labels, ray_mask, encode_t=True, predict_label=True,
                use_bg_nerf=False, obj_mask=None, loss_pos_ray_ratio=1):
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

        ret = {}
        if self.label_only:
            label_pred = pred['static_label']
            inside_rays_mask = pred['static_positive_rays_fine']
            if inside_rays_mask.sum() == 0:
                ret['label_cce'] = torch.nn.functional.cross_entropy(label_pred, gt_labels.to(torch.long))
            else:
                ret['label_cce'] = torch.nn.functional.cross_entropy(label_pred[inside_rays_mask],
                                                                     gt_labels[inside_rays_mask].to(torch.long))
            return ret

        bg_mask = ~obj_mask
        if use_bg_nerf and bg_mask.sum() != 0:
            gt_rgbs_obj = gt_rgbs[obj_mask]
            ray_mask_obj = ray_mask[obj_mask]
            gt_labels_obj = gt_labels[obj_mask]
            gt_rgbs_bg = gt_rgbs[bg_mask]
            ray_mask_bg = ray_mask[bg_mask]
            gt_labels_bg = gt_labels[bg_mask]

            ret['color_loss_bg'] = 0.5 * (((pred['rgb_bg'] - gt_rgbs_bg) ** 2) * ray_mask_bg[:, None]).sum() / ray_mask_bg.sum()
            if predict_label:
                ret['label_cce_bg'] = torch.nn.functional.cross_entropy(pred['label_bg'],
                                                                        gt_labels_bg.to(torch.long))

            if obj_mask.sum() == 0:
                for k, v in ret.items():
                    ret[k] = self.weights[k] * v
                return ret
        else:
            gt_rgbs_obj = gt_rgbs
            ray_mask_obj = ray_mask
            gt_labels_obj = gt_labels

        ray_mask_sum = ray_mask_obj.sum() + 1e-20
        # TODO: for now only consider using fine nerf
        if encode_t:
            ret['color_loss'] = (((pred['combined_rgb_map'] - gt_rgbs_obj) ** 2 / (2 * pred['beta'].unsqueeze(1) ** 2)) * ray_mask_obj[:, None]).sum() / ray_mask_sum
            ret['beta_loss'] = 3 + (torch.log(pred['beta']) * ray_mask_obj).sum() / ray_mask_sum  # TODO: what's the difference between this line here and the paper eqn?
            ret['transient_reg'] = self.lambda_u * pred['transient_occ'].mean()
        else:
            # TODO: Which losses should I count for coarse? Is there duplicate gradient given there is duplicated part of input?
            ret['color_loss_c'] = 0.5 * (((pred['static_rgb_map_coarse'] - gt_rgbs_obj) ** 2) * ray_mask_obj[:, None]).sum() / ray_mask_sum
            ret['color_loss_f'] = 0.5 * (((pred['static_rgb_map_fine'] - gt_rgbs_obj) ** 2) * ray_mask_obj[:, None]).sum() / ray_mask_sum

        if predict_label:
            if encode_t:
                label_pred = pred['combined_label']
            else:
                label_pred = pred['static_label']
            inside_rays_mask = pred['static_positive_rays_fine']
            if inside_rays_mask.sum() == 0:
                ret['label_cce'] = torch.tensor(0, device=label_pred.device)
            else:
                ret['label_cce'] = torch.nn.functional.cross_entropy(label_pred[inside_rays_mask],
                                                                     gt_labels_obj[inside_rays_mask].to(torch.long))

        # Mask loss
        ret['mask_loss'] = torch.mean((pred['static_mask_fine'] - ray_mask_obj) ** 2)

        # Occupancy loss
        ray_max_occ = pred['static_occ_fine'].max(-1)[0].max(-1)[0]
        ret['occupancy_loss'] = torch.nn.functional.binary_cross_entropy(ray_max_occ, ray_mask_obj.to(torch.float32), reduction='mean')


        # Occupancy loss for ellipsoids
        ray_max_ell_occ = pred['static_ellipsoid_occ_fine'].max(-1)[0].max(-1)[0]
        ret['occupancy_loss_ell'] = torch.nn.functional.binary_cross_entropy(ray_max_ell_occ, ray_mask_obj.to(torch.float32), reduction='mean')

        # Coverage loss
        part_ray_max_occ = pred['static_occ_fine'].max(1)[0]
        part_ray_max_occ_topk = torch.topk(part_ray_max_occ, self.min_num_rays_per_part, dim=0, sorted=False)[0]
        sm = part_ray_max_occ_topk.new_tensor(1e-6)
        ret['coverage_loss'] = -torch.log(part_ray_max_occ_topk + sm).mean()

        # Overlapping loss
        # ray_occ_sum = part_ray_max_occ.sum(-1)
        ray_occ_sum = pred['static_ellipsoid_occ_fine'].max(1)[0].sum(-1)
        zero_tensor = torch.zeros_like(ray_occ_sum)
        ret['overlap_loss'] = torch.maximum(ray_occ_sum - self.max_hitting_parts_per_ray, zero_tensor).mean()


        for k, v in ret.items():
            ret[k] = self.weights[k] * v

        return ret


class NerfWGarfieldLoss(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        b_l: beta loss (2nd term in equation 13)
        s_l: sigma loss (3rd term in equation 13)
    """

    def __init__(self, coef=None, lambda_u=0.01):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u

    def forward(self, inputs, gt_rgbs, ray_mask):
        ray_mask_sum = ray_mask.sum() + 1e-20
        # if ray_mask_sum < len(inputs['rgb_fine']):
        #     print(ray_mask_sum)

        # print(inputs["transient_accumulation"].shape)

        ret = {}
        ret['c_l'] = 0.5 * (((inputs['rgb_coarse'] - gt_rgbs) ** 2) * ray_mask).sum() / ray_mask_sum
        if 'rgb_fine_combined' in inputs:
            if 'beta' not in inputs:  # no transient head, normal MSE loss
                ret['f_l'] = 0.5 * (((inputs['rgb_fine_combined'] - gt_rgbs) ** 2) * ray_mask).sum() / ray_mask_sum
            else:
                ret['f_l'] = \
                    (((inputs['rgb_fine_combined'] - gt_rgbs) ** 2 / (2 * inputs['beta'].unsqueeze(1) ** 2)) * ray_mask).sum() / ray_mask_sum
                ret['b_l'] = 3 + (torch.log(inputs['beta']) * ray_mask).sum() / ray_mask_sum
                ret['s_l'] = self.lambda_u * inputs['transient_sigmas'].mean()

        for k, v in ret.items():
            ret[k] = self.coef[k] * v

        return ret

loss_dict = {'color': ColorLoss,
             'nerfw': NerfWLoss,
             'nerfletw': NerfletWLoss,
             'nerfw_garfield': NerfWGarfieldLoss}
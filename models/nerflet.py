import torch
from torch import nn
from .nerf import PosEmbedding
from .model_utils import *
import math


class TranslationPredictor(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, use_spread_out_bias=False, bbox=None):
        super().__init__()
        self.use_spread_out_bias = use_spread_out_bias
        self.fc = nn.Sequential(
            # TODO: do we need hidden layers?
            # nn.Linear(in_features=in_channels, out_features=hidden_dim),
            # TODO: might want nomralization?
            # nn.ReLU(),
            nn.Linear(in_features=in_channels, out_features=3)
        )
        if use_spread_out_bias:
            spread_out_bias = torch.rand(3,) - 0.5
            self.fc[0].bias = nn.Parameter(spread_out_bias)
        self.bbox = bbox
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        res = self.fc(x)
        if self.bbox is not None:
            res = self.activation(res)
            min_point = torch.tensor(self.bbox[0], device=x.device, dtype=torch.float32)\
                .unsqueeze(0).expand(res.shape[0], -1)
            max_point = torch.tensor(self.bbox[1], device=x.device, dtype=torch.float32)\
                .unsqueeze(0).expand(res.shape[0], -1)
            res = (max_point - min_point) * res + min_point
        return res


class RotationPredictor(nn.Module):
    def __init__(self, in_channels, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=4)
        )

    def forward(self, x):
        quats = self.fc(x)
        # Apply an L2-normalization non-linearity to enforce the unit norm constrain
        rotations = quats / torch.norm(quats, 2, -1, keepdim=True)
        return rotations


class ScalePredictor(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, min_a=0.05, max_a=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=3)
        )
        self.min_a = min_a
        self.max_a = max_a

    def forward(self, x):
        scale = torch.sigmoid(self.fc(x)) * self.max_a + self.min_a
        return scale


class RayAssociator(nn.Module):
    def __init__(self, occ_threshold=0.5):
        super().__init__()
        self.occ_threshold = occ_threshold

    def forward(self, occ):
        # occ: (num_rays, num_pts_per_ray, M)
        max_occ_per_pt, part_idx_per_pt = occ.max(dim=-1)   # (num_rays, num_pts_per_ray)
        points_in = max_occ_per_pt >= self.occ_threshold
        positive_rays, positive_pt_idx = points_in.max(dim=-1)  # (num_rays,)
        ray_associations = torch.gather(part_idx_per_pt, dim=-1, index=positive_pt_idx[..., None])
        return positive_rays, ray_associations.squeeze()



class Nerflet(nn.Module):
    def __init__(self, D=8, W=256, skips=[4],
                 N_emb_xyz=10, N_emb_dir=4, encode_a=True, encode_t=True,
                 in_channels_a=48, in_channels_t=16,
                 predict_label=True, num_classes=127, beta_min=0.03,
                 M=16, dim_latent=128, scale_min=0.05, scale_max=2, disable_ellipsoid=False,
                 use_spread_out_bias=False, bbox=None, label_only=False, disable_tf=False,
                 sharpness=100, predict_density=False):
        """
        ---Parameters for the original NeRF---
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        beta_min: minimum pixel color variance
        M: number of object-specific NeRF in the scene.
        channels_latent: number of channels for latent codes.
        scale_min/max: scale range of each part
        TODO:
        1. Check whether z_shape is only used for predicting pose of nerflet
        2. Do we need PosEmbedding for xyz and dir? A: yes
        3. Different mlps between part-nerf and nerfw, which one is faster, better?
        4. Do we have the concept of coarse or fine nerf here?  A: extend fine nerf later.
        """
        super().__init__()
        self.D = D
        self.W = W
        self.skips = skips

        self.encode_a = encode_a
        self.encode_t = encode_t
        self.in_channels_a = in_channels_a
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min
        self.predict_label = predict_label
        self.num_classes = num_classes
        self.M = M
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.dim_latent = dim_latent
        self.dim_point_feat = W
        self.embedding_xyz = PosEmbedding(N_emb_xyz - 1, N_emb_xyz)
        self.embedding_dir = PosEmbedding(N_emb_dir - 1, N_emb_dir)
        self.in_channels_xyz = 6 * N_emb_xyz + 3
        self.in_channels_dir = 6 * N_emb_dir + 3
        self.disable_ellipsoid = disable_ellipsoid
        self.bbox = bbox
        self.label_only = label_only
        self.disable_tf = disable_tf
        self.sharpness = sharpness
        self.predict_density = predict_density
        # Sanity checks


        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + self.in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers, which takes the position encoding as part of the input
        self.dir_encoding = nn.Sequential(
                        nn.Linear(W + self.in_channels_dir, W//2), nn.ReLU(True))

        # static output layers
        # static_occ should take xyz_encoding and the shape latent code as input;
        # static_rgb should take dir_encoding, the shape and texture latent codes, and the image-wise appearance
        # embedding as the input (if encode_a is true)
        self.static_occ = nn.Sequential(nn.Linear(W + self.dim_latent, 1))
        if self.encode_a:
            self.static_rgb = nn.Sequential(nn.Linear(W//2 + self.dim_latent * 2 + self.in_channels_a, 3), nn.Sigmoid())
        else:
            self.static_rgb = nn.Sequential(nn.Linear(W // 2 + self.dim_latent * 2, 3), nn.Sigmoid())
        # if self.predict_label:
        #     self.static_label = nn.Linear(W + self.dim_latent, num_classes)

        # Transient output layers
        if self.encode_t:
            self.transient_encoding = nn.Sequential(
                nn.Linear(W + in_channels_t, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True))
            # transient output layers
            self.transient_occ = nn.Sequential(nn.Linear(W // 2, 1), nn.Sigmoid())
            self.transient_rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(W // 2, 1), nn.Softplus())
            if self.predict_label:
                self.transient_label = nn.Linear(W // 2, num_classes)

        # Latent codes for shape and texture
        self.z_shape = nn.Embedding(num_embeddings=self.M, embedding_dim=self.dim_latent)
        self.z_texture = nn.Embedding(num_embeddings=self.M, embedding_dim=self.dim_latent)

        # Label logits
        # self.part_label_logits = nn.Embedding(num_embeddings=self.M, embedding_dim=self.num_classes)
        self.part_label_logits = nn.Parameter(torch.zeros(self.M, self.num_classes))

        # Structure networks (predicting pose of each nerflet)
        # self.translation_predictor = TranslationPredictor(in_channels=dim_latent,
        #                                                   use_spread_out_bias=use_spread_out_bias,
        #                                                   bbox=self.bbox)
        self.translation_predictor = nn.Parameter((torch.rand(self.M, 3) - 0.5) * 0.2)
        # self.rotation_predictor = RotationPredictor(in_channels=dim_latent)
        self.rotation_predictor = nn.Parameter((torch.rand((self.M, 3)) - 0.5) * 2 * math.pi)
        # self.scale_predictor = ScalePredictor(in_channels=dim_latent, min_a=scale_min, max_a=scale_max)
        self.scale_predictor = nn.Parameter((torch.rand(self.M, 3) - 0.5) * 2)

        # Ray associator
        self.ray_associator = RayAssociator()


    def forward(self, pts, dir, a_emb, t_emb):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: the embedded vector of position (+ direction + appearance + transient)
            sigma_only: whether to infer sigma only.

        Outputs (concatenated):
            if sigma_ony:
                static_sigma
            else:
                static_rgb, static_sigma
        """

        '''========================Static predictions========================'''
        # Predict pose for each part nerf
        prediction = {}
        z_shape_ts = self.z_shape(torch.arange(self.M).to(pts.device))  # (M, dim_latent)
        z_texture_ts = self.z_texture(torch.arange(self.M).to(pts.device))

        # rotations = self.rotation_predictor(z_shape_ts)
        rotations = self.rotation_predictor
        prediction['part_rotations'] = rotations.data

        # translations = self.translation_predictor(z_shape_ts)
        translations = self.translation_predictor
        prediction['part_translations'] = translations.data

        # scales = self.scale_predictor(z_shape_ts)
        scales = torch.nn.functional.sigmoid(self.scale_predictor) * self.scale_max
        prediction['part_scales'] = scales.data

        # Perform transformation
        num_rays, num_pts_per_ray, _ = pts.shape
        pts = pts.reshape(-1, 3)
        if self.disable_tf:
            xyz_ = pts.unsqueeze(1).expand(-1, self.M, -1)
        else:
            xyz_ = self.transform_points(pts, translations, rotations)      # (N, M, 3)
        N = xyz_.shape[0]
        if self.disable_ellipsoid:
            ellipsoid_occ = torch.ones(N, self.M, device=xyz_.device)
            points_in_mask = torch.ones(N, self.M, device=xyz_.device).to(torch.bool)
        else:
            # Get per-part nerflet ellipsoid distances (||diag(scale^-1) * point||^2)
            ellipoid_dist = self.compute_point_ellipsoid_dist(xyz_, scales)     # (N, M)
            # Get ellipsoid occupancy (eqn 6 in the paper PartNeRF)
            ellipsoid_occ = self.apply_sigmoid_with_sharpness(1 - ellipoid_dist,
                                                              sharpness_in=self.sharpness, sharpness_out=self.sharpness)
            points_in_mask = ellipoid_dist <= 1

        '''Get static occupancy, pointwise features, and ellipsoid occupancy'''
        # Parallelization by batching over the part dimension for occupancy calculation
        num_points_inside_ellipsoid = points_in_mask.sum(dim=0)
        max_points_inside_ellipsoid = num_points_inside_ellipsoid.max().item()
        if self.predict_density:
            static_occ = torch.zeros((N, self.M), device=xyz_.device)
        else:
            static_occ = torch.zeros((N, self.M), device=xyz_.device) - 100  # -100 to make sure sigmoid is 0
        point_feat = torch.zeros((N, self.M, self.dim_point_feat), device=xyz_.device)
        if max_points_inside_ellipsoid != 0:
            # Keep only the points that are inside the ellipsoid in a batched way (batch over M)
            masked_points = torch.zeros((max_points_inside_ellipsoid, self.M, 3), device=xyz_.device)
            batched_mask = torch.zeros((max_points_inside_ellipsoid, self.M), device=xyz_.device, dtype=torch.bool)
            for i in range(self.M):
                true_idx = points_in_mask[:, i].nonzero().squeeze()
                num_inds = true_idx.numel()
                if num_inds > 0:
                    batched_mask[range(num_inds), i] = points_in_mask[true_idx, i]
                    masked_points[range(num_inds), i, :] = xyz_[true_idx, i, :]

            # Predict occupancy using mlp
            static_pt_emb = self.embedding_xyz(masked_points)     # (max_pts, M, d_emb_pos)
            static_pt_feat = static_pt_emb
            for i in range(self.D):
                if i in self.skips:
                    static_pt_feat = torch.cat([static_pt_emb, static_pt_feat], 2)
                static_pt_feat = getattr(self, f"xyz_encoding_{i + 1}")(static_pt_feat) # (max_pts, M, W)

            static_occ_inputs = z_shape_ts.expand(max_points_inside_ellipsoid, -1, -1) # (max_pts, M, dim_latent)
            static_occ_inputs = torch.cat((static_pt_feat, static_occ_inputs), dim=-1)    # (max_pts, M, W+dim_latent)
            static_occ_pred = self.static_occ(static_occ_inputs).squeeze(-1)  # (max_pts, M)
            # Return the predicted  into its original shape
            for i in range(self.M):
                true_idx = points_in_mask[:, i].nonzero().squeeze()
                num_inds = true_idx.numel()
                if num_inds > 0:
                    static_occ[true_idx, i] = static_occ_pred[range(num_inds), i]
                    point_feat[true_idx, i, :] = static_pt_feat[range(num_inds), i, :]

        if self.predict_density:
            static_occ = torch.nn.functional.sigmoid(static_occ)
        else:
            static_occ = torch.nn.functional.sigmoid(static_occ)
            # Multiply the occupancy of ellipsoid with predicted sigma
            static_occ = ellipsoid_occ * static_occ

        # TODO: think about what's more to add or remove
        prediction['static_ellipsoid_occ'] = ellipsoid_occ.reshape(num_rays, num_pts_per_ray, -1)
        prediction['static_occ'] = static_occ.reshape(num_rays, num_pts_per_ray, -1)
        # prediction['static_point_feat'] = point_feat.reshape(num_rays, num_pts_per_ray, self.M, -1)   # Might not need it
        # prediction['static_points_transformed'] = xyz_.reshape(num_rays, num_pts_per_ray, self.M, 3) # Might not need it
        # prediction['static_points_in_mask'] = points_in_mask.reshape(num_rays, num_pts_per_ray, -1)

        '''Get ray association'''
        if self.predict_density:
            positive_rays = torch.ones(num_rays, dtype=torch.bool, device=xyz_.device)
            ray_associations = torch.zeros(num_rays, dtype=torch.int64, device=xyz_.device)
        else:
            positive_rays, ray_associations = self.ray_associator(
                static_occ.reshape(num_rays, num_pts_per_ray, self.M))
            if positive_rays.dim() == 0:
                positive_rays = positive_rays.unsqueeze(0)
            if ray_associations.dim() == 0:
                ray_associations = ray_associations.unsqueeze(0)
        prediction['static_positive_rays'] = positive_rays
        prediction['static_ray_associations'] = ray_associations

        '''Get static rgb colors'''
        num_inside_rays = int(positive_rays.sum())
        static_rgb_pred = torch.zeros((num_rays, num_pts_per_ray, 3), dtype=torch.float32, device=xyz_.device)
        if num_inside_rays == 0 or self.label_only:
            prediction['static_rgb'] = static_rgb_pred
        else:
            # Get transformed directions
            if self.disable_tf:
                dir_transformed = dir
            else:
                dir_transformed = self.transform_directions(dir, rotations) # (num_rays, M, 3)
            inside_dir_transformed = dir_transformed[positive_rays]    # (num_inside_rays, M, 3)
            inside_ray_associations = ray_associations[positive_rays]
            inside_point_feat = point_feat.reshape(num_rays, num_pts_per_ray, self.M, -1)[positive_rays]
            inside_point_feat = inside_point_feat.reshape(-1, self.M, self.dim_point_feat)
            num_inside_pts = inside_point_feat.shape[0]

            # Select transformed directions, point features, and texture/shape latents based on ray association
            dir_select_idx = inside_ray_associations[..., None, None].expand(-1, -1, 3)  # (num_rays, 1, 3)
            dir_selected = torch.gather(inside_dir_transformed, dim=1, index=dir_select_idx).squeeze(dim=1)    # (num_rays, 3)
            pt_feat_select_idx = inside_ray_associations[..., None, None].expand(-1, num_pts_per_ray, self.dim_point_feat).reshape(-1, 1, self.dim_point_feat) # (N, 1, d_pt_feat)
            pt_feat_selected = torch.gather(inside_point_feat, dim=1, index=pt_feat_select_idx).squeeze(dim=1)     # (N, W)
            z_select_idx = inside_ray_associations[..., None, None].expand(-1, num_pts_per_ray, self.dim_latent).reshape(-1, 1, self.dim_latent) # (N, 1, d_latent)
            z_texture_selected = torch.gather(z_texture_ts.expand(num_inside_pts, -1, -1), dim=1, index=z_select_idx).squeeze(dim=1)    # (N, d_latent)
            z_shape_selected = torch.gather(z_shape_ts.expand(num_inside_pts, -1, -1), dim=1, index=z_select_idx).squeeze(dim=1)

            # Get direction encoding using direction embeddings and point features
            dir_selected_emb = self.embedding_dir(dir_selected) # (num_rays, d_emb_dir)
            dir_selected_emb = dir_selected_emb.unsqueeze(1).expand(-1, num_pts_per_ray, -1).reshape(-1, self.in_channels_dir)   # (N, d_emb_dir)
            dir_selected_encoding = self.dir_encoding(torch.cat((pt_feat_selected, dir_selected_emb), dim=-1))  # (N, W//2)

            # Predict static rgb
            static_rgb_inputs = torch.cat((z_texture_selected, z_shape_selected), dim=-1)   # (N, 2 * dim_latent)
            static_rgb_inputs = torch.cat((static_rgb_inputs, dir_selected_encoding), dim=-1)   # (N, 2*dim_latent + W//2)
            if self.encode_a:
                a_emb_ = a_emb[positive_rays].unsqueeze(1).expand(-1, num_pts_per_ray, -1).reshape(-1, self.in_channels_a) # (N, dim_a_emb)
                static_rgb_inputs = torch.cat((static_rgb_inputs, a_emb_), dim=-1)
            inside_static_rgb_pred = self.static_rgb(static_rgb_inputs)
            inside_static_rgb_pred = inside_static_rgb_pred.reshape(num_inside_rays, num_pts_per_ray, -1)
            static_rgb_pred[positive_rays] = inside_static_rgb_pred
            prediction['static_rgb'] = static_rgb_pred.reshape(num_rays, num_pts_per_ray, -1)

        '''Get static semantics'''
        if self.predict_label:
            static_label_logits = self.part_label_logits[ray_associations]
            if static_label_logits.dim() == 1:
                static_label_logits = static_label_logits.unsqueeze(0)
            static_label_logits[~positive_rays] = 0     # will use positive ray mask to mask out these logits when calculating loss or rendering
            prediction['static_label'] = static_label_logits


        '''========================Transient predictions========================'''
        if self.encode_t:
            transient_pt_emb = self.embedding_xyz(pts)
            transient_pt_feat = transient_pt_emb
            for i in range(self.D):
                if i in self.skips:
                    transient_pt_feat = torch.cat([transient_pt_emb, transient_pt_feat], dim=-1)
                transient_pt_feat = getattr(self, f"xyz_encoding_{i + 1}")(transient_pt_feat)  # (N, W)

            t_emb_ = t_emb.unsqueeze(1).expand(-1, num_pts_per_ray, -1).reshape(-1, self.in_channels_t)
            transient_input = torch.cat((transient_pt_feat, t_emb_), dim=-1)
            transient_encoding = self.transient_encoding(transient_input)
            transient_occ = self.transient_occ(transient_encoding)
            transient_rgb = self.transient_rgb(transient_encoding)
            transient_beta = self.transient_beta(transient_encoding)
            prediction['transient_occ'] = transient_occ.reshape(num_rays, num_pts_per_ray)
            prediction['transient_rgb'] = transient_rgb.reshape(num_rays, num_pts_per_ray, 3)
            prediction['transient_beta'] = transient_beta.reshape(num_rays, num_pts_per_ray)
            if self.predict_label:
                transient_label = self.transient_label(transient_encoding)
                prediction['transient_label'] = transient_label.reshape(num_rays, num_pts_per_ray, -1)

        return prediction


    def transform_points(self, points, translations, rotations):
        # R = quaternions_to_rotation_matrices(rotations)
        # R = R.unsqueeze(0)  # (1, M, 3, 3)
        R = rpy2mat(rotations).unsqueeze(0)
        points = points.unsqueeze(1)    # (N, 1, 3)
        translations = translations.unsqueeze(0)    # (1, M, 3)
        points_transformed = points - translations  # (N, M, 3)
        points_transformed = R.matmul(points_transformed.unsqueeze(-1))    # TODO: check if this is correct (esp. broadcasting)

        points_signs = (points_transformed > 0).float() * 2 - 1
        points_abs = points_transformed.abs()
        points_transformed = points_signs * torch.max(points_abs, points_abs.new_tensor(1e-5))
        return points_transformed.squeeze(-1)

    def transform_directions(self, directions, rotations):
        # R = quaternions_to_rotation_matrices(rotations) # (M, 3, 3)
        # R = R.unsqueeze(0)  # (1, M, 3, 3)
        R = rpy2mat(rotations).unsqueeze(0)
        directions = directions.unsqueeze(1).transpose(2, 3)    # (N, 1, 3, 1)
        directions_transformed = R.matmul(directions)   # (N, M, 3, 1)
        directions_transformed = torch.nn.functional.normalize(directions_transformed.squeeze(-1), dim=-1)
        # TODO: do we need to replace 0 with a small value here?
        return directions_transformed

    def compute_point_ellipsoid_dist(self, points, scales):
        a1 = scales[:, 0].unsqueeze(0)
        a2 = scales[:, 1].unsqueeze(0)
        a3 = scales[:, 2].unsqueeze(0)

        # Basically set zero to 1e-6 in points.
        X = ((points > 0).float() * 2 - 1) * torch.max(torch.abs(points), points.new_tensor(1e-6))
        F = (X[..., 0] / a1) ** 2 + (X[..., 1] / a2) ** 2 + (X[..., 2] / a3) ** 2
        return F

    def apply_sigmoid_with_sharpness(self, x, sharpness_in=100, sharpness_out=100):
        mask_inside = (x > x.new_tensor(0.0)).float()
        x_inside = torch.sigmoid(sharpness_in * x) * mask_inside
        x_out = torch.sigmoid(sharpness_out * x) * (1 - mask_inside)
        x_bar = x_inside + x_out
        return x_bar


class BgNeRF(nn.Module):
    def __init__(self,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
                 predict_label=False, num_classes=80,
                 beta_min=0.03,
                 use_view_dirs=True):
        """
        ---Parameters for the original NeRF---
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_t: number of input channels for t

        ---Parameters for NeRF-W (used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        in_channels_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance
        """
        super().__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.use_view_dirs = use_view_dirs

        # self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.encode_appearance = encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = encode_transient
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min
        self.predict_label = predict_label

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i + 1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        if self.use_view_dirs:
            self.dir_encoding = nn.Sequential(
                nn.Linear(W + in_channels_dir + self.in_channels_a, W // 2), nn.ReLU(True))
        else:
            self.dir_encoding = nn.Sequential(
                nn.Linear(W + self.in_channels_a, W // 2), nn.ReLU(True))

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())
        if self.predict_label:
            self.static_label = nn.Linear(W, num_classes)

        if self.encode_transient:
            # transient encoding layers
            self.transient_encoding = nn.Sequential(
                nn.Linear(W + in_channels_t, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True),
                nn.Linear(W // 2, W // 2), nn.ReLU(True))
            # transient output layers
            self.transient_sigma = nn.Sequential(nn.Linear(W // 2, 1), nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(W // 2, 1), nn.Softplus())
            if self.predict_label:
                self.transient_label = nn.Linear(W // 2, num_classes)

    def forward(self, x, sigma_only=False, output_transient=True):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: the embedded vector of position (+ direction + appearance + transient)
            sigma_only: whether to infer sigma only.
            has_transient: whether to infer the transient component.

        Outputs (concatenated):
            if sigma_ony:
                static_sigma
            elif output_transient:
                static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_rgb, static_sigma
        """
        if sigma_only:
            input_xyz = x
        elif output_transient:
            input_xyz, input_dir_a, input_t = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir + self.in_channels_a,
                                self.in_channels_t], dim=-1)
        else:
            input_xyz, input_dir_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir + self.in_channels_a], dim=-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i + 1}")(xyz_)

        static_sigma = self.static_sigma(xyz_)  # (B, 1)
        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        if self.use_view_dirs:
            dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
            dir_encoding = self.dir_encoding(dir_encoding_input)
            static_rgb = self.static_rgb(dir_encoding)  # (B, 3)
        else:
            dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a[:, self.in_channels_dir:]], 1)
            dir_encoding = self.dir_encoding(dir_encoding_input)
            static_rgb = self.static_rgb(dir_encoding)  # (B, 3)

        if self.predict_label:
            static_label = self.static_label(xyz_)  # (B, num_classes)
            static = torch.cat([static_rgb, static_sigma, static_label], 1)  # (B, 4 + num_classes)
        else:
            static = torch.cat([static_rgb, static_sigma], 1)  # (B, 4)

        if not output_transient:
            return static

        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding)  # (B, 1)
        transient_rgb = self.transient_rgb(transient_encoding)  # (B, 3)
        transient_beta = self.transient_beta(transient_encoding)  # (B, 1)
        if self.predict_label:
            transient_label = self.transient_label(transient_encoding)
            transient = torch.cat([transient_rgb, transient_sigma,
                                   transient_beta, transient_label], 1)  # (B, 5 + num_classes)
        else:
            transient = torch.cat([transient_rgb, transient_sigma,
                                   transient_beta], 1)  # (B, 5)

        return torch.cat([static, transient], 1)  # (B, 9)
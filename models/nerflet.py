import torch
from torch import nn
from nerf import PosEmbedding
from model_utils import quaternions_to_rotation_matrices


class TranslationPredictor(nn.Module):
    def __init__(self, in_channels, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            # TODO: do we need hidden layers?
            # nn.Linear(in_features=in_channels, out_features=hidden_dim),
            # TODO: might want nomralization?
            # nn.ReLU(),
            nn.Linear(in_features=in_channels, out_features=3)
        )

    def forward(self, x):
        return self.fc(x)


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
    def __init__(self):
        pass

    def forward(self, x):
        pass


class Nerflet(nn.Module):
    def __init__(self, D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27, in_channels_a=48,
                 predict_label=True, num_classes=127, beta_min=0.03,
                 M=16, channels_latent=128, scale_min=0.05, scale_max=10):
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
        """
        super().__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_a = in_channels_a
        self.beta_min = beta_min
        self.predict_label = predict_label
        self.M = M
        self.channels_latent = channels_latent

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers TODO: add input channel for z_texture
        self.dir_encoding = nn.Sequential(
                        nn.Linear(W+in_channels_dir+self.in_channels_a, W//2), nn.ReLU(True))

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
        if self.predict_label:
            self.static_label = nn.Linear(W, num_classes)

        # Latent codes for shape and texture
        self.z_shape = nn.Embedding(num_embeddings=self.M, embedding_dim=self.channels_latent)
        self.z_texture = nn.Embedding(num_embeddings=self.M, embedding_dim=self.channels_latent)

        # Structure networks (predicting pose of each nerflet)
        self.translation_predictor = TranslationPredictor(in_channels=channels_latent)
        self.rotation_predictor = RotationPredictor(in_channels=channels_latent)
        self.scale_predictor = ScalePredictor(in_channels=channels_latent, min_a=scale_min, max_a=scale_max)

        # Ray associator
        self.ray_associator = RayAssociator()


    def forward(self, x, sigma_only=False):
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
        if sigma_only:
            input_xyz = x
        else:
            input_xyz, input_dir_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir + self.in_channels_a], dim=-1)

        # Predict pose for each part nerf
        prediction = {}
        rotations = self.rotation_predictor(self.z_shape)
        prediction['rotations'] = rotations
        translations = self.translation_predictor(self.z_shape)
        prediction['translations'] = translations
        scales = self.scale_predictor(self.z_shape)
        prediction['scales'] = scales

        # Perform transformation
        xyz_ = self.transform_points(input_xyz, translations, rotations)



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

        return static


    def transform_points(self, points, translations, rotations):
        R = quaternions_to_rotation_matrices(rotations)
        points_transformed = R.matmul(points - translations)

        points_signs = (points_transformed > 0).float() * 2 - 1
        points_abs = points_transformed.abs()
        points_transformed = points_signs * torch.max(points_abs, points_abs.new_tensor(1e-5))
        return points_transformed

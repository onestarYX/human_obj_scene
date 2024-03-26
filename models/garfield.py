import torch
from torch import nn
from sklearn.preprocessing import QuantileTransformer
import tinycudann as tcnn
from typing import Optional, Union, Tuple
from functorch import jacrev, vmap
import numpy as np

# class GarfieldPredictor(nn.Module):
#     def __init__(self, D=2, W=256):
#         super().__init__()
#         self.D = D
#         self.W = W
#         layers = [nn.Linear(W + 1, W)]
#         for i in range(1, D):
#             layers.append(nn.Linear(W, W))
#         self.garfield_mlp = nn.Sequential(*layers)
#         self.quantile_transformer = None
#
#     def get_quantile_func(self, scales: torch.Tensor, distribution="normal"):
#         """
#         Use 3D scale statistics to normalize scales -- use quantile transformer.
#         """
#         scales = scales.flatten()
#         # scales = scales[(scales > 0) & (scales < self.config.max_grouping_scale)]
#
#         scales = scales.detach().cpu().numpy()
#
#         # Calculate quantile transformer
#         quantile_transformer = QuantileTransformer(output_distribution=distribution)
#         quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))
#
#         def quantile_transformer_func(scales):
#             # This function acts as a wrapper for QuantileTransformer.
#             # QuantileTransformer expects a numpy array, while we have a torch tensor.
#             return torch.Tensor(
#                 quantile_transformer.transform(scales.cpu().numpy())
#             ).to(scales.device)
#
#         self.quantile_transformer = quantile_transformer_func
#
#     def forward(self, x):
#         out = self.garfield_mlp(x)
#         return torch.nn.functional.normalize(out, dim=-1)
#
#     def infer_garfield(self, pt_encodings, weights, scales):
#         scales = scales.unsqueeze(-1)
#         # Quantile transform scales (want it or not?)
#         assert self.quantile_transformer is not None
#         scales = self.quantile_transformer(scales)
#         scales = scales.view(-1, 1, 1).expand(-1, pt_encodings.shape[1], -1)
#         garfield_input = torch.cat([pt_encodings, scales], dim=-1)
#         garfield_out = self.forward(garfield_input)
#         garfield_out = weights.unsqueeze(-1) * garfield_out
#         garfield_out = garfield_out.sum(dim=1)
#         return garfield_out

class Gaussians:
    """Stores Gaussians

    Args:
        mean: Mean of multivariate Gaussian
        cov: Covariance of multivariate Gaussian.
    """
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

class SceneContraction(nn.Module):
    """Contract unbounded space using the contraction was proposed in MipNeRF-360.
        We use the following contraction equation:

        .. math::

            f(x) = \\begin{cases}
                x & ||x|| \\leq 1 \\\\
                (2 - \\frac{1}{||x||})(\\frac{x}{||x||}) & ||x|| > 1
            \\end{cases}

        If the order is not specified, we use the Frobenius norm, this will contract the space to a sphere of
        radius 2. If the order is L_inf (order=float("inf")), we will contract the space to a cube of side length 4.
        If using voxel based encodings such as the Hash encoder, we recommend using the L_inf norm.

        Args:
            order: Order of the norm. Default to the Frobenius norm. Must be set to None for Gaussians.

    """

    def __init__(self, order: Optional[Union[float, int]] = None) -> None:
        super().__init__()
        self.order = order

    def forward(self, positions):
        def contract(x):
            mag = torch.linalg.norm(x, ord=self.order, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

        if isinstance(positions, Gaussians):
            means = contract(positions.mean.clone())

            def contract_gauss(x):
                return (2 - 1 / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True)) * (
                    x / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True)
                )

            jc_means = vmap(jacrev(contract_gauss))(positions.mean.view(-1, positions.mean.shape[-1]))
            jc_means = jc_means.view(list(positions.mean.shape) + [positions.mean.shape[-1]])

            # Only update covariances on positions outside the unit sphere
            mag = positions.mean.norm(dim=-1)
            mask = mag >= 1
            cov = positions.cov.clone()
            cov[mask] = jc_means[mask] @ positions.cov[mask] @ torch.transpose(jc_means[mask], -2, -1)

            return Gaussians(mean=means, cov=cov)

        return contract(positions)


class GarfieldPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_instance_dims: int = 256
        self.hashgrid_cfg = {
            "resolution_range": [(16, 256), (256, 2048)],
            "level": [12, 12],
        }

        self.spatial_distortion: SceneContraction = SceneContraction()

        self.enc_list = torch.nn.ModuleList(
            [
                self._get_encoding(
                    self.hashgrid_cfg["resolution_range"][i], self.hashgrid_cfg["level"][i]
                )
                for i in range(len(self.hashgrid_cfg["level"]))
            ]
        )
        tot_out_dims = sum([e.n_output_dims for e in self.enc_list])

        # This is the MLP that takes the hashgrid encoding as input.
        # Note the +1 for the scale input.
        self.instance_net = tcnn.Network(
            n_input_dims=tot_out_dims + 1,
            n_output_dims=self.n_instance_dims,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 4,
            },
        )

        self.quantile_transformer = None

    @staticmethod
    def _get_encoding(
            res_range: Tuple[int, int], levels: int, indim=3, hash_size=19
    ) -> tcnn.Encoding:
        """
        Helper function to create a HashGrid encoding.
        """
        start_res, end_res = res_range
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))
        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": 8,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc

    def get_quantile_func(self, scales: torch.Tensor, distribution="normal"):
        """
        Use 3D scale statistics to normalize scales -- use quantile transformer.
        """
        scales = scales.flatten()
        # scales = scales[(scales > 0) & (scales < self.config.max_grouping_scale)]

        scales = scales.detach().cpu().numpy()

        # Calculate quantile transformer
        quantile_transformer = QuantileTransformer(output_distribution=distribution)
        quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))

        def quantile_transformer_func(scales):
            # This function acts as a wrapper for QuantileTransformer.
            # QuantileTransformer expects a numpy array, while we have a torch tensor.
            return torch.Tensor(
                quantile_transformer.transform(scales.cpu().numpy())
            ).to(scales.device)

        self.quantile_transformer = quantile_transformer_func

    def get_hash(self, positions):
        """Get the hashgrid encoding. Note that this function does *not* normalize the hash values."""
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        xs = [e(positions.view(-1, 3)) for e in self.enc_list]
        x = torch.concat(xs, dim=-1)
        hash = x.view(*positions.shape[:2], -1)     # (B, N_pts, C)
        return hash

    def get_mlp(self, hash, instance_scales):
        """
        Get the GARField affinity field outputs. Note that this is scale-conditioned.
        This function *does* assume that the hash values are normalized.
        The MLP output is normalized to unit length.
        """
        assert self.quantile_transformer is not None

        # Check that # of rays is the same as # of scales
        assert hash.shape[0] == instance_scales.shape[0]

        epsilon = 1e-5
        scales = instance_scales.contiguous().view(-1, 1)

        # Normalize scales before passing to MLP
        scales = self.quantile_transformer(scales)
        instance_pass = self.instance_net(torch.cat([hash, scales], dim=-1))

        norms = instance_pass.norm(dim=-1, keepdim=True)
        return instance_pass / (norms + epsilon)

    def infer_garfield(self, positions, weights, scales):
        scales = scales.unsqueeze(-1)
        assert self.quantile_transformer is not None
        scales = self.quantile_transformer(scales)

        # Select points with topk weights
        N_pts_per_rays = weights.shape[1]
        topk_weights, topk_indices = torch.topk(weights, N_pts_per_rays // 4, dim=1)
        topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, 3)     # (B, N_pts/4, 3)
        positions_ = torch.gather(positions, 1, topk_indices)    # (B, N_pts/4, 3)
        pos_hash = self.get_hash(positions_)    # (B, N_pts, C)
        hash_rendered = (pos_hash * topk_weights.unsqueeze(-1)).sum(dim=-2)
        # After rendering, normalize features.
        hash_rendered = hash_rendered / torch.linalg.norm(hash_rendered, dim=-1, keepdim=True)
        garfield = self.get_mlp(hash_rendered, scales)

        return garfield
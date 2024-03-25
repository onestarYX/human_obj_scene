import torch
from torch import nn
from sklearn.preprocessing import QuantileTransformer

class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRFW(nn.Module):
    def __init__(self, typ,
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
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.use_view_dirs = use_view_dirs

        # self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.encode_appearance = encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = False if typ == 'coarse' else encode_transient
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


class NeRFWG(nn.Module):
    def __init__(self, typ,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
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
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.use_view_dirs = use_view_dirs

        # self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.encode_appearance = encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = False if typ == 'coarse' else encode_transient
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min

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

    def forward(self, x, output_transient=True):
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
        if output_transient:
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

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        if self.use_view_dirs:
            dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
            dir_encoding = self.dir_encoding(dir_encoding_input)
            static_rgb = self.static_rgb(dir_encoding)  # (B, 3)
        else:
            dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a[:, self.in_channels_dir:]], 1)
            dir_encoding = self.dir_encoding(dir_encoding_input)
            static_rgb = self.static_rgb(dir_encoding)  # (B, 3)

        static = torch.cat([static_rgb, static_sigma], 1)  # (B, 4)

        if not output_transient:
            return static, xyz_

        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding)  # (B, 1)
        transient_rgb = self.transient_rgb(transient_encoding)  # (B, 3)
        transient_beta = self.transient_beta(transient_encoding)  # (B, 1)

        transient = torch.cat([transient_rgb, transient_sigma,
                                transient_beta], 1)  # (B, 5)

        return torch.cat([static, transient], 1), xyz_  # (B, 9)


class GarfieldPredictor(nn.Module):
    def __init__(self, D=2, W=256):
        super().__init__()
        self.D = D
        self.W = W
        layers = [nn.Linear(W + 1, W)]
        for i in range(1, D):
            layers.append(nn.Linear(W, W))
        self.garfield_mlp = nn.Sequential(*layers)
        self.quantile_transformer = None

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

    def forward(self, x):
        out = self.garfield_mlp(x)
        return torch.nn.functional.normalize(out, dim=-1)

    def infer_garfield(self, pt_encodings, weights, scales):
        scales = scales.unsqueeze(-1)
        # Quantile transform scales (want it or not?)
        assert self.quantile_transformer is not None
        scales = self.quantile_transformer(scales)
        scales = scales.view(-1, 1, 1).expand(-1, pt_encodings.shape[1], -1)
        garfield_input = torch.cat([pt_encodings, scales], dim=-1)
        garfield_out = self.forward(garfield_input)
        garfield_out = weights.unsqueeze(-1) * garfield_out
        garfield_out = garfield_out.sum(dim=1)
        return garfield_out
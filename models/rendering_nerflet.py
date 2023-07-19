import torch
from einops import rearrange, reduce, repeat

__all__ = ['render_rays']


def compute_opaqueness_mask(weights, depth_threshold=0.5):
    """Computes a mask which will be 1.0 at the depth point.
    Args:
      weights: the density weights from NeRF.
      depth_threshold: the accumulation threshold which will be used as the depth
        termination point.
    Returns:
      A tensor containing a mask with the same size as weights that has one
        element long the sample dimension that is 1.0. This element is the point
        where the 'surface' is.
    """
    cumulative_contribution = torch.cumsum(weights, axis=-1)
    depth_threshold = torch.tensor(depth_threshold, dtype=weights.dtype)
    opaqueness = cumulative_contribution >= depth_threshold
    false_padding = torch.zeros_like(opaqueness[..., :1])
    padded_opaqueness = torch.cat([false_padding, opaqueness[..., :-1]], axis=-1)
    opaqueness_mask = torch.logical_xor(opaqueness, padded_opaqueness)
    opaqueness_mask = opaqueness_mask.type(weights.dtype)
    return opaqueness_mask


def compute_depth_map(weights, z_vals, depth_threshold=0.5):
    """Compute the depth using the median accumulation.
    Note that this differs from the depth computation in NeRF-W's codebase!
    Args:
      weights: the density weights from NeRF.
      z_vals: the z coordinates of the samples.
      depth_threshold: the accumulation threshold which will be used as the depth
        termination point.
    Returns:
      A tensor containing the depth of each input pixel.
    """
    opaqueness_mask = compute_opaqueness_mask(weights, depth_threshold)
    return torch.sum(opaqueness_mask * z_vals, axis=-1)


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum')  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def shifted_cumprod(x, shift: int = 1) -> torch.Tensor:
    """
    Computes `torch.cumprod(x, dim=-1)` and prepends `shift` number of
    ones and removes `shift` trailing elements to/from the last dimension
    of the result.

    Code adapted from
    https://github.com/facebookresearch/pytorch3d/blob/1701b76a31e3e8c97d51b49dfcaa060762ab3764/pytorch3d/renderer/implicit/raymarching.py#L165
    """
    x_cumprod = torch.cumprod(x, dim=-1)
    x_cumprod_shift = torch.cat(
        [torch.ones_like(x_cumprod[..., :shift]), x_cumprod[..., :-shift]], dim=-1
    )

    return x_cumprod_shift


def render_rays(models,
                embeddings,
                rays,
                ts,
                predict_label,
                num_classes=80,
                N_samples=64,
                use_disp=False,
                N_importance=0,
                chunk=1024 * 32,
                white_back=False,
                test_time=False,
                **kwargs
                ):
    """
    Render rays by computing the output of @model applied on @rays and @ts
    Inputs:
        models: dict of NeRF models (coarse and fine) defined in nerf.py
        embeddings: dict of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3), ray origins and directions
        ts: (N_rays), ray time as embedding index
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """
    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)
    # Embed direction
    # dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d))

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
    if not use_disp:  # use linear sampling in depth space
        z_vals = near * (1 - z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    # TODO: extend coarse and fine NeRF later
    # if N_importance > 0:  # sample points for fine model
    #     z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
    #     z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
    #                          N_importance, det=(perturb == 0))
    #     # detach so that grad doesn't propogate to weights_coarse from here
    #     z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]

    xyz = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')
    a_embedded = embeddings['a'](ts)
    t_embedded = embeddings['t'](ts)
    model = models['nerflet']
    pred = model(xyz, rays_d, a_embedded, t_embedded)

    '''Rendering. We want:
        static: occ, rgb, labels, depth
        transient: occ, rgb, labels, depth, beta
        combined: rgb, labels
    '''
    results = {}
    # Retrieve values
    static_occ = pred['static_occ']
    static_rgb = pred['static_rgb']
    static_labels = pred['static_label']
    static_ray_associations = pred['static_ray_associations']
    positive_rays = pred['static_positive_rays']
    transient_rgb = pred['transient_rgb']
    transient_occ = pred['transient_occ']
    transient_beta = pred['transient_beta']
    transient_labels = pred['transient_label']

    results['static_occ'] = static_occ
    results['transient_occ'] = transient_occ
    results['static_ray_associations'] = static_ray_associations

    # TODO: Might consider just using associated parts to determine occupancy. Here using max to stabilize training
    static_occ = torch.max(static_occ, dim=-1)[0]
    transmittance = shifted_cumprod((1 - static_occ + 1e-10) * (1 - transient_occ + 1e-10))
    # This is the "part" weights for static contents when you render the combined rgb/label/depth maps.
    static_part_weights = static_occ * transmittance
    static_part_weights = static_part_weights * (positive_rays[..., None])
    static_part_rgb_map = torch.sum(static_part_weights[..., None] * static_rgb, dim=1)
    results['static_label'] = static_labels     # TODO: Think about if it makes sense to add this with transient_part_labels below.
    # This is the "part" weights for transient contents
    transient_part_weights = transient_occ * transmittance
    transient_part_rgb_map = torch.sum(transient_part_weights[..., None] * transient_rgb, dim=1)
    transient_part_labels = torch.sum(transient_part_weights[..., None] * transient_labels, dim=1)

    combined_rgb_map = static_part_rgb_map + transient_part_rgb_map
    combined_labels = static_labels + transient_part_labels
    results['combined_rgb_map'] = combined_rgb_map
    results['combined_label'] = combined_labels

    # if test_time:
    # Compute standalone static/transient rgb/depth/label maps
    static_transmittance = shifted_cumprod(1 - static_occ + 1e-10)
    static_weights = static_occ * static_transmittance
    static_weights = static_weights * (positive_rays[..., None])
    static_depth = torch.sum(static_weights * z_vals, dim=1)
    static_rgb_map = torch.sum(static_weights[..., None] * static_rgb, dim=1)

    results['static_depth'] = static_depth
    results['static_rgb_map'] = static_rgb_map


    transient_transmittance = shifted_cumprod(1 - transient_occ + 1e-10)
    transient_weights = transient_occ * transient_transmittance
    transient_depth = torch.sum(transient_weights * z_vals, dim=1)
    transient_ray_beta = torch.sum(transient_weights * transient_beta, dim=1)
    # Add beta_min AFTER the beta composition. Different from eq 10~12 in the paper.
    # See "Notes on differences with the paper" in README.
    transient_ray_beta += model.beta_min
    transient_rgb_map = torch.sum(transient_weights[..., None] * transient_rgb, dim=1)
    transient_ray_labels = torch.sum(transient_weights[..., None] * transient_labels, dim=1)

    results['transient_depth'] = transient_depth
    results['transient_rgb_map'] = transient_rgb_map
    results['transient_label'] = transient_ray_labels
    results['beta'] = transient_ray_beta

    return results

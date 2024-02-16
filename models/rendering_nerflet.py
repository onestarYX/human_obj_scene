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


def render_rays_nerflets(typ, model, embeddings,
                         xyz, rays_d, z_vals, ts,
                         predict_label, white_back=False,
                         predict_density=False, use_associated=False):
    pred = get_nerflet_pred(model, embeddings, xyz, rays_d, ts)

    '''Rendering. We want:
        static: occ, rgb, labels, depth
        transient: occ, rgb, labels, depth, beta
        combined: rgb, labels
    '''
    encode_t = model.encode_t
    results = {}
    # Retrieve values
    if predict_density:
        static_density = pred['static_occ']
        # TODO: this delta shouldn't be using cam's z_vals because we are in the local nerf coordinate?
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e2 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
        deltas = deltas.unsqueeze(-1).expand(-1, -1, model.M)
        # noise = torch.randn_like(static_density)
        noise = 0
        static_occ = 1 - torch.exp(-deltas * torch.relu(static_density + noise))
    else:
        static_occ = pred['static_occ']

    static_rgb = pred['static_rgb']
    static_ray_associations = pred['static_ray_associations']
    positive_rays = pred['static_positive_rays']
    results[f'static_occ_{typ}'] = static_occ
    results[f'static_ray_associations_{typ}'] = static_ray_associations
    results[f'static_positive_rays_{typ}'] = positive_rays
    results[f'static_ellipsoid_occ_{typ}'] = pred['static_ellipsoid_occ']

    # Compute standalone static/transient rgb/depth/label maps
    # TODO: Might consider just using associated parts to determine occupancy. Here using max to stabilize training
    # TODO: think about it!!!
    if use_associated:
        static_ray_associations_ = static_ray_associations[:, None, None].expand(-1, static_occ.shape[1], -1)
        static_occ = torch.gather(static_occ, dim=-1, index=static_ray_associations_).squeeze(-1)
    else:
        static_occ = torch.max(static_occ, dim=-1)[0]
    static_transmittance = shifted_cumprod(1 - static_occ + 1e-10)
    static_weights = static_occ * static_transmittance
    static_weights = static_weights * (positive_rays[..., None])
    static_depth = torch.sum(static_weights * z_vals, dim=1)
    static_rgb_map = torch.sum(static_weights[..., None] * static_rgb, dim=1)
    static_mask = torch.sum(static_weights, dim=1)
    if white_back:
        static_rgb_map += 1 - static_mask.unsqueeze(-1)
    results[f'static_depth_{typ}'] = static_depth
    results[f'static_rgb_map_{typ}'] = static_rgb_map
    results[f'static_weights_{typ}'] = static_weights
    results[f'static_mask_{typ}'] = static_mask

    if predict_label:
        static_labels = pred['static_label']
        results['static_label'] = static_labels  # TODO: Think about if it makes sense to add this with transient_part_labels below.

    if typ == 'fine' and encode_t:
        transient_rgb = pred['transient_rgb']
        transient_occ = pred['transient_occ']
        transient_beta = pred['transient_beta']
        results['transient_occ'] = transient_occ

        transmittance = shifted_cumprod((1 - static_occ + 1e-10) * (1 - transient_occ + 1e-10))
        # This is the "part" weights for static contents when you render the combined rgb/label/depth maps.
        static_part_weights = static_occ * transmittance
        static_part_weights = static_part_weights * (positive_rays[..., None])
        static_part_rgb_map = torch.sum(static_part_weights[..., None] * static_rgb, dim=1)
        # TODO: This is not correct. Should be the "combined" weights sum. How do we define the "combined" weights
        if white_back:
            static_part_rgb_map += 1 - torch.sum(static_part_weights, dim=1).unsqueeze(-1)
        # This is the "part" weights for transient contents
        transient_part_weights = transient_occ * transmittance
        transient_part_rgb_map = torch.sum(transient_part_weights[..., None] * transient_rgb, dim=1)
        combined_rgb_map = static_part_rgb_map + transient_part_rgb_map
        results['combined_rgb_map'] = combined_rgb_map

        if predict_label:
            transient_labels = pred['transient_label']
            transient_part_labels = torch.sum(transient_part_weights[..., None] * transient_labels, dim=1)
            combined_labels = static_labels + transient_part_labels
            results['combined_label'] = combined_labels

        '''Transient-only renderings'''
        # if test_time:
        transient_transmittance = shifted_cumprod(1 - transient_occ + 1e-10)
        transient_weights = transient_occ * transient_transmittance
        transient_depth = torch.sum(transient_weights * z_vals, dim=1)
        transient_ray_beta = torch.sum(transient_weights * transient_beta, dim=1)
        # Add beta_min AFTER the beta composition. Different from eq 10~12 in the paper.
        # See "Notes on differences with the paper" in README.
        transient_ray_beta += model.beta_min
        transient_rgb_map = torch.sum(transient_weights[..., None] * transient_rgb, dim=1)

        results['transient_depth'] = transient_depth
        results['transient_rgb_map'] = transient_rgb_map
        results['beta'] = transient_ray_beta

        if predict_label:
            transient_ray_labels = torch.sum(transient_weights[..., None] * transient_labels, dim=1)
            results['transient_label'] = transient_ray_labels

    return results


def render_rays_bgNerf(model,
                       embeddings,
                       rays,
                       ts,
                       predict_label,
                       num_classes=80,
                       N_samples=64,
                       use_disp=False,
                       perturb=0,
                       noise_std=1,
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

    def inference(results, model, xyz, z_vals, predict_label=False, num_classes=80,
                  test_time=False, validation_version=False, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points on each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        """
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c', c=3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []

        dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
        # create other necessary inputs
        if model.encode_appearance:
            a_embedded_ = repeat(a_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
        if output_transient:
            t_embedded_ = repeat(t_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
        for i in range(0, B, chunk):
            # inputs for original NeRF
            inputs = [embedding_xyz(xyz_[i:i + chunk]), dir_embedded_[i:i + chunk]]
            # additional inputs for NeRF-W
            if model.encode_appearance:
                inputs += [a_embedded_[i:i + chunk]]
            if output_transient:
                inputs += [t_embedded_[i:i + chunk]]
            out_chunks += [model(torch.cat(inputs, 1), output_transient=output_transient)]

        out = torch.cat(out_chunks, 0)
        out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_)
        if predict_label:
            static_rgbs = out[..., :3]  # (N_rays, N_samples_, 3)
            static_sigmas = out[..., 3]  # (N_rays, N_samples_)
            static_labels = out[..., 4:4 + num_classes]  # (N_rays, num_classes)
            if output_transient:
                transient_rgbs = out[..., 4 + num_classes:7 + num_classes]
                transient_sigmas = out[..., 7 + num_classes]
                transient_betas = out[..., 8 + num_classes]
                transient_labels = out[..., 9 + num_classes:]
        else:
            static_rgbs = out[..., :3]  # (N_rays, N_samples_, 3)
            static_sigmas = out[..., 3]  # (N_rays, N_samples_)
            if output_transient:
                transient_rgbs = out[..., 4:7]
                transient_sigmas = out[..., 7]
                transient_betas = out[..., 8]

        # Convert these values using volume rendering
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e2 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        if output_transient:
            static_alphas = 1 - torch.exp(-deltas * static_sigmas)
            transient_alphas = 1 - torch.exp(-deltas * transient_sigmas)
            alphas = 1 - torch.exp(-deltas * (static_sigmas + transient_sigmas))
        else:
            noise = torch.randn_like(static_sigmas) * noise_std
            alphas = 1 - torch.exp(-deltas * torch.relu(static_sigmas + noise))

        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas], -1)  # [1, 1-a1, 1-a2, ...]
        transmittance = torch.cumprod(alphas_shifted[:, :-1], -1)  # [1, 1-a1, (1-a1)(1-a2), ...]

        if output_transient:
            # part of the static/transient weights when computing the combined rgb map (i.e. eqn (8) in nerf-w paper)
            static_weights = static_alphas * transmittance
            transient_weights = transient_alphas * transmittance

        weights = alphas * transmittance
        weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum')

        results[f'weights_bg'] = weights
        results[f'opacity_bg'] = weights_sum
        if output_transient:
            raise NotImplementedError
        else:  # no transient field
            rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1') * static_rgbs,
                             'n1 n2 c -> n1 c', 'sum')
            if white_back:
                rgb_map += 1 - rearrange(weights_sum, 'n -> n 1')
            results[f'rgb_bg'] = rgb_map
            if predict_label:
                label_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1') * static_labels,
                             'n1 n2 c -> n1 c', 'sum')
                results[f'label_bg'] = label_map

        results[f'depth_bg'] = reduce(weights * z_vals, 'n1 n2 -> n1', 'sum')
        return

    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)
    # Embed direction
    dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d))

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    results = {}
    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
    if not use_disp:  # use linear sampling in depth space
        z_vals = near * (1 - z_steps) + far * z_steps
    else:  # use linear sampling in disparity space
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

    if model.encode_appearance:
        if 'a_embedded' in kwargs:
            a_embedded = kwargs['a_embedded']
        else:
            a_embedded = embeddings['a'](ts)
    output_transient = model.encode_transient
    if output_transient:
        if 't_embedded' in kwargs:
            t_embedded = kwargs['t_embedded']
        else:
            t_embedded = embeddings['t'](ts)
    inference(results, model, xyz_coarse, z_vals, predict_label, num_classes, test_time, **kwargs)

    return results



def render_rays(models,
                embeddings,
                rays,
                ts,
                predict_label,
                num_classes=80,
                N_samples=64,
                use_disp=False,
                N_importance=0,
                use_bg_nerf=False,
                obj_mask=None,
                white_back=False,
                predict_density=False,
                use_fine_nerf=False,
                perturb=0,
                use_associated=False,
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
        test_time: whether it is to test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """
    # if test_time:
    #     model.eval()
    # else:
    #     model.train()

    results = {}
    # Decompose the inputs
    if use_bg_nerf:
        assert obj_mask is not None
        obj_rays = rays[obj_mask]
        bg_rays = rays[~obj_mask]
        bg_ts = ts[~obj_mask]
        if bg_rays.shape[0] != 0:
            results.update(render_rays_bgNerf(model=models['bg_nerf'], embeddings=embeddings,
                                              rays=bg_rays, ts=bg_ts, predict_label=predict_label,
                                              num_classes=num_classes, N_samples=N_samples,
                                              use_disp=use_disp, perturb=perturb,
                                              white_back=white_back, test_time=test_time))
        if obj_rays.shape[0] == 0:
            return results
    else:
        obj_rays = rays

    xyz, rays_d, z_vals = get_input_from_rays(obj_rays, N_samples, use_disp, perturb)
    results.update(render_rays_nerflets(typ='coarse', model=models['nerflet'], embeddings=embeddings,
                                        xyz=xyz, rays_d=rays_d, z_vals=z_vals, ts=ts,
                                        predict_label=predict_label, white_back=white_back,
                                        predict_density=predict_density, use_associated=use_associated))

    if use_fine_nerf:
        assert N_importance != 0
        coarse_weights = results['static_weights_coarse'][:, 1:-1].detach()
        xyz_fine, rays_d_fine, z_vals_fine = get_input_from_rays(obj_rays, N_samples, use_disp, perturb,
                                                                 N_importance, z_vals_from_coarse=z_vals,
                                                                 coarse_weights=coarse_weights)

        results.update(render_rays_nerflets(typ='fine', model=models['nerflet'], embeddings=embeddings,
                                            xyz=xyz_fine, rays_d=rays_d_fine, z_vals=z_vals_fine, ts=ts,
                                            predict_label=predict_label, white_back=white_back,
                                            predict_density=predict_density, use_associated=use_associated))

    return results


def compose_nerflet_bgnerf(models,
                           embeddings,
                           rays,
                           ts,
                           predict_label,
                           num_classes=80,
                           N_samples=64,
                           use_disp=False,
                           N_importance=0,
                           white_back=False,
                           predict_density=False,
                           use_fine_nerf=False,
                           perturb=0,
                           use_associated=False,
                           test_time=False,
                           **kwargs):

    raw_results = {}
    results = {}
    # bgnerf
    raw_results.update(render_rays_bgNerf(model=models['bg_nerf'], embeddings=embeddings,
                                      rays=rays, ts=ts, predict_label=predict_label,
                                      num_classes=num_classes, N_samples=N_samples,
                                      use_disp=use_disp, perturb=perturb,
                                      white_back=white_back, test_time=test_time))

    xyz, rays_d, z_vals = get_input_from_rays(rays, N_samples, use_disp, perturb)
    raw_results.update(render_rays_nerflets(typ='coarse', model=models['nerflet'], embeddings=embeddings,
                                        xyz=xyz, rays_d=rays_d, z_vals=z_vals, ts=ts,
                                        predict_label=predict_label, white_back=white_back,
                                        predict_density=predict_density, use_associated=use_associated))

    if use_fine_nerf:
        assert N_importance != 0
        coarse_weights = raw_results['static_weights_coarse'][:, 1:-1].detach()
        xyz_fine, rays_d_fine, z_vals_fine = get_input_from_rays(rays, N_samples, use_disp, perturb,
                                                                 N_importance, z_vals_from_coarse=z_vals,
                                                                 coarse_weights=coarse_weights)

        raw_results.update(render_rays_nerflets(typ='fine', model=models['nerflet'], embeddings=embeddings,
                                            xyz=xyz_fine, rays_d=rays_d_fine, z_vals=z_vals_fine, ts=ts,
                                            predict_label=predict_label, white_back=white_back,
                                            predict_density=predict_density, use_associated=use_associated))


    nerflet_weights = raw_results['static_mask_fine']
    nerflet_rgb = raw_results['static_rgb_map_fine']
    nerflet_label = raw_results['static_label']
    bg_rgb = raw_results['rgb_bg']
    bg_label = raw_results['label_bg']

    comp_rgb = nerflet_weights.unsqueeze(-1) * nerflet_rgb + (1 - nerflet_weights).unsqueeze(-1) * bg_rgb
    results['comp_rgb'] = comp_rgb

    nerflet_mask = nerflet_weights > 0.5
    results['nerflet_mask'] = nerflet_mask
    comp_label = torch.zeros_like(bg_label)
    nerflet_mask_ = nerflet_mask.unsqueeze(-1).expand(-1, num_classes)
    comp_label[nerflet_mask_] = nerflet_label[nerflet_mask_]
    comp_label[~nerflet_mask_] = bg_label[~nerflet_mask_]
    results['comp_label'] = comp_label

    return results


def get_input_from_rays(rays, N_samples, use_disp, perturb, N_importance=0, z_vals_from_coarse=None, coarse_weights=None):
    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    # Sample depth points
    # TODO: maybe for fine, we don't need to inherit z_vals from coarse
    if z_vals_from_coarse is None:
        z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
        if not use_disp:  # use linear sampling in depth space
            z_vals = near * (1 - z_steps) + far * z_steps
        else:  # use linear sampling in disparity space
            z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

        z_vals = z_vals.expand(N_rays, N_samples)

        if perturb > 0:  # perturb sampling depths (z_vals)
            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
            # get intervals between samples
            upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
            lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

            perturb_rand = perturb * torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * perturb_rand
    else:
        z_vals = z_vals_from_coarse

    if N_importance > 0:  # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        z_vals_ = sample_pdf(z_vals_mid, coarse_weights, N_importance, det=(perturb == 0))
        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]

    xyz = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')
    return xyz, rays_d, z_vals


def get_nerflet_pred(model, embeddings, xyz, rays_d, ts):
    a_emb = None
    t_emb = None
    if model.encode_a:
        a_emb = embeddings['a'](ts)
    if model.encode_t:
        t_emb = embeddings['t'](ts)
    pred = model(xyz, rays_d, a_emb, t_emb)
    return pred

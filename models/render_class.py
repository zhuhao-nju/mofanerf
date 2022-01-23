import os
import sys
import time

import imageio

sys.path.append("../..")
from tools.run_nerf_helpers import *
from models.tex_encoder_mod import EnDeUVmap  # network_query_net
from models.model import StyleModule


class lossesLog:
    def __init__(self, lossesList, Weight):
        self.lossesNameList = lossesList
        self.lossesDict = {}
        self.chunkDict = {}
        self.lossesWeight = {}
        self.chunkNum = 0
        for i, name in enumerate(lossesList):
            self.lossesDict[name] = 0
            self.chunkDict[name] = 0
            self.lossesWeight[name] = Weight[i]

    def update(self, lossesList, chunk):
        for name, value in lossesList.items():
            self.lossesDict[name] += torch.sum(value)
            self.chunkDict[name] += chunk

    def out(self):
        loss = 0
        for name, value in self.lossesDict.items():
            if value != 0:
                loss += value / self.chunkDict[name] * self.lossesWeight[name]  # out
            self.lossesDict[name] = 0  # zero
            self.chunkDict[name] = 0
        return loss


class myRenderer(torch.nn.Module):
    def __init__(self, embed_fn=None, embeddirs_fn=None, netchunk=1024 * 64, uvCodesLen=256, expCodesLen=4, input_ch=3,
                 shapeCodes=50):
        super(myRenderer, self).__init__()
        self.embed_fn = embed_fn
        self.embeddirs_fn = embeddirs_fn
        self.netchunk = netchunk
        self.texEncoder = EnDeUVmap(uvCodesLen)
        self.lossList = ["loss_deformReg", "loss_kldiv", "loss_offsets"]
        self.lossWeight = [0.05, 1, 0.01]
        self.lossLog = lossesLog(self.lossList, self.lossWeight)
        self.idSpecificMod = StyleModule()
        self.is_run_fineNet = True
        self.expCodes_Sigma = [
            torch.rand([1, expCodesLen]).cuda()
            for _ in range(20)  # 20 kinds of expression
        ]
        for latent in self.expCodes_Sigma:
            latent.requires_grad = True

    def grad_parameter(self):
        grad_vars = []
        grad_vars += self.expCodes_Sigma
        if self.texEncoder is not None:
            grad_vars += list(self.texEncoder.parameters())
        if self.idSpecificMod is not None:
            grad_vars += list(self.idSpecificMod.parameters())
        return grad_vars

    def run_network(self, inputs, viewdirs, fn=None):
        """Prepares inputs and applies network 'fn'.
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        shapeCodes = self.shapeCodes[0, :].expand([inputs_flat.shape[0], self.shapeCodes.shape[-1]])
        exp_scale, exp_bias = self.idSpecificMod(self.shapeCodes[0, :].reshape(1, -1))
        embedded = inputs_flat
        embedded = self.embed_fn(embedded)

        if self.expCodes_Sigma is not None:
            in_ExpCodes_Sigma = self.expCodes_Sigma[self.expType]
            in_ExpCodes_Sigma = exp_scale * in_ExpCodes_Sigma + exp_bias
            in_ExpCodes_Sigma = in_ExpCodes_Sigma.expand([inputs_flat.shape[0], -1])
        embedded = torch.cat([embedded.cuda(), in_ExpCodes_Sigma.cuda()], -1)  # new! exp15!
        embedded = [embedded]
        embedded.append(shapeCodes)

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape).cuda()
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded.append(embedded_dirs)
        outputs_flat = self.batchify(fn, self.netchunk)(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs

    def batchify(self, fn, chunk):
        """Constructs a version of 'fn' that applies to smaller batches.
        """
        if chunk is None:
            return fn

        def ret(inputs):
            v1, v2, v3 = inputs
            t = self.decoding_texCodes.expand(v1.shape[0], -1)
            out = torch.cat([fn(v1[i:i + chunk], v2[i:i + chunk], v3[i:i + chunk], t[i:i + chunk]) for i in
                             range(0, v1.shape[0], chunk)], 0)
            return out

        return ret

    def batchify_rays(self, chunk=1024 * 32, **kwargs):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        for i in range(0, self.rays.shape[0], chunk):  # traverse all points
            ret = self.render_rays([i, i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def render(self, H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True, shapeCodes=None, uvMap=None, expType=None,
               near=0., far=1.,
               use_viewdirs=False, c2w_staticcam=None,
               **kwargs):
        """Render rays
        ADD idCodes
        ADD uvMap: to trans uv coordinate to corresponding rgb pixel value
        Args:
          H: int. Height of image in pixels.
          W: int. Width of image in pixels.
          focal: float. Focal length of pinhole camera.
          chunk: int. Maximum number of rays to process simultaneously. Used to
            control maximum memory usage. Does not affect final results.
          rays: array of shape [images, batch_size, 3]. Ray origin and direction for
            each example in batch.
          c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
          ndc: bool. If True, represent ray origin, direction in NDC coordinates.
          near: float or array of shape [batch_size]. Nearest distance for a ray.
          far: float or array of shape [batch_size]. Farthest distance for a ray.
          use_viewdirs: bool. If True, use viewing direction of a point in space in model.
          c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
           camera while using other c2w argument for viewing directions.
        Returns:
          rgb_map: [batch_size, 3]. Predicted RGB values for rays.
          disp_map: [batch_size]. Disparity map. Inverse of depth.
          acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
          extras: dict with everything returned by render_rays().
        """

        if c2w is not None:
            rays_o, rays_d = get_rays(H, W, K, c2w)
        else:
            rays_o, rays_d = rays
        if use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1,
                                             keepdim=True)  # uniform it to [0,1], now the object center(WC O) becomes the direction
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape
        if ndc:
            # for forward facing scenes
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)  ##### new
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)
        self.shapeCodes = shapeCodes
        self.rays = rays
        self.uvMap = uvMap
        self.expType = expType
        self.decoding_texCodes, enlosses = self.texEncoder(  # texture map -> texture code
            uvMap.permute([2, 0, 1]).unsqueeze(0), self.lossList)
        self.lossLog.update(enlosses, 1)
        all_ret = self.batchify_rays(chunk, **kwargs)
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'disp_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        if self.lossList is not None:
            ret_dict['losses'] = self.lossLog.out()
        return ret_list + [ret_dict]

    def render_path(self, render_poses, hwf, K, chunk, render_kwargs, uvMap=None, expType=None, gt_imgs=None,
                    savedir=None,
                    render_factor=0, shapeCodes=None, name=None):
        H, W, focal = hwf
        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor
            focal = focal / render_factor
        rgbs = []
        disps = []

        t = time.time()
        if savedir is not None:  # for render final
            filename = os.path.join(savedir, '{}.png'.format(name))
            if os.path.exists(filename):
                print("exists")
                return 0, 0

        for i, c2w in enumerate(render_poses):
            print(i, time.time() - t)
            t = time.time()
            rgb, disp, acc, _ = self.render(H, W, K, chunk=chunk, c2w=c2w[:3, :4],
                                            shapeCodes=shapeCodes[i, :].reshape(1, -1),
                                            uvMap=uvMap[i, :], expType=expType[i], **render_kwargs)
            rgbs.append(rgb.cpu().numpy())
            disps.append(disp.cpu().numpy())
            if savedir is not None:
                rgb8 = to8b(rgbs[-1])
                if name is not None:
                    filename = os.path.join(savedir, '{}.png'.format(name))
                else:
                    filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)

        rgbs = np.stack(rgbs, 0)
        disps = np.stack(disps, 0)

        return rgbs, disps

    def render_rays(self, ray_batch,
                    network_fn,
                    N_samples,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    network_fine=None,
                    white_bkgd=False,
                    raw_noise_std=0.,
                    network_query_fn=None,
                    verbose=False,
                    pytest=False):
        """Volumetric rendering.
        add
        uvMap
        idCodes
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.   Now Add id codes
          network_fn: function. Model for predicting RGB and density at each point
            in space.
          N_samples: int. Number of different times to sample along each ray.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          network_fine: "fine" network with same spec as network_fn.
          white_bkgd: bool. If True, assume a white background.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
          disp_map: [num_rays]. Disparity map. 1 / depth.
          acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
          raw: [num_rays, num_samples, 4]. Raw predictions from model.
          rgb0: See rgb_map. Output for coarse model.
          disp0: See disp_map. Output for coarse model.
          acc0: See acc_map. Output for coarse model.
          z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        ray_batch = self.rays[ray_batch[0]:ray_batch[1]]
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, 8:11] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0., 1., steps=N_samples)
        if not lindisp:  # False
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand  # near to far, with same interval

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        raw = self.run_network(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                     pytest=pytest)

        if N_importance > 0 and self.is_run_fineNet == True:
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals.cuda(), z_samples.cuda()], -1), -1)
            pts = rays_o[..., None, :].cuda() + rays_d[..., None, :].cuda() * z_vals[..., :,
                                                                              None].cuda()  # [N_rays, N_samples + N_importance, 3]

            run_fn = network_fn if network_fine is None else network_fine
            raw = self.run_network(
                inputs=pts, viewdirs=viewdirs, fn=run_fn)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                         pytest=pytest)

        ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
        if retraw:
            ret['raw'] = raw
        if N_importance > 0 and self.is_run_fineNet == True:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        DEBUG = False
        for k in ret:
            if ((torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG):
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret

    def render_fitting(self, H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True, shapeCodes=None, uvCodes=None,
                       expType=20, expCodes=None,
                       near=0., far=1.,
                       use_viewdirs=False, c2w_staticcam=None, network_query_fn=None,
                       **kwargs):
        """Render rays
        ADD idCodes
        ADD uvMap: to trans uv coordinate to corresponding rgb pixel value
        Args:
          H: int. Height of image in pixels.
          W: int. Width of image in pixels.
          focal: float. Focal length of pinhole camera.
          chunk: int. Maximum number of rays to process simultaneously. Used to
            control maximum memory usage. Does not affect final results.
          rays: array of shape [images, batch_size, 3]. Ray origin and direction for
            each example in batch.
          c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
          ndc: bool. If True, represent ray origin, direction in NDC coordinates.
          near: float or array of shape [batch_size]. Nearest distance for a ray.
          far: float or array of shape [batch_size]. Farthest distance for a ray.
          use_viewdirs: bool. If True, use viewing direction of a point in space in model.
          c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
           camera while using other c2w argument for viewing directions.
        Returns:
          rgb_map: [batch_size, 3]. Predicted RGB values for rays.
          disp_map: [batch_size]. Disparity map. Inverse of depth.
          acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
          extras: dict with everything returned by render_rays().
        """
        kwargs['network_fine'].eval()
        kwargs['network_fn'].eval()

        if c2w is not None:
            # special case to render full image
            rays_o, rays_d = get_rays(H, W, K, c2w)

        else:
            # use provided ray batch
            rays_o, rays_d = rays
        if use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1,
                                             keepdim=True)  # uniform it to [0,1], now the object center(WC O) becomes the direction
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        if ndc:
            # for forward facing scenes
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)  ##### new
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        self.shapeCodes = shapeCodes
        self.rays = rays
        self.expType = expType  # exp add for fitting (default 20)
        if self.expCodes_Sigma.__len__() == 20:
            self.expCodes_Sigma.append(expCodes)  # exp add for fitting, requires for grad
        else:
            self.expCodes_Sigma[20] = expCodes
        self.decoding_texCodes = uvCodes
        # Render and reshape  :: input all rays ,output rays results

        all_ret = self.batchify_rays(chunk, **kwargs)
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'disp_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        if self.lossList is not None:
            ret_dict['losses'] = self.lossLog.out()
        return ret_list + [ret_dict]


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw.cuda()) * dists.cuda())
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists.cuda(), torch.Tensor([1e10]).cuda().expand(dists[..., :1].shape)],
                      -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :].cuda(), dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std
        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1. - alpha + 1e-10], -1), -1)[:,
                      :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals.cuda(), -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map

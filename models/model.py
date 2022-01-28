# from run_nerf_helpers.
import torch

torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import numpy as np

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def hook(grad):
    for i in grad:
        if i.max() != 0.:
            print("val: ", i.max())
    print("val: None")


def back_hook(module, grad_in, grad_out):
    for i in grad_in:
        if i.max() != 0.:
            print(module, "grad: ", i.max())
    print("grad: None")


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_textureCodes=10,
                 input_ch_shapeCodes=128, output_ch=4, skips=[4], use_viewdirs=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_shapeCodes = input_ch_shapeCodes
        self.input_ch_textureCodes = input_ch_textureCodes

        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.uv_Width = 4095

        self.xyzEncode = skipMLP(D=3, W=W, input_ch=self.input_ch, skip=None)

        self.linear_BiM_xyz = skipMLP(D=D, W=W, input_ch=self.input_ch_shapeCodes + W, skip=skips[0])
        self.linear_uv_xyzBiM = skipMLP(D=D, W=W, input_ch=self.input_ch_textureCodes + W,
                                        skip=skips[0])
        self.linear_view_xyBMuv = nn.Sequential(
            nn.Linear(self.input_ch_views + W, W // 2),
            nn.ReLU())

        if use_viewdirs:
            # self.feature_linear = nn.Linear(W, W)
            # self.feature_linear.register_backward_hook(back_hook)
            self.alpha_linear = nn.Sequential(nn.Linear(W, 1))
            self.rgb_linear = nn.Linear(W // 2, 3)

        else:
            self.output_linear = nn.Linear(W, output_ch)
        self.initialize()
        # new---
        # self.cat_oneHot_bmCode = nn.Sequential(
        #     nn.Linear(self.input_ch_bmcoding + self.W, self.W), nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU())
        # self.cat_oneHot_texCode = nn.ModuleList([nn.Linear(self.input_ch_uvcoding+self.W, self.W),  nn.Linear(self.W, self.W)])
        # self.cat_h_ = nn.Linear(self.input_ch_uvcoding+self.W, self.W)

    def forward(self, input_pts, input_bmCodes, input_views, input_uvCodes):

        # exp_scale, exp_bias = self.linear_style(input_bmCodes)
        # input_expCodes = exp_scale * input_expCodes + exp_bias
        # h = torch.cat([input_pts, input_expCodes], -1)
        h = input_pts
        xyz_code = self.xyzEncode(h)

        sigmaCodes = self.linear_BiM_xyz(torch.cat([input_bmCodes, xyz_code], dim=1))
        alpha = self.alpha_linear(sigmaCodes)

        rgbCodes = self.linear_uv_xyzBiM(torch.cat([input_uvCodes, sigmaCodes], dim=1))
        rgbCodes = self.linear_view_xyBMuv(torch.cat([input_views, rgbCodes], dim=1))
        rgb = self.rgb_linear(rgbCodes)
        outputs = torch.cat([rgb, alpha], -1)

        return outputs

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


class StyleModule(nn.Module):
    def __init__(self, D=4, W=256, input_ch_bm=50, out_ch=30):
        super(StyleModule, self).__init__()
        self.linears1 = nn.Sequential()

        for i in range(D):
            if i == 0:
                self.linears1.add_module("Linear{}".format(0), nn.Linear(input_ch_bm, W))
                self.linears1.add_module("relu{}".format(0), nn.ReLU())
            else:
                self.linears1.add_module("Linear{}".format(i), nn.Linear(W, W))
                self.linears1.add_module("relu{}".format(i), nn.ReLU())
        self.linears_scale = nn.Linear(W, out_ch)
        self.linears_bias = nn.Linear(W, out_ch)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, bmcodes):
        feature = self.linears1(bmcodes)
        scale = self.linears_scale(feature)
        bias = self.linears_bias(feature)
        return scale, bias


class skipMLP(nn.Module):
    def __init__(self, D=8, W=256, input_ch=256, skip=None):
        super(skipMLP, self).__init__()
        self.skips = skip
        self.linears1 = nn.Sequential()
        self.linears1.add_module("Linear{}".format(0), nn.Linear(input_ch, W))
        self.linears1.add_module("relu{}".format(0), nn.ReLU())
        self.linears2 = nn.Sequential()
        if skip is not None:
            for i in range(self.skips):
                self.linears1.add_module("Linear{}".format(i + 1), nn.Linear(W, W))
                self.linears1.add_module("relu{}".format(i + 1), nn.ReLU())

            self.linears2.add_module("Linear0", nn.Linear(W + input_ch, W))
            self.linears2.add_module("relu0", nn.ReLU())
            for i in range(D - self.skips - 2):
                self.linears2.add_module("Linear{}".format(i + 1), nn.Linear(W, W))
                self.linears2.add_module("relu{}".format(i + 1), nn.ReLU())
        else:
            for i in range(D):
                self.linears1.add_module("Linear{}".format(i + 1), nn.Linear(W, W))
                self.linears1.add_module("relu{}".format(i + 1), nn.ReLU())
        self.initialize()

    def forward(self, x):
        h = self.linears1(x)
        if self.skips is not None:
            h = self.linears2(torch.cat([x, h], dim=1))
        return h

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight.data, std=np.sqrt(1/self.neural_num))    # normal: mean=0, std=1

                # a = np.sqrt(6 / (self.neural_num + self.neural_num))
                #
                # tanh_gain = nn.init.calculate_gain('tanh')
                # a *= tanh_gain
                #
                # nn.init.uniform_(m.weight.data, -a, a)
                #         gainv
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

        # nn.init.normal_(m.weight.data, std=np.sqrt(2 / self.neural_num))


def BilinearSampling(uvMap, coordinate, width=4095):
    def distance(x, y, codn_x, codn_y):
        return (1 - torch.abs(x - codn_x.reshape(x.shape))) * (1 - torch.abs(y - codn_y.reshape(x.shape)))

    # x,y = torch.split(   torch.clip(torch.floor(coordinate.detach()), 0, width-1 ).long(), 1,dim=1)
    x, y = torch.split(torch.clip(torch.div(coordinate, 1, rounding_mode='trunc'), 0, width - 1).long(), 1, dim=1)
    #
    # x = x.detach()
    # y = y.detach()
    # t=uvMap[x.long(),y.long(),:].reshape(-1,3)*distance(x,y,coordinate[:,0],coordinate[:,1])
    rgb = uvMap[x, y, :].reshape(-1, 3) * distance(x.float(), y.float(), coordinate[:, 0], coordinate[:, 1]) + \
          uvMap[x + 1, y, :].reshape(-1, 3) * distance(x.float() + 1, y.float(), coordinate[:, 0], coordinate[:, 1]) + \
          uvMap[x, y + 1, :].reshape(-1, 3) * distance(x.float(), y.float() + 1, coordinate[:, 0], coordinate[:, 1]) + \
          uvMap[x + 1, y + 1, :].reshape(-1, 3) * distance(x.float() + 1, y.float() + 1, coordinate[:, 0],
                                                           coordinate[:, 1])
    return rgb


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)],
                       -1)  # stack,and dirs create in pixel coordinate and use K, convert to camera coordinate
    # Rotate ray directions from camera frame to the world frame  #camera->world
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  #:::this is the same as dirs@c2w[:3,:3].T
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)  # ray center o is the same as camera center
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.images)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, images)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-images)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-images)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

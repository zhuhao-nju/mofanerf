import os

import torch

from models import render_class as render_class
from models.model import *
from tools.logger import Logger


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    new----idcodes
            input_ch_idcodes default10

    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth, input_ch_shapeCodes=args.input_ch_shapeCodes, \
                 input_ch_textureCodes=args.input_ch_textureCodes,
                 input_ch=input_ch + args.input_ch_expCodes, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(args.device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine, input_ch_shapeCodes=args.input_ch_shapeCodes, \
                          input_ch_textureCodes=args.input_ch_textureCodes,
                          input_ch=input_ch + args.input_ch_expCodes, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(args.device)
        grad_vars += list(model_fine.parameters())

    # network_query_fn = lambda inputs, viewdirs, idcodes, network_fn : run_network(inputs, viewdirs, idcodes=idcodes , fn =network_fn,
    #                                                             embed_fn=embed_fn,
    #                                                             embeddirs_fn=embeddirs_fn,
    #                                                             netchunk=args.netchunk)

    # network_query_fn = lambda inputs, viewdirs, idcodes, uvMap, network_fn : run_network(inputs, viewdirs,
    #                                                             idcodes=idcodes, uvMap=uvMap, fn =network_fn,
    #                                                             embed_fn=embed_fn,
    #                                                             embeddirs_fn=embeddirs_fn,
    #                                                             netchunk=args.netchunk)
    render = render_class.myRenderer(embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, netchunk=args.netchunk,
                                     uvCodesLen=args.input_ch_textureCodes, expCodesLen=args.input_ch_expCodes)
    network_query_fn = render.run_network
    grad_vars += list(render.grad_parameter())
    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]  # -1
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if render is not None:
            render.texEncoder.load_state_dict(ckpt['network_render_textureEncoder'])
            render.idSpecificMod.load_state_dict(ckpt['network_render_idSpecific'])
            if render.expCodes_Sigma is not None:
                for latent, saved_latent in zip(
                        render.expCodes_Sigma, ckpt["expression_latent_codes_sigma"]
                ):
                    latent.data[:] = saved_latent[:].detach().clone()
        start = int(ckpt_path.split("/")[-1][:-4])
    # create Model Logger
    logger = Logger("{}/logNew.txt".format(os.path.join(basedir, expname)), bool(~args.no_reload), start)

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, logger, render

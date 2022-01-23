import json

import cv2
from tqdm import tqdm, trange

from models.render_class import *
from tools.config_parser import config_parser
from tools.create_model_condition import create_nerf
from tools.load_facescape import pose_spherical
from tools.run_nerf_helpers import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def load_uvmap(basedir="../data/textureMap300/", personList=None):
    fileList = {}
    for id in personList:
        fileList[id] = basedir + "{}/1_neutral.jpg".format(id)
    return fileList


def load_facescape_data(basedir, half_res=False, testskip=1, personList=None):
    rawShapeCodes = load_bmData()

    basedir = basedir  # "." + basedir
    splits = ['train', 'val', 'test']
    metas = {}
    all_imgs = []  # all images
    all_poses = []  # all poses
    all_idCode = []  # all id number
    all_shapeCodes = []  # shape code
    all_expTypes = []
    counts = [0]  # calculate image number
    for s in splits:
        count_id = 0  # calculate all the id num in training set
        for kk, id in enumerate(personList):
            with open(os.path.join(basedir, 'transforms_{}_{}.json'.format(s, id)), 'r') as fp:
                metas[s] = json.load(fp)
            # for s in splits:
            meta = metas[s]
            imgs = []
            poses = []
            idCodes = []
            expTypes = []
            if s == 'train' or testskip == 0:
                skip = 1
            else:
                skip = testskip
            shapeCodes = rawShapeCodes[int(id)].reshape(1, 50).repeat(len(meta['frames'][::skip]), axis=0)
            # oad img, accoding to meta, 100images for trains, 13vals, 25 tests
            for frame in meta['frames'][::skip]:
                fname = os.path.join(basedir + frame['file_path'] + '.png')
                # imgs.append(imageio.imread(fname))
                imgs.append(fname)
                poses.append(np.array(frame['transform_matrix']))
                idCodes.append(np.long(id))
                expTypes.append(int(frame['expression']))
            # imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
            poses = np.array(poses).astype(np.float32)
            all_imgs.extend(imgs)
            all_poses.append(poses)
            all_idCode.append(idCodes)
            all_shapeCodes.append(shapeCodes)
            all_expTypes.append(expTypes)
            count_id = count_id + len(imgs)  # calculate number of images and sum in id axis
        counts.append(counts[-1] + count_id)  # three number to seperate training / test/ val
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = all_imgs
    poses = np.concatenate(all_poses, 0)
    idCodes = np.concatenate(all_idCode, 0)
    shapeCodes = np.concatenate(all_shapeCodes, 0)
    expCodes = np.concatenate(all_expTypes, 0)
    # read one example images
    imgTmp = imageio.imread(imgs[0])
    H, W = imgTmp.shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack(
        [pose_spherical(angle, 0.0, 800.0 / 50) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
    return imgs, poses, idCodes, shapeCodes, expCodes, render_poses, [H, W, focal], i_split


def readImgFromPath(imgPath, half_res=True, white_bkgd=False, is_uvMap=False):
    imgs = imageio.imread(imgPath)
    imgs = (np.array(imgs) / 255.).astype(np.float32)
    H, W, _ = imgs.shape
    if half_res:
        H = H // 2
        W = W // 2
        imgs_half_res = cv2.resize(imgs, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
    if is_uvMap:
        H_new = 512
        W_new = 512
        if H_new != H:
            imgs = cv2.resize(imgs, (W_new, H_new), interpolation=cv2.INTER_AREA)
    if white_bkgd:
        imgs = imgs[..., :3] * imgs[..., -1:] + (1. - imgs[..., -1:])
    else:
        imgs = imgs[..., :3]
    return torch.Tensor(imgs)


def load_bmData():
    bmModel = np.load('../data/factors_id.npy')
    return bmModel


class LMModule:
    def __init__(self, H=None):
        self.landmark = np.load("../data/1_975_landmarks.npy")
        self.H = H

    def sample_point(self, numOfPoint=None, K=None, pose=None, id=None, exp=None, coords=None):
        if exp == None:
            exp = 0
        pose = pose.cpu().numpy()
        id = int(id.item())
        lm3d = self.landmark[id, exp, :, :] / 50.
        Rt = np.eye(4)
        M = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        Rt[:3, :3] = pose[:3, :3].T
        Rt[:3, 3] = -pose[:3, :3].T.dot(pose[:3, 3])  # .T
        # project 3d to 2d
        lm2d = K @ Rt[:3, :] @ (np.concatenate([lm3d, np.ones([lm3d.shape[0], 1])], 1).T)
        lm2d_half = lm2d // lm2d[2, :]
        # rot the image
        lm2d = np.round(lm2d_half).astype(np.long)[:2, :].T @ M[:2, :2]  # .T[:,:2]  #[68,2]

        p = np.long(numOfPoint / 5 * 3 // 68)
        wid = self.H * 0.025
        rand = np.random.randn(p, 2) * wid
        sampleLandMark = lm2d[:, None, :].repeat(p, 1) + rand[None, :, :].repeat(68, 0)
        sampleLandMark = sampleLandMark.reshape(-1, 2).astype(np.int)
        sampleUniform = np.random.choice(coords.shape[0], size=[numOfPoint - sampleLandMark.shape[0]], replace=False)
        sampleUniform = coords[sampleUniform]

        return torch.cat([sampleUniform, torch.Tensor(sampleLandMark).cuda()], 0).long()


def getValidPerson(datadir):
    t = os.listdir(datadir)  #
    tt = sorted(t)
    tt.sort(key=len)  # sort by length
    t1 = tt[:359]
    invalidPerList = ['39', '52', '69', '295', '307', '413', '417', '587', '237', '353', '356', '440',
                      '363']  # need reupload
    changeId = ['615', '616', '619', '620', '622', '623', '624', '626', '627', '722', '725', '728', '733', '734']
    for i, invalidPer in enumerate(invalidPerList):
        id = t1.index(invalidPer)
        t1[id] = changeId[i]
    return t1


def train():
    parser = config_parser()
    args = parser.parse_args()
    validPerson = getValidPerson(args.datadir)
    args.device = device
    if args.personList is not None:
        args.personList = args.personList.split(",")
        args.person_num = len(args.personList)
        args.expname = args.expname + "_" + "_".join(args.personList)
    else:
        if args.person_num is None:
            args.person_num = 20
        begin = 0
        args.personList = validPerson[begin:begin + args.person_num]
        args.expname = args.expname + "_{}to{}".format(begin, begin + args.person_num)
    # Load data
    K = None
    if args.dataset_type == 'blender':
        images, poses, idcodes, shapeCodes, expTypes, render_poses, hwf, i_split = load_facescape_data(args.datadir,
                                                                                                       args.half_res,
                                                                                                       args.testskip,
                                                                                                       args.personList)
        uv_images = load_uvmap(personList=args.personList)  # ODO: uv_images
        print('Loaded facescape', shapeCodes.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        SCLAE = args.scale
        poses[:, :3, 3] = poses[:, :3, 3] / SCLAE
        render_poses[:, :3, 3] = render_poses[:, :3, 3] / SCLAE

        near = 8
        far = 26
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:  # ;CAMERA K
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, logger, render = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    LM = LMModule(H)
    # Move testing data to GPU

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching  # default False!
    if use_batching:
        # For random ray batching
        print('get rays')  # p is 3*4
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    N_iters = 600000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    render_kwargs_train["network_fn"] = torch.nn.DataParallel(render_kwargs_train["network_fn"])
    render_kwargs_train["network_fine"] = torch.nn.DataParallel(render_kwargs_train["network_fine"])
    render_kwargs_test["network_fn"] = torch.nn.DataParallel(render_kwargs_test["network_fn"])
    render_kwargs_test["network_fine"] = torch.nn.DataParallel(render_kwargs_test["network_fine"])
    render.idSpecificMod = torch.nn.DataParallel(render.idSpecificMod)
    start = start + 1
    for i in trange(start, N_iters):
        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, images+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
        else:
            # Random from one image
            img_i = np.random.choice(i_train)  ##
            target_uvmap = readImgFromPath(uv_images["{}".format(idcodes[img_i])], half_res=False, is_uvMap=True).to(
                device)
            target_expType = expTypes[img_i]
            target = images[img_i]
            target = readImgFromPath(target, half_res=args.half_res)
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]  # pose [3,4]
            idcode_target = torch.Tensor([idcodes[img_i]]).to(device)  # one id codes
            shapeCodes_target = torch.Tensor(shapeCodes[img_i]).to(device)

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                         -1)  # (H, W, images)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, images) #400*400 number of index of an image
                select_coords = LM.sample_point(numOfPoint=N_rand, K=K, pose=pose, id=idcode_target, exp=target_expType,
                                                coords=coords)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)  select image RGB data value
                batch_shapeCodes = shapeCodes_target[None, :].expand(N_rand, -1)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render.render(H, W, K, chunk=args.chunk, rays=batch_rays, shapeCodes=batch_shapeCodes,
                                               uvMap=target_uvmap,
                                               expType=target_expType,
                                               verbose=i < 10, retraw=True,
                                               **render_kwargs_train)
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:  # another loss, from rgb0
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
        if "losses" in extras:
            loss = loss + extras["losses"]
        try:
            loss.backward()
        except:
            optimizer.zero_grad()
            print("### error ####", i)
            optimizer.zero_grad()
            render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, logger, render = create_nerf(args)

            continue
        optimizer.step()
        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1500  # 250*1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        ####   About saving and logging ###########
        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].module.state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].module.state_dict(),
                'network_render_textureEncoder': render.texEncoder.state_dict(),  # new save model parameters
                'network_render_idSpecific': render.idSpecificMod.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'expression_latent_codes_sigma': render.expCodes_Sigma
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            now_test = np.random.choice(i_test, 1)
            print('test poses shape', poses[now_test].shape)
            with torch.no_grad():
                render.render_path(torch.Tensor(poses[now_test]).to(device), [i // 2 for i in hwf], K // 2,
                                   args.chunk // 4, render_kwargs_test,
                                   shapeCodes=torch.Tensor(shapeCodes[now_test]).to(device),
                                   uvMap=torch.stack(
                                       [readImgFromPath(uv_images["{}".format(idcodes[i])], half_res=False,
                                                        is_uvMap=True) for i in
                                        now_test]),
                                   expType=expTypes[now_test],
                                   gt_imgs=np.array([readImgFromPath(images[i]) for i in now_test]),
                                   savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} lr: {optimizer.param_groups[0]['lr']}")
            logger.write(f"{i} Loss: {loss.item()}  PSNR: {psnr.item()}\n")
        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()

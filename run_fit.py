import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch

print(torch.cuda.device_count())
from tools.wild_fit_base import randSp, randTex
from models.render_class import *
import cv2
from tools.config_parser import config_parser
from tools.create_model_condition import create_nerf
from tools.run_nerf_helpers import *
import matplotlib.pyplot as plt
import sys
from tools.load_facescape import pose_spherical

sys.path.append("..")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False
to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)


def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
):
    lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
    if num_iterations > 1500:
        lr = 5e-4
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class LMModule:
    def __init__(self, LM, H=512):
        self.landmark = LM
        self.H = H
        self.lmNum = 68

    def sample_point(self, numOfPoint=None, K=None, pose=None, coords=None, tar_img=None, scale=1, is_debug=False):
        # change landmakr face around ratio
        lm2d = self.landmark // scale

        p = np.long(numOfPoint * 2 // self.lmNum)
        wid = self.H * 0.025 / scale
        rand = np.random.randn(p, 2) * wid
        sampleLandMark = lm2d[:, None, :].repeat(p, 1) + rand[None, :, :].repeat(self.lmNum, 0)
        sampleLandMark = sampleLandMark.reshape(-1, 2).astype(np.int)
        if tar_img is not None:
            sum_tar_img = np.sum(tar_img, 2)[:, :, None]
            # delete out of face
            k = (sum_tar_img[sampleLandMark[:, 0], sampleLandMark[:, 1], :] != 0)[:, 0]
            sampleLandMark = sampleLandMark[k, :]

            # add newpoints around face
            points_face_outline = np.concatenate([lm2d[1:5], lm2d[12:16]], 0)
            numt = sampleLandMark.shape[0] // 50 * scale
            rand = np.random.randn(numt, 2) * wid  # 1:6 = around face rigion : inside face
            points_face_outline = points_face_outline[:, None, :].repeat(numt, 1) + rand[None, :, :].repeat(8, 0)

            sampleLandMark = np.vstack([sampleLandMark, points_face_outline.reshape(-1, 2)])

            lenOfPoint = sampleLandMark.shape[0]
            if lenOfPoint < numOfPoint:  # add more points
                tmNum = numOfPoint // lenOfPoint + 1
                sampleLandMark = sampleLandMark.repeat(tmNum, 0)
                res_sampleLandMark = sampleLandMark[:numOfPoint, :]
                sampleLandMark = res_sampleLandMark
            else:
                res_sampleLandMark = sampleLandMark[np.random.choice(np.arange(lenOfPoint), numOfPoint), :]
                sampleLandMark = res_sampleLandMark
            sampleLandMark = sampleLandMark.astype(np.int)
            assert sampleLandMark.max() < tar_img.shape[0] and sampleLandMark.min() > 0

        if is_debug == True:
            img = tar_img * 0.5  # np.zeros([256,256,3])
            img[lm2d[:, 0], lm2d[:, 1], :] = np.ones(3)
            img[sampleLandMark[:, 0], sampleLandMark[:, 1], :] = np.ones(3)
            plt.imshow(img)
            plt.show()
        return torch.Tensor(sampleLandMark).cuda().long()


def RGB2YUV(RGB):  # max 1 RGB
    R, G, B = np.split(RGB * 255., [1, 2], axis=2)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    V = 0.5 * R - 0.4187 * G - 0.0813 * B + 128
    return [Y, U, V]


def YUV2RGB(YUV):  # max 255 YUV
    Y, U, V = YUV
    R = Y + 1.402 * (V - 128)
    G = Y - 0.34414 * (U - 128) - 0.71414 * (V - 128)
    B = Y + 1.772 * (U - 128)
    RGB_res = np.concatenate([R, G, B], 2)
    return RGB_res


def load_pose(src_path):  # process src, load npy
    a = ""
    for p in src_path.split("/")[:-2]:
        a = a + p + "/"
    # print(a)
    a = a + ("pose_" + src_path.split("/")[-1][:-4] + ".npy")
    print("load poses: ", a)
    dict = np.load(a, allow_pickle=True).item()
    pose = dict["pose"]
    kp2d = dict["kp"]
    print(pose.shape, kp2d.shape)
    return pose, kp2d


def get_rays_withGrad(H, W, K, c2w, focal):  # torch : get ray
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t().to(device)
    j = j.t().to(device)
    dirs = torch.stack([(i - K[0][2]) / focal, -(j - K[1][2]) / focal, -torch.ones_like(i).to(device)],
                       -1)  # stack,and dirs create in pixel coordinate and use K, convert to camera coordinate
    # Rotate ray directions from camera frame to the world frame  #camera->world
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  #:::this is the same as dirs@c2w[:3,:3].T
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)  # ray center o is the same as camera center
    return rays_o, rays_d

expressionName = ["neutral", "smile", "mouth_stretch", "anger", "jaw_left",
                      "jaw_right", "jaw_forward", "mouth_left", "mouth_right", "dimpler",
                      "chin_raiser", "lip_puckerer", "lip_funneler", "sadness", "lip_roll",
                      "grin", "cheek_blowing", "eye_closed", "brow_raiser", "brow_lower"]

def train(src_path=None, renderType=None, num_iterations=2000, is_load_par=False, args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # args = parser.parse_args()
    args.device = device
    args.bmCodesLength = 50
    args.PersonNumber = 300  # Person number
    # Cast intrinsics to right types
    hwf = [512, 512, 1200.0]
    H_raw, W_raw, focal_raw = hwf
    H_raw, W_raw = int(H_raw), int(W_raw)
    K_raw = np.array([
        [focal_raw, 0, 0.5 * H_raw],
        [0, focal_raw, 0.5 * W_raw],
        [0, 0, 1]
    ])
    near = 8
    far = 26

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname + "_0to{}".format(args.person_num)
    args.expname = expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, logger, render = create_nerf(args)
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    render_kwargs_test["network_fn"] = torch.nn.DataParallel(render_kwargs_test["network_fn"])
    render_kwargs_test["network_fine"] = torch.nn.DataParallel(render_kwargs_test["network_fine"])
    render.idSpecificMod = torch.nn.DataParallel(render.idSpecificMod)
    # Move testing data to GPU

    # use in mode 1 random_uv
    myrandomSp = randSp()
    myrandomTex = randTex()
    render.eval()
    render = render.cuda()

    # target image:
    if src_path == None:
        src_path = "/data/myNerf/data/generateFace/segRelRes/00028.png"
    if renderType == None:
        renderType = "rendering"  # "rendering"  # "" "fitting"

    typeName = src_path.split("/")[-2]  # "segRelRes"  # postfix

    src_img_name = src_path.split("/")[-1][:-4]
    crop_img_raw_uint8 = imageio.imread(src_path)
    pose, kp2d_raw = load_pose(src_path)

    print("!!!!! no crop !!!!")
    crop_img_raw = crop_img_raw_uint8 / 255.
    small_scale = np.log2(8).astype(np.int)
    if args.half_res:
        small_scale = small_scale + 1
        scale_now = 2 ** (small_scale)

    else:
        scale_now = 2 ** small_scale

    LM = LMModule(kp2d_raw, H_raw)
    target_image = torch.from_numpy(crop_img_raw).cuda()  # src_img_scale
    render_poses = torch.from_numpy(pose.astype(np.float32)).to(device)  # 40 4 4

    render_bm = myrandomSp.getRand(device)
    render_uv = myrandomTex.getRand(device)
    render_expCodes = render.expCodes_Sigma[0].detach().clone()
    render_bm.requires_grad = True
    render_uv.requires_grad = True
    render_expCodes.requires_grad = True
    render_poses.requires_grad = True
    # training setting

    lr = 2e-3
    lr_shape = 4e-3
    light_scale = torch.Tensor([1, 1]).cuda()
    light_scale.requires_grad = True

    optimizer_bm = torch.optim.Adam([light_scale, render_poses], lr=lr)
    optimizer_uv = torch.optim.Adam([render_uv], lr=lr)
    optimizer_exp = torch.optim.Adam([render_expCodes, render_bm], lr=lr_shape)
    decreased_by = 1.1
    adjust_lr_every = int(num_iterations / 6)
    adjust_lr_every_bm = int(num_iterations / 6)
    N_rand = 1024
    loss_l1 = torch.nn.L1Loss()

    # path initial
    ttype = src_path.split("/")[-2]
    tfileName = src_path.split("/")[-1]
    testsavedir = src_path[:-len(tfileName) - len(ttype) - 2]
    os.makedirs(testsavedir, exist_ok=True)
    name = None
    testsavedir = os.path.join(testsavedir, "fitting/{}_{}".format(typeName, src_img_name))

    os.makedirs(testsavedir, exist_ok=True)
    print('log dir', testsavedir)
    imageio.imwrite(testsavedir + "/target.png", to8b(target_image))

    begin_iter = 0
    if is_load_par == True or renderType in ['rendering', "rendering_modulation"]:
        load_path = testsavedir + "/saving_Parameters.tar"
        if os.path.exists(load_path):
            ckpt = torch.load(load_path)
            print("loading fitting par from ", load_path)
            render_poses.data = ckpt['saving_pose'].data
            render_uv.data = ckpt['saving_uv'].data
            render_bm.data = ckpt['saving_bm'].data
            render_expCodes.data = ckpt['saving_exp'].data
            light_scale.data = ckpt['saving_global_light'].data
            optimizer_bm.load_state_dict(ckpt['optimizer_state_dict_bm'])
            optimizer_uv.load_state_dict(ckpt['optimizer_state_dict_uv'])
            optimizer_exp.load_state_dict(ckpt['optimizer_state_dict_exp'])
            begin_iter = ckpt['iter']
        print("load epoch ", begin_iter)
    else:
        print("No load, as initialized. Mode--{}".format(renderType))

    if renderType == "fitting":
        if os.path.exists(testsavedir) and begin_iter == num_iterations:
            print("finnish fitting", testsavedir)
            return
        K_raw = np.array([
            [focal_raw, 0, 0.5 * W_raw],
            [0, focal_raw, 0.5 * H_raw],
            [0, 0, 1]
        ])
        K_raw = torch.from_numpy(K_raw).cuda()
        change_epoch = [0, 600, 1000, 1300, 1500]
        for e in range(begin_iter, begin_iter + num_iterations + 1):
            is_debug = False
            if e in change_epoch[:small_scale] or e % 2000 == 0:
                print(e)
                scale_now = max(int(scale_now / 2), 1)
                size = int(512 // scale_now)
                print("epoch {}, scale {}. size{}".format(e, scale_now, size))
                crop_img = cv2.resize(crop_img_raw, (size, size)).astype(np.float32)
                target_image = torch.from_numpy(crop_img).cuda()
                H = int(H_raw // scale_now)
                W = int(W_raw // scale_now)
                focal = focal_raw / scale_now
                K = K_raw / scale_now
            rays_o, rays_d = get_rays_withGrad(H, W, K, render_poses, focal)
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                 -1)  # (H, W, images)
            coords = torch.reshape(coords, [-1, 2])

            select_coords = LM.sample_point(numOfPoint=args.N_rand, K=K, pose=render_poses.detach(), coords=coords,
                                            tar_img=crop_img, scale=scale_now, is_debug=is_debug)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)

            target_s = target_image[
                select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)  select image RGB data value

            batch_bmCodes = render_bm.expand(N_rand, args.bmCodesLength)

            adjust_learning_rate(lr, optimizer_bm, e, decreased_by, adjust_lr_every_bm)

            adjust_learning_rate(lr, optimizer_uv, e, decreased_by, adjust_lr_every)
            adjust_learning_rate(lr_shape, optimizer_exp, e, decreased_by, adjust_lr_every_bm)

            optimizer_bm.zero_grad()
            optimizer_uv.zero_grad()
            optimizer_exp.zero_grad()
            rgb, disp, acc, _ = render.render_fitting(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                      shapeCodes=batch_bmCodes,
                                                      uvCodes=render_uv, expType=20, expCodes=render_expCodes,
                                                      **render_kwargs_test)
            loss = loss_l1(light_scale[0] * (rgb), target_s)
            loss.backward()
            optimizer_exp.step()
            optimizer_bm.step()
            optimizer_uv.step()

            if e % 10 == 0:
                print("iter{} loss{} lr-uv{} lr-bm/exp{} global_light{}".format(e, loss,
                                                                                optimizer_uv.param_groups[0]['lr'],
                                                                                optimizer_exp.param_groups[0]['lr'],
                                                                                light_scale))
            if e % 100 == 0:
                saving = {}
                saving['saving_bm'] = render_bm.detach()  # .cpu().numpy()
                saving['saving_uv'] = render_uv.detach()  # .cpu().numpy()
                saving['saving_exp'] = render_expCodes.detach()
                saving['saving_pose'] = render_poses.detach()
                saving['saving_global_light'] = light_scale.detach()
                saving['iter'] = num_iterations
                saving['optimizer_state_dict_bm'] = optimizer_bm.state_dict()
                saving['optimizer_state_dict_uv'] = optimizer_uv.state_dict()
                saving['optimizer_state_dict_exp'] = optimizer_exp.state_dict()
                torch.save(saving, testsavedir + '/saving_Parameters.tar')
            if e % 500 == 0 or e in [100, 200, 300]:
                with torch.no_grad():
                    if H > 250:
                        scale_now_r = 2
                        H_render = int(H_raw // scale_now_r)
                        W_render = int(W_raw // scale_now_r)
                        K_render = K_raw / scale_now_r
                    else:
                        H_render = H
                        W_render = W
                        K_render = K
                    rgb, disp, acc, _ = render.render_fitting(H_render, W_render, K_render, chunk=args.chunk // 2,
                                                              c2w=render_poses[:3, :4],
                                                              shapeCodes=render_bm, uvCodes=render_uv,
                                                              expType=20, expCodes=render_expCodes,
                                                              **render_kwargs_test)
                    rgb8 = to8b(rgb)
                    filename = os.path.join(testsavedir, "{}_{}_{}.png".format(typeName, src_img_name, e))
                    imageio.imwrite(filename, rgb8)

    elif renderType == "rendering":
        print("rendering~!")
        H = H_raw
        W = W_raw
        K = K_raw
        is_harf = True
        if is_harf == True:
            scale_now = 2
            H = int(H_raw // scale_now)
            W = int(W_raw // scale_now)
            K = K_raw / scale_now
        testsavedir = os.path.join(testsavedir, "render")
        os.makedirs(testsavedir, exist_ok=True)
        for angle in [-60, 0, 60]:
            target_pose = pose_spherical(float(angle), 0, 800.0 / 50)
            with torch.no_grad():
                rgb, disp, acc, _ = render.render_fitting(H, W, K, chunk=args.chunk, c2w=target_pose[:3, :4],
                                                          shapeCodes=render_bm, uvCodes=render_uv, expType=20,
                                                          expCodes=render_expCodes,
                                                          **render_kwargs_test)
            rgb8 = to8b(rgb)

            filename = os.path.join(testsavedir, 'fitRes_{}.png'.format(angle))

            imageio.imwrite(filename, rgb8)
            print("rendering: ", filename)

    elif renderType == "rendering_modulation":
        print("rendering modulation~!")
        H = H_raw
        W = W_raw
        K = K_raw
        target_pose = pose_spherical(0, 0, 800.0 / 50)
        is_harf = True
        if is_harf == True:
            scale_now = 2
            H = int(H_raw // scale_now)
            W = int(W_raw // scale_now)
            K = K_raw / scale_now
        testsavedir = os.path.join(testsavedir, "render")
        os.makedirs(testsavedir, exist_ok=True)
        # Face Rigging
        for expType in [9,14,2,16,17]:  #number in [0-20)
            with torch.no_grad():
                c_render_expCodes = render.expCodes_Sigma[expType]
                rgb, disp, acc, _ = render.render_fitting(H, W, K, chunk=args.chunk, c2w=target_pose[:3, :4],
                                                          shapeCodes=render_bm, uvCodes=render_uv, expType=20,
                                                          expCodes=c_render_expCodes,
                                                          **render_kwargs_test)
            rgb8 = to8b(rgb)
            filename = os.path.join(testsavedir, 'rigging_{}.png'.format(expressionName[expType]))
            imageio.imwrite(filename, rgb8)
            print("rendering: ", filename)
        par = np.load("./predef_par.npy", allow_pickle=True).item()
        # Face Editing -- change shape
        for i in range(3):
            # c_render_bm = myrandomSp.getRand(device=device)  #Random Genaration
            c_render_bm = par['shape'][i][None, ...].to(device)
            with torch.no_grad():
                rgb, disp, acc, _ = render.render_fitting(H, W, K, chunk=args.chunk, c2w=target_pose[:3, :4],
                                                          shapeCodes=c_render_bm, uvCodes=render_uv, expType=20,
                                                          expCodes=render_expCodes,
                                                          **render_kwargs_test)
            rgb8 = to8b(rgb)
            filename = os.path.join(testsavedir, 'chg_shape_{}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            print("rendering: ", filename)
        # Face Editing -- change texture
        for i in range(3):
            # c_render_tex = myrandomTex.getRand(device)  #Random Genaration
            c_render_tex = par['texture'][i].to(device)
            with torch.no_grad():
                rgb, disp, acc, _ = render.render_fitting(H, W, K, chunk=args.chunk, c2w=target_pose[:3, :4],
                                                          shapeCodes=render_bm, uvCodes=c_render_tex, expType=20,
                                                          expCodes=render_expCodes,
                                                          **render_kwargs_test)
            rgb8 = to8b(rgb)
            filename = os.path.join(testsavedir, 'chg_tex_{}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            print("rendering: ", filename)
    print('Done rendering', testsavedir)

    return


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    parser.add_argument("--filePath", type=str, default=None, help="file path of the image to be fitted", required=True)
    parser.add_argument("--renderType", type=str, default="fitting",
                        help="\"rendering\" or \"fitting\": if \"rendering\", we load the fitted parameters and render rgb images from three views; "
                             "if \"fitting\", we optimize our codes to fit to the image.")
    parser.add_argument("--is_load_par", type=bool, default=None, help="if load the fitted results")
    parser.add_argument("--num_iterations", type=int, default=2000, help="if load the fitted results")
    args = parser.parse_args()
    train(src_path=args.filePath, renderType=args.renderType, num_iterations=args.num_iterations,
          is_load_par=args.is_load_par, args=args)

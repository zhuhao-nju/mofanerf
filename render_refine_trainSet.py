import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch

from tools.wild_fit_base import randSp, randTex
from models.render_class import *

import cv2
from tqdm import tqdm, trange

from models.render_class import *
from tools.config_parser import config_parser
from tools.create_model_condition import create_nerf
from tools.run_nerf_helpers import *
import matplotlib.pyplot as plt
import sys
from tools.load_facescape import pose_spherical
import random
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

class LoggerModule():
    def __init__(self, path):
        self.log = open(path, "a", encoding="utf-8")


    def write(self, message):
        self.log.write(message+'\n')
        self.log.flush()

    def __del__(self):
        self.log.close()


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
    bmModel = np.load('./data/factors_id.npy')
    return bmModel



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
    expressionName = ["neutral", "smile", "mouth_stretch", "anger", "jaw_left",
                      "jaw_right", "jaw_forward", "mouth_left", "mouth_right", "dimpler",
                      "chin_raiser", "lip_puckerer", "lip_funneler", "sadness", "lip_roll",
                      "grin", "cheek_blowing", "eye_closed", "brow_raiser", "brow_lower"]

    parser = config_parser()
    args = parser.parse_args()
    args.device = device
    validPerson = getValidPerson(args.datadir)

    # for specific:
    args.begin_person = 0
    args.end_penson = 300

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
    args.half_res = True
    K = None
    if args.dataset_type == 'blender':
        images, poses, idcodes, shapeCodes, expTypes, render_poses, hwf, i_split = load_facescape_data(args.datadir,
                                                                                                       args.half_res,
                                                                                                       args.testskip,
                                                                                                       args.personList)
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

    if args.render_test:
        render_poses = np.array(poses[i_test])

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

    poses = torch.Tensor(poses).to(device)

    testsavedir = os.path.join(basedir, expname,
                               'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
    os.makedirs(os.path.join(testsavedir, "rf_trainSet"), exist_ok=True)
    logger = LoggerModule(testsavedir + "/renderImageList.txt")

    num_images_per_person = 100 * 20   # number of images per person in the images dataset
    num_exp_type = 10                  # setting of the number of expressions each person
    num_images_per_exp = 8             # setting of the number of rendering images each expression

    with torch.no_grad():
        with tqdm(total=10*8 * (args.end_penson - args.begin_person), desc='images', leave=True,
                  unit='frame', unit_scale=True, ncols=80) as pbar:
            for i in range(args.begin_person * num_images_per_person, args.end_penson * num_images_per_person,
                           num_images_per_person):
                info = images[i].split("/")
                curr_id = int(info[-3])
                exp_dir = os.path.join(testsavedir, f'rf_trainSet/train/{curr_id}')
                os.makedirs(exp_dir, exist_ok=True)
                exp_render = os.listdir(exp_dir)
                if len(exp_render) < num_exp_type:  #make sure there are ten types of expressions for each identity
                    exp_not_use = []
                    for kk in range(0, 20):
                        if expressionName[kk] not in exp_render:
                            exp_not_use.append(kk)
                    selected_expression = random.sample(exp_not_use, num_exp_type - len(exp_render))
                    print(curr_id, len(selected_expression))
                    pbar.update(len(selected_expression) * num_images_per_exp)
                else:
                    pbar.update(10 * 8)
                    continue
                selected_views = []
                for i_exp in selected_expression:
                    i_view = (np.array(random.sample(range(0, 100), num_images_per_exp))).astype(np.int)
                    selected_views.extend((i_view + i_exp * 100).tolist())

                view_index = 0
                for view in selected_views:
                    info = images[i + view].split("/")
                    curr_id = int(info[-3])
                    curr_exp_name = info[-2]
                    curr_img_name = info[-1][:-4]
                    pathNameRender = os.path.join(testsavedir, f'rf_trainSet/train/{curr_id}/{curr_exp_name}')
                    if os.path.exists(pathNameRender):
                        if len(os.listdir(pathNameRender)) >= 8:
                            print("pass", pathNameRender)
                            pbar.update(1)
                            continue
                    else:
                        os.makedirs(pathNameRender, exist_ok=True)

                    logger.write("{},{},imagesID,{},{}".format(curr_id, curr_exp_name, i + view, curr_img_name))
                    curr_filename = f'rf_trainSet/train/{curr_id}/{curr_exp_name}/{curr_img_name}'

                    curr_render_pose = poses[i + view].unsqueeze(0)
                    curr_render_shape = torch.Tensor(shapeCodes[i + view]).reshape(1, -1)
                    curr_render_uv = f'/data/myNerf/data/textureMap300/{curr_id}/1_neutral.jpg'
                    curr_uvMap = readImgFromPath(curr_render_uv, half_res=False, is_uvMap=True).unsqueeze(0)
                    curr_expType = torch.Tensor([expressionName.index(curr_exp_name)]).long().to(device)


                    view_index += 1
                    pbar.update(1)
                    rgbs, _ = render.render_path(render_poses=curr_render_pose, hwf=hwf, K=K, chunk=args.chunk,
                                                 render_kwargs=render_kwargs_test,
                                                 gt_imgs=images,
                                                 savedir=testsavedir,
                                                 render_factor=args.render_factor,
                                                 shapeCodes=curr_render_shape,
                                                 uvMap=curr_uvMap,
                                                 expType=curr_expType,
                                                 name=curr_filename
                                                 )

                    print(
                        f"[Finish] {i}/40000, mode: train, id: {curr_id}, exp: {curr_expType.item(), curr_exp_name}, view: {view_index}")


    sys.exit()

    return


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()

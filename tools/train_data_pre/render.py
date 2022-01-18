import multiprocessing
import random
import time

import cv2
import json
import numpy as np
import os
import pyrender
import trimesh


def render_glcam(model_in,
                 K=None,
                 Rt=None,
                 scale=1.0,
                 rend_size=(512, 512),
                 light_trans=np.array([[0], [100], [0]]),
                 flat_shading=False, scaleMesh=1.0):
    # Mesh creation
    if isinstance(model_in, str) is True:
        mesh = trimesh.load(model_in, process=False)
    else:
        copy = model_in.copy()
        mesh = copy
    aa = rot_phi(90 / 180 * np.pi)[:3, :3]

    mesh.vertices = mesh.vertices * scaleMesh
    pr_mesh = pyrender.Mesh.from_trimesh(mesh)

    scene = pyrender.Scene(bg_color=[0, 0, 0])

    # Adding objects to the scene
    face_node = scene.add(pr_mesh)

    # Caculate fx fy cx cy from K
    fx, fy = K[0][0] * scale, K[1][1] * scale
    cx, cy = K[0][2] * scale, K[1][2] * scale

    # Camera Creation
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                    znear=0.1, zfar=1000)
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = Rt[:3, :3].T
    cam_pose[:3, 3] = -Rt[:3, :3].T.dot(Rt[:, 3])  # center

    scene.add(cam, pose=cam_pose)

    # Set up the light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    light_pose = cam_pose.copy()
    light_pose[0:3, :] += light_trans
    scene.add(light, pose=light_pose)

    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0],
                                   point_size=1.0)
    if flat_shading is True:
        color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    else:
        color, depth = r.render(scene)

    # rgb to bgr for cv2
    color = color[:, :, [2, 1, 0]]

    return depth, color


def render_cvcam(model_in,  # model name or trimesh
                 K=None,
                 Rt=None,
                 scale=1.0,
                 rend_size=(512, 512),
                 light_trans=np.array([[0], [100], [0]]),
                 flat_shading=False):
    if np.array(K).all() == None:
        K = np.array([[2000, 0, 256],
                      [0, 2000, 256],
                      [0, 0, 1]], dtype=np.float64)

    if np.array(Rt).all() == None:
        Rt = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]], dtype=np.float64)

    # define R to transform from cvcam to glcam
    R_cv2gl = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
    Rt_cv = R_cv2gl.dot(Rt)

    return render_glcam(model_in, K, Rt_cv, scale, rend_size, light_trans, flat_shading)


trans_t = lambda t: np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).astype(np.float32)

# rotate phi around x axes
rot_phi = lambda phi: np.array([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).astype(np.float32)

# rotate theta around y axes
rot_theta = lambda th: np.array([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).astype(np.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = rot_phi(phi / 180. * np.pi) @ c2w

    center = c2w[:3, -1]
    R = c2w[:3, :3]
    c2w[:3, -1] = np.array([0, 0, -radius])  # openGL Method decides it to be negative

    c2w[-1, -1] = 0
    return c2w


def generate_Rt_fix_view():
    all_Rt = []
    for theta in range(-30, 60, 15):
        scaleMesh = 50
        Rt = [pose_spherical(angle, theta, 800.0 / scaleMesh) for angle in np.linspace(-90, 90, 20 + 1)[
                                                                           :-1]]  # par1, along axis y, left and right; par2 along axis x, up and down(90 -> -60)

        for i, rt in enumerate(Rt):
            all_Rt.append((theta, i, rt))

    return all_Rt


expressionName = ["neutral", "smile", "mouth_stretch", "anger", "jaw_left", "jaw_right", "jaw_forward", "mouth_left",
                  "mouth_right", "dimpler",
                  "chin_raiser", "lip_puckerer", "lip_funneler", "sadness", "lip_roll", "grin", "cheek_blowing",
                  "eye_closed", "brow_raiser", "brow_lower"]

h, w = 512, 512
K = np.array(
    [
        [1200, 0, h // 2],
        [0, 1200, w // 2],
        [0, 0, 1]
    ]
)

camera_angle_x = np.arctan(h // 2. / 1200.) * 2

save_img = True
save_config = True
configList = []
transform_train = []
transform_test = []
transform_val = []
transform_all = []


def processSingle(id_idx):
    transform_train = []
    transform_test = []
    transform_val = []
    transform_all = []

    align_fold = "../data/models_out"
    fold_name = '../data/multiViewImages'

    for exp_idx in range(1, 21):
        configList.clear()
        mesh_dirname = "{}/{}/{}_{}.obj".format(align_fold, id_idx, exp_idx, expressionName[exp_idx - 1])
        if not os.path.exists(mesh_dirname):
            print(mesh_dirname + " not exists")
            continue
        print("process {}".format(mesh_dirname))

        selected_Rt = generate_Rt_fix_view()
        list = [i for i in range(len(selected_Rt))]
        test_list = random.sample(list, 20)

        scaleMesh = 50
        for index, (theta, i, rt) in enumerate(selected_Rt):
            if save_img == True:
                rend_depth, rend_img = render_glcam(mesh_dirname, K, rt[:-1, :], rend_size=(h, w), flat_shading=True,
                                                    scaleMesh=1 / scaleMesh)

                os.makedirs("{}/{}/{}".format(fold_name, id_idx, expressionName[exp_idx - 1]), exist_ok=True)
                cv2.imwrite("{}/{}/{}/{}_{}.png".format(fold_name, id_idx, expressionName[exp_idx - 1], theta, i),
                            rend_img)

                print(f'[finish] id: {id_idx}, exp: {exp_idx}, {index + 1}/{len(selected_Rt)}')

            if save_config == True:
                cam_pose = np.eye(4)
                cam_pose[:3, :3] = rt[:3, :3].T
                cam_pose[:3, 3] = -rt[:3, :3].T.dot(rt[:3, 3])
                t = {"file_path": "/{}/{}/{}_{}".format(id_idx, expressionName[exp_idx - 1], theta, i),
                     "rotation": 0.666, "expression": exp_idx - 1, "transform_matrix": cam_pose.tolist()}
                configList.append(t)

                if index in test_list:
                    transform_test.append(t)
                    transform_val.append(t)
                else:
                    transform_train.append(t)
                transform_all.append(t)

    if len(transform_all) == 0:
        return

    # train set
    if save_config == True:
        conf = {"camera_angle_x": camera_angle_x, "frames": transform_train}
        with open('{}/transforms_train_{}.json'.format(fold_name, id_idx), 'w') as result_file:
            conf = json.dumps(conf, indent=1, sort_keys=False)
            result_file.write(conf)

    # val set
    if save_config == True:
        conf = {"camera_angle_x": camera_angle_x, "frames": transform_val}
        with open('{}/transforms_val_{}.json'.format(fold_name, id_idx), 'w') as result_file:
            conf = json.dumps(conf, indent=1, sort_keys=False)
            result_file.write(conf)

    # test set
    if save_config == True:
        conf = {"camera_angle_x": camera_angle_x, "frames": transform_test}
        with open('{}/transforms_test_{}.json'.format(fold_name, id_idx), 'w') as result_file:
            conf = json.dumps(conf, indent=1, sort_keys=False)
            result_file.write(conf)

    # all set
    if save_config == True:
        conf = {"camera_angle_x": camera_angle_x, "frames": transform_all}
        with open('{}/transforms_all_{}.json'.format(fold_name, id_idx), 'w') as result_file:
            conf = json.dumps(conf, indent=1, sort_keys=False)
            result_file.write(conf)


def render_by_multiprocessing():
    pool = multiprocessing.Pool(processes=5)
    for id_idx in range(622, 627):
        print("id", id_idx)
        pool.apply_async(processSingle, (id_idx,))
    pool.close()
    pool.join()


def render_by_order():
    for id_idx in range(1, 2):
        processSingle(id_idx)


def render_for_special_index():
    pool = multiprocessing.Pool(processes=1)
    ids = [1]
    for id_idx in ids:
        print("id", id_idx)
        pool.apply_async(processSingle, (id_idx,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    start = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"start: {start}")
    render_by_order()
    # render_by_multiprocessing()
    # render_for_special_index()
    end = time.strftime("%Y-%m-%d, %H:%M:%S")
    print(f"end: {end}")

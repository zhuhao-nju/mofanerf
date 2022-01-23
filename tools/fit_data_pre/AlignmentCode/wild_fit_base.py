import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

glob_neutral_tmp_LM = np.array(
    [[143, 214], [146, 244], [151, 273], [158, 302], [168, 328], [184, 352], [205, 371], [229, 386], [259, 390],
     [287, 385], [311, 371], [331, 352], [347, 329], [356, 303], [364, 274], [368, 245], [370, 214], [163, 186],
     [177, 172], [197, 168], [218, 173], [236, 182], [276, 180], [296, 168], [317, 163], [339, 167], [352, 184],
     [257, 206], [257, 226], [257, 246], [257, 267], [237, 286], [248, 288], [258, 289], [268, 288], [278, 285],
     [183, 210], [195, 203], [210, 204], [223, 215], [209, 217], [194, 217], [290, 213], [303, 203], [318, 201],
     [330, 207], [320, 214], [305, 215], [219, 328], [234, 320], [249, 314], [258, 317], [268, 314], [281, 320],
     [295, 328], [282, 338], [269, 342], [258, 343], [248, 343], [234, 339], [226, 328], [248, 326], [258, 327],
     [268, 326], [289, 328], [268, 327], [258, 328], [248, 327]])


class mfnerf_fitting:
    def __init__(self, lm_file):
        if lm_file != None:
            self.face_pred = dlib.shape_predictor(lm_file)
            self.detector = dlib.get_frontal_face_detector()
        self.fp_size = 512
        self.meanDist_nerfDS_front = 35.74
        self.meanPoint_nerfDS_front = [257.45, 281.245]
        self.tmpLM = glob_neutral_tmp_LM
        self.fcFitter = pose_estimate()

    def detect_kp2d(self, src_img, is_show_img=False, tar_img=None, tar_kp=None, is_rigid=False):
        # ========== extract landmarks ==========
        dshape = [512, 512, 3]
        faces = self.detector(src_img, 1)  # detect faces
        pts = self.face_pred(src_img, faces[0])  # get landmarks for the first face
        kp2d_raw = np.array(([[p.x, p.y] for p in pts.parts()]))  # 17 28 #kp[i,:] = [col index, row index]

        # ---  scale target image with average distance---
        if tar_img is not None:
            faces = self.detector(tar_img, 1)  # detect faces
            pts = self.face_pred(tar_img, faces[0])  # get landmarks for the first face
            kp2d_tmpl = np.array(([[p.x, p.y] for p in pts.parts()]))
            dshape = tar_img.shape
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        elif tar_kp is not None:
            kp2d_tmpl = tar_kp[:, ::-1]
        else:
            kp2d_tmpl = self.tmpLM
        M, scale = self.transformation_from_points(src_points=kp2d_raw, tmpt_points=kp2d_tmpl)
        if is_rigid:
            scale_x = (M[0, 0] + M[1, 1]) / 2.  # np.sqrt(np.sum(np.array(M[0, :2])**2))
            scale_y = scale_x  # np.sqrt(np.sum(np.array(M[1, :2])**2))
            M[:2, :2] = np.array([[scale_x, 0], [0, scale_y]])
        out = self.warp_im(src_img, M, dshape=dshape)
        # dst = out
        dst = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        # ==== method1 use M to convert point X ===
        # kp2d_t = (((kp2d_raw ).dot(M[:2, :2]/scale) - M[:2, 2].T)/scale).astype(np.int)  #S*M = T T
        # ===== method2 re-detect the kp ===
        faces = self.detector(out, 1)  # detect faces
        pts = self.face_pred(out, faces[0])
        kp2d_t = np.array(([[p.x, p.y] for p in pts.parts()]))
        kp2d_l = np.zeros_like(kp2d_t)
        kp2d_l[:, 0], kp2d_l[:, 1] = kp2d_t[:, 1], kp2d_t[:, 0]
        kp2d_l = np.array(kp2d_l)
        if is_show_img:
            img = dst.copy() * 0.5
            img[kp2d_t[:, 1], kp2d_t[:, 0], :] = np.ones(3) * 255
            plt.imshow(img / 255.)
            plt.show()
            img = out.copy()
        return kp2d_l, dst

    def warp_im(self, im, M, dshape=[512, 512, 3]):
        output_im = np.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im,
                       M[:2],
                       (dshape[1], dshape[0]),
                       dst=output_im,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)
        return output_im

    def transformation_from_points(self, src_points, tmpt_points):
        tmpt_points = tmpt_points.astype(np.float64)
        src_points = src_points.astype(np.float64)

        c1 = np.mean(tmpt_points, axis=0)
        c2 = np.mean(src_points, axis=0)
        tmpt_points -= c1
        src_points -= c2

        s1 = np.std(tmpt_points)
        s2 = np.std(src_points)
        tmpt_points /= s1
        src_points /= s2
        U, S, Vt = np.linalg.svd(tmpt_points.T.dot(src_points))  # C=BT * A
        R = (U.dot(Vt)).T
        return np.vstack([np.hstack([(s2 / s1) * R,
                                     (c2.T - (s2 / s1) * R.dot(c1.T)).reshape(-1, 1)]),  # 2,4   ??
                          np.matrix([0., 0., 1.])]), s2 / s1

    def visual_kp(self, kp2d, img):

        kp2d = kp2d.parts()
        for index, i in enumerate(kp2d):
            pt_pos = (i.x, i.y)
            cv2.circle(img, center=pt_pos, radius=2, color=(0, 255, 0), thickness=1)
        cv2.imshow('Frame', img)
        # cv2.waitKey(0)

    def crop_face(self, image, image_landmarks):
        image_shape = image.shape
        hull_mask = get_image_hull_mask(image_shape, image_landmarks)
        hull_mask = hull_mask.astype(np.uint8)
        crop_res = merge_add_mask(image, hull_mask)
        return crop_res

    def get_pose_from_kp2d(self, kp2d):
        return self.fcFitter.from_kp2d_get_pos(kp2d)


class pose_estimate():
    def __init__(self):
        landmark = np.load("./data/1_975_landmarks.npy")
        self.tmpLM = landmark[1, 0, :, :] / 50.

    def fit_kp2d(self, kp2d):
        # ========== initialize ==========
        lm_pos = np.asarray(kp2d)

        rot_vector = np.array([0, 0, 0], dtype=np.double)
        trans = np.array([0, 0])
        scale = 1.

        mesh_verts = self.tmpLM  # self.shape_bm_core.dot(id).dot(exp).reshape((-1, 3))  #all mesh  -> only template landmark
        tmpLM_2D = self.project(self.tmpLM, rot_vector, scale, trans)  # landmark 3d -> 2d

        # ========== iterative optimize ==========
        for optimize_loop in range(4):
            tmpLM_2D = tmpLM_2D / scale
            vertices_mean = np.mean(tmpLM_2D, axis=0)
            vertices_2d = tmpLM_2D - vertices_mean

            lm_mean = np.mean(lm_pos, axis=0)
            lm = lm_pos - lm_mean
            scale = np.sum(np.linalg.norm(lm, axis=1)) / np.sum(np.linalg.norm(vertices_2d, axis=1))
            trans = lm_mean - vertices_mean * scale
            lm_pos_3D = self.tmpLM
            scale, trans, rot_vector = self._optimize_rigid_pos_2d(scale, trans, rot_vector,
                                                                   lm_pos_3D, lm_pos)
            tmpLM_2D = self.project(mesh_verts, rot_vector, scale, trans)

        params = [scale, trans, rot_vector]

        return params

    def from_kp2d_get_pos(self, kp2d):
        # ========== initialize ==========
        lm_pos = np.asarray(kp2d)

        rot_vector = np.array([0, 0, 0], dtype=np.double)
        trans = np.array([0, 0])
        scale = 1.

        mesh_verts = self.tmpLM
        tmpLM_2D = self.project(self.tmpLM, rot_vector, scale, trans)  # landmark 3d -> 2d
        # ========== iterative optimize ==========
        for optimize_loop in range(4):
            tmpLM_2D = tmpLM_2D / scale
            vertices_mean = np.mean(tmpLM_2D, axis=0)
            vertices_2d = tmpLM_2D - vertices_mean

            lm_mean = np.mean(lm_pos, axis=0)
            lm = lm_pos - lm_mean
            scale = np.sum(np.linalg.norm(lm, axis=1)) / np.sum(np.linalg.norm(vertices_2d, axis=1))
            trans = lm_mean - vertices_mean * scale
            lm_pos_3D = self.tmpLM
            scale, trans, rot_vector = self._optimize_rigid_pos_2d(scale, trans, rot_vector,
                                                                   lm_pos_3D, lm_pos)
            tmpLM_2D = self.project(mesh_verts, rot_vector, scale, trans)

        params = [scale, trans, rot_vector]

        R = self.convert_rot_vector(rot_vector)
        f = 1200
        depth = f / scale
        C = [0, 0, depth]  # init pos
        camRT = R.T.dot(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))  # first to trans the axis,  second to R->RT
        camCenter = camRT.dot(C)
        campos = np.hstack([camRT, camCenter.reshape(3, 1)])
        campos = np.vstack([campos, np.array([0, 0, 0, 1]).reshape(1, 4)])
        print("campos:", campos)
        return campos, trans

    def convert_rot_vector(self, rot_vec):
        # -- 1. rot-vector -> rot matrix --
        theta = np.linalg.norm(rot_vec)  # 2-norm
        with np.errstate(invalid='ignore'):
            v = rot_vec / theta
            v = np.nan_to_num(v).reshape(3, 1)  # nan->0 inf->big number
        t = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.cos(theta) * (np.eye(3)) + (1 - np.cos(theta)) * (v).dot(v.T) + np.sin(theta) * t
        return R

    # ================================= inner functions ==================================
    def _optimize_rigid_pos_2d(self, scale, trans, rot_vector, lm_pos_3D, lm_pos):
        lm_pos_3D = lm_pos_3D.ravel()
        lm_pos = lm_pos.ravel()
        params = np.hstack((scale, trans, rot_vector))
        result = least_squares(self._compute_res_rigid, params, verbose=0,
                               x_scale='jac', ftol=1e-5, method='lm',
                               args=(lm_pos_3D, lm_pos))
        return result.x[0], result.x[1:3], result.x[3:6]

    def _rotate(self, points, rot_vec):
        """Rotate points by given rotation vectors.
        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vec)  # 2 - norm
        with np.errstate(invalid='ignore'):
            v = rot_vec / theta
            v = np.nan_to_num(v)  # nan->0 inf->big number
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + \
               (points.dot(v.T) * (1 - cos_theta)).dot(v)

    def project(self, points, rot_vec, scale, trans, keepz=False):  # 这就是一个正交投影的过程，成像面在xy上。
        points_proj = self._rotate(points, rot_vec.reshape(1, 3))
        points_proj = points_proj * scale
        if keepz:
            points_proj[:, 0:2] = points_proj[:, 0:2] + trans
        else:
            points_proj = points_proj[:, 0:2] + trans
        return points_proj

    def _compute_res_rigid(self, params, lm_pos_3D, lm_pos):
        lm_pos_3D = lm_pos_3D.reshape(-1, 3)
        lm_proj = self.project(lm_pos_3D, params[3:6], params[0], params[1:3])
        return lm_proj.ravel() - lm_pos


def get_image_hull_mask(image_shape, image_landmarks, ie_polys=None):
    # get the mask of the image
    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int)

    # hull_mask = np.zeros(image_shape[0:2]+(1,), dtype=np.float32)
    hull_mask = np.full(image_shape[0:2] + (1,), 0, dtype=np.float32)

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[0:9],
                        int_lmrks[17:18]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[8:17],
                        int_lmrks[26:27]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:20],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[24:27],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[19:25],
                        int_lmrks[8:9],
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:22],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[22:27],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    # nose
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(int_lmrks[27:36]), (1,))
    hull_mask = cv2.transpose(hull_mask)
    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)
    return hull_mask


def merge_add_mask(img_1, mask):
    if mask is not None:
        height = mask.shape[0]
        width = mask.shape[1]
        channel_num = 1
        for row in range(height):
            for col in range(width):
                for c in range(channel_num):
                    if mask[row, col] == 0:
                        mask[row, col] = 0
                    else:
                        mask[row, col] = 255
        mask = mask
        r_channel, g_channel, b_channel = cv2.split(img_1)
        r_channel = cv2.bitwise_and(r_channel, mask)
        g_channel = cv2.bitwise_and(g_channel, mask)
        b_channel = cv2.bitwise_and(b_channel, mask)
        res_img = cv2.merge((r_channel, g_channel, b_channel))
    else:
        res_img = img_1
    return res_img

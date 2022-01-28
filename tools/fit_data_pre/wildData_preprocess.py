import argparse
import os
import sys

import imageio
import numpy as np


def project(lm3d, pose, debug=True):
    K = np.array([
        [1200, 0, 256],
        [0, 1200, 256],
        [0, 0, 1]
    ])
    # get Rt
    Rt = np.eye(4)
    M = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Rt[:3, :3] = pose[:3, :3].T  # pose[:3,:3] --> RT
    Rt[:3, 3] = -pose[:3, :3].T.dot(pose[:3, 3])  # T = R.dot(C)

    # project 3d to 2d
    lm2d = K @ Rt[:3, :] @ (np.concatenate([lm3d, np.ones([lm3d.shape[0], 1])], 1).T)
    lm2d_half = lm2d // lm2d[2, :]
    lm2d = np.round(lm2d_half).astype(np.int64)[:2, :].T @ M[:2, :2]  # .T[:,:2]  #[68,2]

    lm2d[:, 1] = 512 + lm2d[:, 1]
    print(lm2d.max(), lm2d.min())
    if debug == True:
        img = np.zeros([256, 256, 3])
        lm2d_view = lm2d.astype(np.int64) // 2
        img[lm2d_view[:, 0], lm2d_view[:, 1], :] = np.ones([3])

        # plt.imshow(img)
        # plt.show()
    return lm2d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filePath", type=str, default=None)
    args = parser.parse_args()
    if not args.filePath:
        print("please input your file path")
        return 0
    else:
        basedir = args.filePath + "/"
    rawDataDir = basedir
    rawDataDir2 = basedir + '/raw/'
    os.makedirs(rawDataDir2, exist_ok=True)
    maskDir = basedir + '/mask/'
    resDir = basedir + '/segRes/'

    IS_ALIGN = True  # if align
    IS_DETECT_MASK = True
    IS_SEGMENT = True
    IS_RELIGHT = True

    if not os.path.exists(resDir):
        os.makedirs(resDir)
    fileList2 = os.listdir(basedir)
    raw_fileList = []
    fileList = []
    for file in fileList2:
        if file[-4:] in ['.JPG', '.jpg', '.png']:
            raw_fileList.append(file)
            if file[-3:] != 'png':
                file = file[:-3] + 'png'
            fileList.append(file)

    if IS_ALIGN:
        from AlignmentCode.wild_fit_base import mfnerf_fitting
        import cv2
        mf_fitter = mfnerf_fitting(lm_file="AlignmentCode/shape_predictor_68_face_landmarks.dat")
        os.makedirs(rawDataDir, exist_ok=True)
        for i, filename in enumerate(raw_fileList):
            img = cv2.imread(basedir + filename)
            # coarse alignment
            kp2d, src_img_scale = mf_fitter.detect_kp2d(img, is_show_img=False)

            # tuning
            pos, trans = mf_fitter.get_pose_from_kp2d(kp2d)
            lm3d_tmplate = mf_fitter.fcFitter.tmpLM.copy()
            lm2d_tmplate = project(lm3d_tmplate, pos)

            kp2d, src_img_scale = mf_fitter.detect_kp2d(cv2.cvtColor(src_img_scale, cv2.COLOR_RGB2BGR),
                                                        tar_kp=lm2d_tmplate, is_show_img=False)
            imageio.imsave(rawDataDir + fileList[i], src_img_scale)
            imageio.imsave(rawDataDir2 + fileList[i], img[:, :, ::-1])
            print("down alignment, ", filename)
            np.save(os.path.join(basedir, "pose_" + filename[:-4] + ".npy"), {"pose": pos, "kp": kp2d})

    if IS_DETECT_MASK:
        sys.path.append("SegmentCode")
        from get_pair_parsing import evaluate
        evaluate(maskDir, rawDataDir, fileList)

    if IS_SEGMENT:
        for filename in fileList:
            img = imageio.imread(rawDataDir + filename)
            mask = imageio.imread(maskDir + filename)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            # hair 255 0 0  cloth 0,255,0
            HAIR_CORLOR = np.array([0, 0, 255]).astype(np.uint8)
            CLOTHES_CORLOR = np.array([0, 255, 0]).astype(np.uint8)
            BG_COLOR = np.array([0, 0, 0]).astype(np.uint8)
            color_list = [CLOTHES_CORLOR, BG_COLOR]
            mymask = (np.ones_like(mask) * 255).astype(np.uint8)
            mymask[450:, ...] = 0
            for color in color_list:
                index = np.where(np.all(mask == color, axis=-1))
                mymask[index[0], index[1]] = 0

            maskImg = np.bitwise_and(mymask, img)
            imageio.imsave(resDir + filename, maskImg)
            print("segmentation with mask over, saving in", resDir + filename)

    if IS_RELIGHT:
        import cv2
        # from myRelighting import trans_get_sh
        from RelightingModule import RelightModule
        resRelDir = basedir + '/segRelRes/'
        if not os.path.exists(resRelDir):
            os.mkdir(resRelDir)
        RelM = RelightModule()
        for filename in fileList:
            img = imageio.imread(resDir + filename)
            LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            img, sh = RelM.trans_get_sh(img)
            # plt.imshow(img)
            # plt.imshow(mymask)
            # plt.show()
            imageio.imsave(resRelDir + filename, img)
            print("relighting over, saving in ", resRelDir + filename)
            # plt.imshow(img)
            # plt.show()
            # continue


if __name__ == "__main__":
    main()

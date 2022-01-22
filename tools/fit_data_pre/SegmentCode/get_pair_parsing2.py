# -*- encoding: utf-8 -*-
import csv
import glob
import os
import os.path as osp

import cv2
import face_alignment
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
# import matplotlib.pyplot as plt
import sys
sys.path.append("./tools/fit_data_pre/")
from utils import *
from model import BiSeNet
from SegmentCode.process_data_asian import draw_landmark_contour, get_landmark_from_img


def vis_parsing_maps(h, w, im, parsing_anno, stride, save_path, save_im=False):
    # Colors for all 20 parts
    # part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
    #                [255, 0, 85], [255, 0, 170],
    #                [0, 255, 0], [85, 255, 0], [170, 255, 0],
    #                [0, 255, 85], [0, 255, 170],
    #                [85, 255, 255], [85, 0, 255], [170, 0, 255],
    #                [0, 85, 255], [0, 170, 255],
    #                [255, 255, 0], [255, 255, 85], [255, 255, 170],
    #                [255, 0, 255], [255, 85, 255], [255, 170, 255],
    #                [0, 255, 255], [0, 0, 255], [170, 255, 255]]

    # part_colors = [[0, 0, 0], [0, 0, 255], [255, 0, 0],
    #                [150, 30, 150], [255, 65, 255],
    #                [150, 80, 0], [170, 120, 65], [125, 125, 125],
    #                [255, 255, 0], [0, 255, 255],
    #                [255, 150, 0], [255, 225, 120], [255, 125, 125],
    #                [200, 100, 100], [0, 255, 0],
    #                [0, 150, 80], [215, 175, 125], [220, 180, 210],
    #                [125, 125, 255]]

    part_colors = [[0, 0, 0], [255, 0, 0], [150, 30, 150], [255, 65, 255],
                   [150, 80, 0], [170, 120, 65], [220, 180, 210], [255, 125, 125],
                   [200, 100, 100], [215, 175, 125], [125, 125, 125], [255, 150, 0],
                   [255, 255, 0], [0, 255, 255], [255, 225, 120], [125, 125, 255],
                   [0, 255, 0], [0, 0, 255], [0, 150, 80]
                   ]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    # vis_parsing_anno[vis_parsing_anno == 18] = 0
    # vis_parsing_anno[vis_parsing_anno == 17] = 0
    # vis_parsing_anno[vis_parsing_anno == 14] = 0
    # vis_parsing_anno[vis_parsing_anno == 15] = 0
    # vis_parsing_anno[vis_parsing_anno == 16] = 0
    # vis_parsing_anno[vis_parsing_anno == 4] = 0

    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    # 新增的一段，用于纠正错误的左右眉毛和眼睛
    index_nose = np.where(vis_parsing_anno == 10)
    index_lefteb = np.where(vis_parsing_anno == 2)
    index_righteb = np.where(vis_parsing_anno == 3)
    index_lefteye = np.where(vis_parsing_anno == 4)
    index_righteye = np.where(vis_parsing_anno == 5)
    index_leftear = np.where(vis_parsing_anno == 7)
    index_rightear = np.where(vis_parsing_anno == 8)

    nose_x = np.mean(index_nose[1])
    if index_lefteb:
        ind_false = np.where(index_lefteb[1] < nose_x)
        if ind_false:
            vis_parsing_anno[index_lefteb[0][ind_false], index_lefteb[1][ind_false]] = 3

    if index_righteb:
        ind_false = np.where(index_righteb[1] > nose_x)
        if ind_false:
            vis_parsing_anno[index_righteb[0][ind_false], index_righteb[1][ind_false]] = 2

    if index_lefteye:
        ind_false = np.where(index_lefteye[1] < nose_x)
        if ind_false:
            vis_parsing_anno[index_lefteye[0][ind_false], index_lefteye[1][ind_false]] = 5

    if index_righteye:
        ind_false = np.where(index_righteye[1] > nose_x)
        if ind_false:
            vis_parsing_anno[index_righteye[0][ind_false], index_righteye[1][ind_false]] = 4

    if index_leftear:
        ind_false = np.where(index_leftear[1] < nose_x)
        if ind_false:
            vis_parsing_anno[index_leftear[0][ind_false], index_leftear[1][ind_false]] = 8

    if index_rightear:
        ind_false = np.where(index_rightear[1] > nose_x)
        if ind_false:
            vis_parsing_anno[index_rightear[0][ind_false], index_rightear[1][ind_false]] = 7

    for pi in range(0, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    # vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.0, vis_parsing_anno_color, 1.0, 0)

    vis_im = vis_parsing_anno_color
    # Save result or not
    if save_im:
        # cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno*10)
        vis_im = cv2.resize(vis_im, (w, h))
        cv2.imwrite(save_path, vis_im[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_im


# def evaluate(respth='/home/zhang/zydDataset/faceRendererData/foreground_p/', dspth='/home/zhang/zydDataset/faceRendererData/data512x512/', cp='Seg_79999_iter.pth'):
def evaluate(respth='', dspth='', img_list=None, cp='Seg_79999_iter.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    # image_list = sorted(glob.glob(dspth + "*.png"))  # 0310
    # image_list = sorted(glob.glob(dspth + "*/*.png"))  # 0311
    # image_list = sorted(glob.glob(dspth + "*.jpg"))  # 0412
    # image_list = sorted(glob.glob(dspth + "/*.png"))  # 0517
    # image_list.extend( sorted(glob.glob(dspth + "/*.JPG")) )
    # image_list.extend(sorted(glob.glob(dspth + "/*.jpg")))
    if img_list is not None:
        image_list = []
        for imgName in img_list:
            image_list.append(os.path.join(dspth, imgName))
    else:
        image_list = sorted(glob.glob(dspth + "/*.png"))  # 0517
        image_list.extend( sorted(glob.glob(dspth + "/*.JPG")) )
        image_list.extend(sorted(glob.glob(dspth + "/*.jpg")))
        print('len_img:',len(image_list), "in", dspth)
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('./tools/fit_data_pre/trained_model/', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for i in range(0, len(image_list)):   # len(image_list)
            image_path = image_list[i]
            # subdir = image_path.split("/")[-2]
            name = image_path.split("/")[-1]

            save_dir = respth
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            img = Image.open(image_path)
            h, w, c = np.array(img).shape
            image = img.resize((512, 512), Image.BILINEAR)
#============================================================lw7.15
            # 试试先进行normalize
            # temp = normalize_SEAN(np.array(image), 0)
            # image = Image.fromarray(temp)
#=============================================================lw7.15
            # image = image.rotate(0, expand=1, fillcolor=(0, 0, 0), translate=(-10*int(pose_ind), 0))   # 0311

            # 为了测试侧脸图像加入的
            # image = image.rotate(0, expand=1, fillcolor=(255, 255, 255), translate=(-80, 0))

            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            # print(np.unique(parsing))
            # name2 = subdir + "_" + name
            name2 = name

            ## respth2 = respth + subdir + "/"

            if not os.path.exists(respth):   # 0313
                os.makedirs(respth)          # 0313
            vis_im = vis_parsing_maps(h, w, image, parsing, stride=1, save_path=osp.join(respth, name2), save_im=False)   # 0311
            print(i, image_path)
            Image.fromarray(vis_im).save(osp.join(respth, name2))        # 保存图像

            ## croppth2 = croppth + subdir + "/"

            # if croppth != "":
            #     if not os.path.exists(croppth):
            #         os.makedirs(croppth)
            #
            #     crop, full_color, ret = get_stylegan_parsing_pair(np.array(image), vis_im)
            #     if ret > -1:
            #         Image.fromarray(crop).save(osp.join(croppth, name2))   # 保存图像

                # # 保存毕业设计用图
                # landmark, ret = get_landmark_from_img(np.array(image))
                # if ret > -1:
                #     landmark_img, mask = draw_landmark(np.array(image), landmark)
                #     landmark_path = croppth.replace("crop", "landmark")
                #     mask_path = croppth.replace("crop", "mask")
                #     if not os.path.exists(landmark_path):
                #         os.makedirs(landmark_path)
                #     if not os.path.exists(mask_path):
                #         os.makedirs(mask_path)
                #     Image.fromarray(landmark_img).save(osp.join(landmark_path, name2))
                #     Image.fromarray(mask).save(osp.join(mask_path, name2))
                #

                # landmark, ret = get_landmark_from_img(np.array(image))
                # if ret < 0:
                #     continue
                # mask = draw_landmark_contour(np.array(image), landmark)
                # crop = vis_im*np.uint8(mask/255)
                #
                # crop2 = align_crop_full(crop, vis_im)
                #
                # Image.fromarray(crop2).save(osp.join(croppth, name2))

def get_faceparsing_partial(img):
    landmark, ret = get_landmark_from_img(img)
    if ret < 0:
        return img, ret
    mask = draw_landmark_contour(img, landmark)
    # mask2 = np.expand_dims(mask, axis=2)
    # mask2 = np.concatenate((mask2, mask2, mask2), axis=2)/255
    img = img * np.uint8(mask / 255)

    img[np.where(mask == 0)] = 255

    return img, ret


def refine_landmark_using_parsing(landmark, label_parsing):
    for i in range(0, 17):
        row = int(landmark[i, 0])
        col = int(landmark[i, 1])
        index = np.where(label_parsing[:, col] == 1)
        if not np.any(index[0]):
            index = np.where(label_parsing[row, :] == 1)
            if not index:
                continue

            left = np.min(index[0])
            right = np.max(index[0])
            if abs(col - left) < abs(col - right):
                landmark[i, 1] = left
            else:
                landmark[i, 1] = right

        bottom = np.max(index[0])
        landmark[i, 0] = bottom

    return landmark


def get_celeba_parsing_pair(image, label_celeba):
    label_parsing = celeba_label2parsinglabel(label_celeba)
    color = parsing_label2color(label_parsing)
    landmark, ret = get_landmark_from_img(image)

    # if ret < 0:
    #     return None, ret

    # refine dlib landmark
    # landmark = refine_landmark_using_parsing(landmark, label_parsing)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, face_detector='dlib')

    preds = fa.get_landmarks(image)
    landmark = preds[0]

    mask = draw_landmark_contour(image, landmark)
    crop = color * np.uint8(mask / 255)

    return crop, color, 0


def parsing_Color2label(img):
    color_list = [[0, 0, 0], [255, 0, 0], [150, 30, 150], [255, 65, 255],
                  [150, 80, 0], [170, 120, 65], [220, 180, 210], [255, 125, 125],
                  [200, 100, 100], [215, 175, 125], [125, 125, 125], [255, 150, 0],
                  [255, 255, 0], [0, 255, 255], [255, 225, 120], [125, 125, 255],
                  [0, 255, 0], [0, 0, 255], [0, 150, 80]
                  ]

    label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(0, len(color_list)):  # len(colors)
        color = color_list[i]
        index = np.where(np.all(img == color, axis=-1))
        label[index[0], index[1]] = i

    return label


def parsing_label2celeba(label):
    map_list = [0, 1, 6, 7, 4, 5, 3, 8, 9, 15, 2, 10, 11, 12, 17, 16, 18, 13, 14]
    res = label.copy()
    for i in range(0, len(map_list)):
        index = np.where(label == i)
        res[index[0], index[1]] = map_list[i]

    return res


def celeba_label2color(label):
    color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0],
                  [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204],
                  [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0],
                  [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204],
                  [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    res = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(color_list):
        res[label == idx] = color

    return res


def celeba_color2label(img):
    color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0],
                  [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204],
                  [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0],
                  [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204],
                  [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(0, len(color_list)):  # len(colors)
        color = color_list[i]
        index = np.where(np.all(img == color, axis=-1))
        label[index[0], index[1]] = i

    return label


def parsing_label2color(label):
    color_list = [[0, 0, 0], [255, 0, 0], [150, 30, 150], [255, 65, 255],
                  [150, 80, 0], [170, 120, 65], [220, 180, 210], [255, 125, 125],
                  [200, 100, 100], [215, 175, 125], [125, 125, 125], [255, 150, 0],
                  [255, 255, 0], [0, 255, 255], [255, 225, 120], [125, 125, 255],
                  [0, 255, 0], [0, 0, 255], [0, 150, 80]
                  ]
    res = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(color_list):
        res[label == idx] = color

    return res


def celeba_label2parsinglabel(label):
    map_list = [0, 1, 6, 7, 4, 5, 3, 8, 9, 15, 2, 10, 11, 12, 17, 16, 18, 13, 14]
    res = label.copy()
    for i in range(0, len(map_list)):
        index = np.where(label == map_list[i])
        res[index[0], index[1]] = i

    return res


def main():
    # 制作celeba的训练集  crop-->full
    lis = sorted(glob.glob(
        "/run/user/1000/gvfs/afp-volume:host=CITE-3D.local,user=anonymous,volume=share/zhangyidi/CelebA-HQ-img/*.jpg"))
    label_dir = "/run/user/1000/gvfs/afp-volume:host=CITE-3D.local,user=anonymous,volume=share/zhangyidi/CelebAMask-HQ-mask/"
    tgt_full_color_dir = "/run/user/1000/gvfs/afp-volume:host=CITE-3D.local,user=anonymous,volume=share/zhangyidi/celeba_full_parsing/"
    tgt_crop_dir = "/run/user/1000/gvfs/afp-volume:host=CITE-3D.local,user=anonymous,volume=share/zhangyidi/celeba_crop/"
    for i in range(0, 10):
        name = str(i) + ".jpg"
        imgname = "/run/user/1000/gvfs/afp-volume:host=CITE-3D.local,user=anonymous,volume=share/zhangyidi/CelebA-HQ-img/" + name
        # name = lis[i].split("/")[-1]
        name = name.replace("jpg", "png")
        img = np.array(Image.open(lis[i]).resize((512, 512)))
        label = np.array(Image.open(label_dir + name))
        crop, full_color, ret = get_celeba_parsing_pair(img, label)
        if ret < 0:
            continue

        Image.fromarray(crop).save(tgt_crop_dir + name)
        Image.fromarray(full_color).save(tgt_full_color_dir + name)
        print(i)


def get_stylegan_parsing_pair(image, color):
    landmark, ret = get_landmark_from_img(image)

    if ret < 0:
        return None, None, ret

    ## refine dlib landmark
    ## landmark = refine_landmark_using_parsing(landmark, label_parsing)

    # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, face_detector='dlib')
    #
    # preds = fa.get_landmarks(image)
    # if not preds:
    #     return None, color, -1
    # landmark = preds[0]

    mask = draw_landmark_contour(image, landmark)
    # crop = color * np.uint8(mask / 255)

    mask[mask > 0] = 1.0
    crop = color * np.uint8(mask)

    # 对齐crop前后的脸部轮廓
    crop_label = parsing_label2celeba(parsing_Color2label(crop))
    full_label = parsing_label2celeba(parsing_Color2label(color))

    index1 = np.where(crop_label == 2)  # 鼻子
    if (not index1) or len(index1[0]) == 0:
        return crop, color, 0
    row = np.min(index1[0])  # 以前是鼻子的最上面一行，这里改成鼻子最上面一行再减去20

    index1 = np.where((crop_label == 6) | (crop_label == 7))  # 眉毛
    if (not index1) or len(index1[0]) == 0:
        row = row
    else:
        row = min(row, np.mean(index1[0]))  # row是眉毛的最下沿，如果没有眉毛，则是鼻子的最上沿

    # index_face_full = np.where(full_label == 1)   # skin
    # index = ([i for i in index_face_full[0] if i > row],
    #          [index_face_full[1][j] for j in range(0, len(index_face_full[1])) if index_face_full[0][j] > row])
    # #     print(len(index[0]), len(index[1]))
    # crop_label[index[0], index[1]] = full_label[index[0], index[1]]

    # 以前没考虑到鼻子超出轮廓的情况，这里是新增的逻辑
    lis = [1, 2, 3, 4, 5, 6, 7, 11, 12]
    for ind in lis:
        index_face_full = np.where(full_label == ind)  # skin
        if (not index_face_full) or len(index_face_full[0]) == 0:
            continue
        index = ([i for i in index_face_full[0] if i > row],
                 [index_face_full[1][j] for j in range(0, len(index_face_full[1])) if index_face_full[0][j] > row])
        #     print(len(index[0]), len(index[1]))
        crop_label[index[0], index[1]] = full_label[index[0], index[1]]

    # crop_final = celeba_label2color(crop_label)
    crop_final = parsing_label2color(celeba_label2parsinglabel(crop_label))

    return crop_final, color, 0
    # return crop, color, 0


def main_0309():
    # 使用stylegan生成器生成的数据得到  crop-->full
    lis = sorted(glob.glob("/home/zhang/zydDataset/faceRendererData/temp_imgs/0309/stylegan_data_jpg/*.jpg"))
    color_dir = "/home/zhang/zydDataset/faceRendererData/temp_imgs/0309/stylegan_data_semantic/"
    tgt_crop_dir = "/home/zhang/zydDataset/faceRendererData/temp_imgs/0309/stylegan_semantic_crop/"
    for i in range(0, len(lis)):
        # name = str(i) + ".jpg"
        name = lis[i].split("/")[-1]
        img = np.array(Image.open(lis[i]).resize((512, 512)))
        label = np.array(Image.open(color_dir + name.replace("jpg", "png")))
        crop, full_color, ret = get_stylegan_parsing_pair(img, label)
        # cv2.imshow("img", crop[:, :, ::-1])
        # cv2.waitKey()

        if ret < 0:
            continue

        Image.fromarray(crop).save(tgt_crop_dir + name.replace("jpg", "png"))  # 语义分割图务必保存成png格式
        # Image.fromarray(full_color).save(tgt_full_color_dir + name)
        print(i)


# 生成SEAN可以测试的数据集
def get_facescape_label():
    # image_dir = "/home/zhang/zydDataset/faceRendererData/testResults/2_facerender-pix2pix-hair/0324_addhair_3/"
    # save_path = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0324_addhair_3/test_label2/"
    # rgbimg_path = "/home/zhang/zydDataset/faceRendererData/testResults/0_facerenderer-pix2pix/0324/"
    # img_path = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0324_addhair_3/test_img/"

    image_dir = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/2_facerender-pix2pix-hair/0517_addhair/"
    save_path = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/3_SEAN/datasets/0517_addhair/test_label2/"
    rgbimg_path = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/0_facerenderer-pix2pix/0517/"
    img_path = "/run/user/1000/gvfs/smb-share:server=cite-3d.local,share=share/zhangyidi/FaceRendererData/testResults/3_SEAN/datasets/0517_addhair/test_img/"

    lis = sorted(glob.glob(image_dir + "*.png"))
    for i in range(0, len(lis)):
        # subdir = lis[i].split("/")[-2]
        name = lis[i].split("/")[-1]
        src_image_path = lis[i]
        src_parsing = np.array(Image.open(src_image_path))
        src_label = parsing_Color2label(src_parsing)
        celeba_label = parsing_label2celeba(src_label)

        # ==================== temp test ===================
        celeba_label[celeba_label == 8] = 9
        # celeba_label[celeba_label == 8] = 13
        # celeba_label[celeba_label == 9] = 13
        # ==================================================

        # 这里先存储到test_label2而不是直接存储到test_label里的原因是，原来的模型输出结果会在脸部多出来一圈，需要使用郭的代码与crop重新对齐
        # 　新版模型没有这个问题了，所以可以直接保存到test_label里
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save_name = os.path.join(save_path, subdir + "_" + name)
        save_name = os.path.join(save_path, name)

        Image.fromarray(celeba_label).save(save_name)

        # rgb_image = cv2.imread("/home/zhang/zydDataset/faceRendererData/testResults/0_facerenderer-pix2pix/0125/" + subdir + "/" + name)[:, :, ::-1]
        rgb_image = cv2.imread(rgbimg_path + name)[:, :, ::-1]

        rgb_image = normalize_SEAN(rgb_image)
        rgb_image = Image.fromarray(rgb_image)
        # rgb_image = rgb_image.rotate(0, expand=1, fillcolor=(255, 255, 255), translate=(-60, 0))

        if not os.path.exists(img_path):
            os.makedirs(img_path)
        rgb_image.save(save_name.replace("test_label2", "test_img"))
        print(i)


def get_facescape_label_for_skin():
    # 将crop中的彩色语义分割转换为celeba的label，用于SEAN提取皮肤参数时的mask
    image_dir = "/home/zhang/zydDataset/faceRendererData/testResults/1_face-parsing/0324/crop/"
    lis = sorted(glob.glob(image_dir + "*.png"))
    for i in range(0, len(lis)):
        # subdir = lis[i].split("/")[-2]
        name = lis[i].split("/")[-1]
        src_image_path = lis[i]
        src_parsing = np.array(Image.open(src_image_path))
        src_label = parsing_Color2label(src_parsing)
        celeba_label = parsing_label2celeba(src_label)
        save_path = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0324/test_label3/"
        #
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, name)

        Image.fromarray(celeba_label).save(save_name)
        print(i)


# 生成SEAN可以测试的数据集
def get_facescape_label_0311():
    image_dir = "/home/zhang/zydDataset/faceRendererData/testResults/2_facerender-pix2pix-hair/0325/"
    lis = sorted(glob.glob(image_dir + "*/*.png"))  # 0311
    for i in range(0, len(lis)):
        name = lis[i].split("/")[-1]
        pose_ind = lis[i].split("/")[-2]  # pose_ind  0311
        src_image_path = lis[i]
        src_parsing = np.array(Image.open(src_image_path))
        src_label = parsing_Color2label(src_parsing)
        celeba_label = parsing_label2celeba(src_label)

        # ==================== temp test ===================
        # celeba_label[celeba_label == 8] = 13
        # celeba_label[celeba_label == 9] = 13
        # ==================================================
        celeba_label[celeba_label == 8] = 9
        save_path = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0325/" + pose_ind + "/test_label/"  # 0311
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save_name = os.path.join(save_path, subdir + "_" + name)
        save_name = os.path.join(save_path, name)

        Image.fromarray(celeba_label).save(save_name)

        rgb_image = cv2.imread(
            "/home/zhang/zydDataset/faceRendererData/testResults/0_facerenderer-pix2pix/0325/" + pose_ind + "/" + name)[
                    :,
                    :, ::-1]  # 0311
        rgb_image = normalize_SEAN(rgb_image)
        rgb_image = Image.fromarray(rgb_image)
        rgb_image = rgb_image.rotate(0, expand=1, fillcolor=(255, 255, 255), translate=(-10 * int(pose_ind), 0))  # 0311

        img_path = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_0325/" + pose_ind + "/test_img/"
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        rgb_image.save(save_name.replace("test_label", "test_img"))
        print(i)


def get_celeba_label():
    csv_name = "/home/zhang/PycharmProjects/facerenderer-pix2pix-hair/dataset/asianAndCeleba_train.csv"
    src_image_dir = "/run/user/1000/gvfs/afp-volume:host=CITE-3D.local,user=anonymous,volume=share/zhangyidi/celeba_full_parsing/"
    tgt_image_dir = "/home/zhang/zydDataset/faceRendererData/celeba_full_parsing_label/"
    with open(csv_name, 'r') as f:
        reader = csv.reader(f)
        image_filenames = list(reader)

    for i in range(0, len(image_filenames)):
        image_name = image_filenames[i][0]
        img = np.array(Image.open(src_image_dir + image_name))
        new_label = parsing_label2celeba(parsing_Color2label(img))
        color = celeba_label2color(new_label)
        # cv2.imwrite("./celeba.png", color[:, :, ::-1])
        save_name = tgt_image_dir + image_name
        # Image.fromarray(new_label).save(save_name)
        print(save_name)


def evaluate_mask():
    # 验证补全前后的mask的人脸形状是否一致，如果不一致，则用补全前的对应部分替代
    src_image_dir = "/home/zhang/zydDataset/faceRendererData/TUFront_parsing_crop/"
    # tgt_image_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_front/test_label/"
    tgt_image_dir = "/home/zhang/zydDataset/faceRendererData/TUFront_parsing_predicted/"
    lis_src = sorted(glob.glob(src_image_dir + "*_*/*.png"))

    for i in range(0, 20):
        mask_src = cv2.imread(lis_src[i], 0)

        # name = lis_src[i].split("/")[-2] + "_" + lis_src[i].split("/")[-1]
        # tgt_name = tgt_image_dir + name
        tgt_name = lis_src[i].replace(src_image_dir, tgt_image_dir)

        mask_tgt = cv2.imread(tgt_name, 0)

        diff = abs(mask_tgt - mask_src)
        cv2.imwrite("/home/zhang/zydDataset/faceRendererData/rawscan_mask_diff{}.png".format(i), diff * 50)


def evaluate_celeba_mask():
    # 验证celeba mask的人脸形状是否一致
    src_image_dir = "/home/zhang/zydDataset/faceRendererData/celeba_crop_label/"
    # tgt_image_dir = "/home/zhang/PycharmProjects/SEAN/datasets/facescape_front/test_label/"
    tgt_image_dir = "/home/zhang/zydDataset/faceRendererData/celeba_full_parsing_label/"
    lis_src = sorted(glob.glob(src_image_dir + "*.png"))

    for i in range(0, 20):
        # i = 3
        mask_src = cv2.imread(lis_src[i], 0)

        # name = lis_src[i].split("/")[-2] + "_" + lis_src[i].split("/")[-1]
        # tgt_name = tgt_image_dir + name
        tgt_name = lis_src[i].replace(src_image_dir, tgt_image_dir)

        mask_tgt = cv2.imread(tgt_name, 0)

        diff = abs(mask_tgt - mask_src)
        cv2.imwrite("/home/zhang/zydDataset/faceRendererData/mask_diff{}.png".format(i), diff * 50)


if __name__ == "__main__":
    # evaluate_celeba_mask()
    # evaluate_mask()
    # evaluate()
    # evaluate(dspth="/home/zhang/zydDataset/faceRendererData/data512x512_2000/")
    # main()

    get_facescape_label()
    # get_facescape_label_0311()
    # get_facescape_label_for_skin()

    # main_0309()

    # get_celeba_label()
    # im = cv2.imread("/home/zhang/zydDataset/faceRendererData/celeba_full_parsing_label/19973.png", 0)

    # count = 0
    # image_dir = "/home/zhang/zydDataset/faceRendererData/celeba_full_parsing_label/"
    # for i in range(0, 20000):
    #     if os.path.isfile(image_dir + str(i) + ".png"):
    #         im = np.array(Image.open(image_dir + str(i) + ".png").convert('L'))
    #         if np.max(im) > 18:
    #             count += 1
    #             print(count)

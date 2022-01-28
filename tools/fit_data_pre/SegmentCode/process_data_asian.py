# -*- coding: utf-8 -*-
import csv
import glob
import os
import os.path as osp

import cv2
import dlib
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import BiSeNet


# 生成合适的csv文件
def get_csv(image_dir, csv_name):
    f = open(csv_name, 'w', encoding='utf-8')
    csv_writer = csv.writer(f)

    for i in range(1, 2001):
        image_name = str(i).zfill(5) + ".jpg"
        full_path = image_dir + image_name
        if os.path.isfile(full_path):
            csv_writer.writerow([image_name])
        else:
            print(image_name)

    f.close()


# dlib检测关键点
def get_landmark_from_img(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    POINTS_NUM_LANDMARK = 68

    dets = detector(img, 1)
    if len(dets) == 0:
        print("no face in the fake image")
        return None, -1
    ret = 0
    rectangle = dets[0]

    # left, top, right, bottom = max(0, rectangle.left()), max(0, rectangle.top()), min(w, rectangle.right()), min(h, rectangle.bottom())
    landmark_shape = predictor(img, rectangle)
    landmark_arr = np.zeros((68, 2))
    for i in range(0, POINTS_NUM_LANDMARK):
        landmark_arr[i, 0] = landmark_shape.part(i).x  # x
        landmark_arr[i, 1] = landmark_shape.part(i).y  # y

    # # convert to relative coordinate
    # x = landmark_arr[0, 0]
    # y = landmark_arr[1, 0]
    # landmark_arr[0, :] = landmark_arr[0, :] - x
    # landmark_arr[1, :] = landmark_arr[1, :] - y

    return landmark_arr, ret


def get_bbox_dlib(img):
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)
    if len(dets) == 0:
        print("no face in the fake image")
        return -1, None
    ret = 0
    rectangle = dets[0]
    h, w, c = img.shape

    left, top, right, bottom = max(0, rectangle.left()), max(0, rectangle.top()), min(w, rectangle.right()), min(h,
                                                                                                                 rectangle.bottom())

    return ret, [top, left, bottom - top, right - left]


def draw_landmark(img, landmark_arr):
    point_size = 2
    point_color = (0, 255, 0)
    thickness = 4
    r, c = landmark_arr.shape
    if r > c:
        for i in range(0, r):
            x = landmark_arr[i, 0]
            y = landmark_arr[i, 1]
            cv2.circle(img, (int(x), int(y)), point_size, point_color, thickness)
            cv2.putText(img, str(i), (int(x) + 1, int(y) + 1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    elif r < c:
        for i in range(0, c):
            x = landmark_arr[0, i]
            y = landmark_arr[1, i]
            cv2.circle(img, (int(x), int(y)), point_size, point_color, thickness)
            cv2.putText(img, str(i), (int(x) + 1, int(y) + 1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("image", img)
    cv2.waitKey()


# 验证contour格式，用dlib的关键点代替contour
def draw_landmark_contour(img_ori, landmark):
    img = img_ori.copy()

    landmark2 = np.zeros((27, 2))
    for i in range(0, 17):
        landmark2[i, 0] = landmark[i, 0]
        landmark2[i, 1] = landmark[i, 1]

    ind = 17
    for i in range(26, 16, -1):
        landmark2[ind, 0] = landmark[i, 0]
        landmark2[ind, 1] = landmark[i, 1]
        ind += 1

    # landmark2 = np.expand_dims(landmark2, axis=1)
    landmark2 = landmark2.reshape((-1, 1, 2)).astype(np.int32)
    print(landmark2.shape)

    contours = [landmark2]
    ind = 0
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, ind, (255, 255, 255), -1)
    # 腐蚀
    kernal = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernal, iterations=1)

    return mask


# 抠出不规则人脸区域
def get_face_img(img):
    landmark, ret = get_landmark_from_img(img)
    if ret < 0:
        return img, ret
    mask = draw_landmark_contour(img, landmark)
    # mask2 = np.expand_dims(mask, axis=2)
    # mask2 = np.concatenate((mask2, mask2, mask2), axis=2)/255
    img = img * np.uint8(mask / 255)

    img[np.where(mask == 0)] = 255

    return img, ret


# 从去除背景的图中获取bbox
def get_bbox(img_gray):
    index = np.where(img_gray < 240)
    left, right, top, bottom = np.min(index[1]), np.max(index[1]), np.min(index[0]), np.max(index[0])
    h, w = bottom - top, right - left
    return [top, left, h, w]


# 从去除背景的图中得到512x512的图
def get_img_512x512(img_name):
    PIL_img = Image.open(img_name)
    img = np.array(PIL_img)
    h, w, c = img.shape

    img_gray = np.array(PIL_img.convert('L'))
    bbox = get_bbox(img_gray)

    face_rec = 480.0

    if bbox[2] > face_rec or bbox[3] > face_rec:
        scale = min(face_rec / bbox[2], face_rec / bbox[3])
        PIL_img = PIL_img.resize((int(w * scale), int(h * scale)))
        img = np.array(PIL_img)
        img_gray = np.array(PIL_img.convert('L'))
        bbox = get_bbox(img_gray)

    img_ret = np.ones((512, 512, 3), dtype='uint8') * 255
    top, left, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
    img_ret[256 - h // 2:256 - h // 2 + h, 256 - w // 2:256 - w // 2 + w, :] = img[top:top + h, left:left + w, :]

    return img_ret


# face_rec不一样，这个是处理语义分割的图像，使之占满全图
def get_img_512x512_2(PIL_img):
    img = np.array(PIL_img)
    h, w, c = img.shape

    img_gray = np.array(PIL_img.convert('L'))
    bbox = get_bbox(img_gray)

    face_rec = 510

    # if bbox[2] > face_rec or bbox[3] > face_rec:
    scale = min(face_rec / bbox[2], face_rec / bbox[3])
    PIL_img = PIL_img.resize((int(w * scale), int(h * scale)))
    img = np.array(PIL_img)
    img_gray = np.array(PIL_img.convert('L'))
    bbox = get_bbox(img_gray)

    img_ret = np.ones((512, 512, 3), dtype='uint8') * 255
    top, left, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
    h = min(h, 512)
    w = min(w, 512)
    img_ret[256 - h // 2:256 - h // 2 + h, 256 - w // 2:256 - w // 2 + w, :] = img[top:top + h, left:left + w, :]

    return img_ret


def batch_process():
    # image_dir = "/home/zhang/zydDataset/faceRendererData/data512x512/"
    # csv_name = "./dataset/celeba_train2.csv"
    # f = open(csv_name, 'w', encoding='utf-8')
    # csv_writer = csv.writer(f)
    #
    # for i in range(1, 1151):
    #     image_name = str(i).zfill(5) + ".jpg"
    #     full_path = image_dir + image_name
    #     if os.path.isfile(full_path):
    #         csv_writer.writerow([image_name])
    #     else:
    #         print(image_name)
    #
    # f.close()

    path_csv_file = "/home/zhang/PycharmProjects/facerenderer-pix2pix-hair/dataset/scan_test.csv"

    tgt_dir = "/home/zhang/zydDataset/faceRendererData/scan_crop_test/"
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)

    with open(path_csv_file, 'r') as f:
        reader = csv.reader(f)
        image_paths = list(reader)
    print(len(image_paths))
    for i in range(0, len(image_paths)):
        imagename = image_paths[i]
        # img = cv2.imread(os.path.join(image_dir, imagename[0]))
        img = cv2.imread(imagename[0])

        new_img, ret = get_face_img(img)
        if ret < 0:
            continue

        cv2.imwrite(os.path.join(tgt_dir, str(i) + ".jpg"), new_img)


# 从含有背景的图中检测人脸，保证人脸占200，然后进行语义分割，去除背景
# 注意，最终图像不一定是512x512
def get_asian_face_512x512(image_name):
    PIL_img = Image.fromarray(cv2.imread(image_name)[:, :, ::-1])

    if PIL_img is None:
        return None, -1

    ret, bbox = get_bbox_dlib(np.array(PIL_img))
    if ret < 0:
        return None, -1
    if bbox[2] < 100 or bbox[3] < 100:
        return None, -1

    face_rec = 200.0
    h, w, c = np.array(PIL_img).shape

    scale = max(face_rec / bbox[2], face_rec / bbox[3])
    PIL_img = PIL_img.resize((int(w * scale), int(h * scale)))
    img = np.array(PIL_img)
    new_h, new_w, _ = img.shape
    ret, bbox = get_bbox_dlib(img)
    if ret < 0:
        return None, -1

    img_ret = np.ones((512, 512, 3), dtype='uint8') * 255
    top, left, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
    center_x = left + w // 2
    center_y = top + h // 2
    # img_ret[256-h//2:256-h//2+h, 256-w//2:256-w//2+w, :] = img[top:top+h, left:left+w, :]

    new_im = img[max(0, center_y - 356): min(new_h, center_y + 156), max(0, center_x - 256):min(new_w, center_x + 256)]
    new_im2, mask = evaluate(Image.fromarray(new_im), image_name.split("/")[-1])

    return new_im2, mask, 0


def evaluate(img, imgname, respth='/home/zhang/zydDataset/asianFaces/foreground/', cp='Seg_79999_iter.pth',
             delete_list=[0], save_im=False):
    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():

        # lis = image_path.split("/")
        # subdir = '/'.join(lis[0:len(lis) - 1])
        save_dir = respth
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # img = Image.open(osp.join(dspth, imgname))
        h, w, c = np.array(img).shape
        # cv2.imwrite("./img.png", np.array(img))

        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing)
        # print(np.unique(parsing))

        vis_im, mask = vis_parsing_maps(h, w, image, parsing, stride=1, save_path=osp.join(respth, imgname),
                                        save_im=save_im, delete_list=delete_list)
    return vis_im, mask


def vis_parsing_maps(h, w, im, parsing_anno, stride, save_path, save_im=False, delete_list=[0]):
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)

    for num in delete_list:
        vis_parsing_anno[vis_parsing_anno == num] = 0
    # vis_parsing_anno[vis_parsing_anno == 0] = 0
    # # vis_parsing_anno[vis_parsing_anno == 18] = 0  # hat
    # # vis_parsing_anno[vis_parsing_anno == 16] = 0  # cloth
    vis_parsing_anno[vis_parsing_anno > 0] = 1

    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    mask = np.expand_dims(vis_parsing_anno, axis=2)

    mask = np.concatenate((mask, mask, mask), axis=2)
    vis_im = im * mask
    vis_im[np.where(mask == 0)] = 255
    vis_im = np.uint8(vis_im)
    # Save result or not
    if save_im:
        # cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno*10)
        vis_im = cv2.resize(vis_im, (w, h))
        # vis_im = get_img_512x512_2(Image.fromarray(vis_im))
        cv2.imwrite(save_path, vis_im[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_im, mask


def rotate(name, angle):
    img = Image.open(name)
    rotated = img.rotate(angle, expand=1, fillcolor=(255, 255, 255))
    return rotated


def translateTocenter(name):
    img = Image.open(name)
    width, height = img.size
    ret, bbox = get_bbox_dlib(np.array(img))
    if ret < 0:
        return None, -1
    x = width // 2 - (bbox[1] + bbox[3] // 2)
    # x = 100
    translated = img.rotate(0, expand=1, fillcolor=(255, 255, 255), translate=(x, 0))
    return translated, 0


if __name__ == "__main__":

    # 从爬虫图像中检测人脸，并crop出包含脖子和头发的人脸部分
    # lis = sorted(glob.glob("/home/zhang/zydDataset/asianFaces/filter_1207/*.jpg"))
    #
    # for i in range(0, len(lis)):
    #     print(i, lis[i])
    #     lastname = lis[i].split("/")[-1]
    #     init = cv2.imread(lis[i])
    #     if init is None:
    #         os.rename(lis[i], "/home/zhang/zydDataset/asianFaces/filter_1207已处理/" + lastname)
    #         continue
    #     img, mask, ret = get_asian_face_512x512(lis[i])
    #     if ret < 0:
    #         print(lis[i])
    #         os.rename(lis[i], "/home/zhang/zydDataset/asianFaces/delete/" + str(i) + lastname)
    #     else:
    #         os.rename(lis[i], "/home/zhang/zydDataset/asianFaces/filter_1207已处理/" + lastname)

    # ----------------------------------------------------------------------------

    # 把不规则人脸图像转换到512x512
    # lis = sorted(glob.glob("/home/zhang/zydDataset/asianFaces/foreground/*.jpg"))
    # for i in range(0, len(lis)):
    #     print(i, lis[i])
    #     # pil = Image.open(lis[i])
    #     pil, ret = translateTocenter(lis[i])
    #     if ret < 0:
    #         continue
    #     img = get_img_512x512_2(pil)
    #     name = lis[i].split("/")[-1]
    #     Image.fromarray(img).save("/home/zhang/zydDataset/asianFaces/foreground_512x512/" + name)

    # 从512x512人脸图像中抠脸
    # lis = glob.glob("/home/zhang/zydDataset/asianFaces/foreground_512x512/*_2.jpg")
    # for i in range(0, len(lis)):
    #     print(i, lis[i])
    #     name = lis[i].split("/")[-1]
    #     im = cv2.imread(lis[i])
    #     img, ret = get_face_img(im)
    #     if ret < 0:
    #         continue
    #     tgt_dir = "/home/zhang/zydDataset/asianFaces/crop_512x512/"
    #     cv2.imwrite(tgt_dir + name, img)

    # 写入csv文件
    # lis = sorted(glob.glob("/home/zhang/zydDataset/asianFaces/crop_p/*.jpg"))
    # csv_name = "/home/zhang/PycharmProjects/facerenderer-pix2pix-hair/dataset/asian_train4.csv"
    # f = open(csv_name, 'w', encoding='utf-8')
    # csv_writer = csv.writer(f)
    # for i in range(0, 4500):
    #     name = lis[i].split("/")[-1]
    #     csv_writer.writerow([name])
    #     print(i)
    # f.close()

    # 从celeba图片中检测前景，并抠出人脸
    lis = sorted(glob.glob("/home/zhang/zydDataset/faceRendererData/data512x512/*.jpg"))
    for i in range(0, 10000):
        print(i, lis[i])
        name = lis[i].split("/")[-1]
        delete_list = [0]
        img = Image.open(lis[i])
        savepath_A = "/home/zhang/zydDataset/faceRendererData/train_A/"
        savepath_B = "/home/zhang/zydDataset/faceRendererData/train_B/"
        vis_im, _ = evaluate(img, name, respth=savepath_B, delete_list=delete_list)

        img, ret = get_face_img(vis_im)
        if ret < 0:
            continue
        Image.fromarray(img).save(savepath_A + name)
        Image.fromarray(vis_im).save(savepath_B + name)

    # 从foreground图片中去除衣服信息，并得到人脸和脖子区域的mask
    # lis = sorted(glob.glob("/home/zhang/zydDataset/asianFaces/foreground_512x512/*.jpg"))
    # for i in range(0, len(lis)):
    #     print(i, lis[i])
    #     name = lis[i].split("/")[-1]
    #     delete_list = [0, 16, 18]
    #     img = Image.open(lis[i])
    #     savepath = "/home/zhang/zydDataset/asianFaces/foreground_no_cloth/"
    #     vis_im, _ = evaluate(img, name, respth=savepath, delete_list=[0, 16, 18])
    #     Image.fromarray(vis_im).save(savepath + name)
    #     _, mask = evaluate(img, name, respth=savepath, delete_list=[0, 16, 17, 18])
    #     Image.fromarray(mask*255).save("/home/zhang/zydDataset/asianFaces/mask_512x512/"+name)

    # 按照SPADE的格式生成数据集
    # lis = sorted(glob.glob("/home/zhang/zydDataset/asianFaces/crop_512x512/*.jpg"))
    # for i in range(0, 4500):
    #     print(i, lis[i])
    #     name = lis[i].split("/")[-1]
    #     dir1 = "/home/zhang/zydDataset/asianFaces/foreground_512x512/"
    #     if not os.path.isfile(dir1 + name):
    #         continue
    #     crop_img = cv2.imread(lis[i])
    #     cv2.imwrite("/home/zhang/PycharmProjects/SPADE/datasets/faceimgs_with_cloth/train_label/"+name, crop_img)
    #
    #     fore_img = cv2.imread(dir1 + name)
    #     cv2.imwrite("/home/zhang/PycharmProjects/SPADE/datasets/faceimgs_with_cloth/train_img/"+name, fore_img)

    # 重新将crop的人脸处理成512x512，脸占图像比例基本一致，并按同样变换处理带有头发的数据
    # def get_crop_img_512x512(img_name):
    #     PIL_img = Image.open(img_name)
    #     img = np.array(PIL_img)
    #     h_, w_, c_ = img.shape
    #
    #     img_gray = np.array(PIL_img.convert('L'))
    #     bbox = get_bbox(img_gray)
    #
    #     face_rec = 260.0
    #
    #     scale = min(face_rec / bbox[2], face_rec / bbox[3])
    #     PIL_img = PIL_img.resize((int(w_ * scale), int(h_ * scale)))
    #     img = np.array(PIL_img)
    #     img_gray = np.array(PIL_img.convert('L'))
    #     bbox = get_bbox(img_gray)
    #
    #     img_ret = np.ones((512, 512, 3), dtype='uint8') * 255
    #     top, left, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
    #     new_h, new_w = int(h_ * scale), int(w_ * scale)
    #     x = new_w//2 - (left + w//2)
    #     y = new_h//3*2 - (top + h//2)
    #     translated = PIL_img.rotate(0, expand=1, fillcolor=(255, 255, 255), translate=(x, y))
    #     translated_arr = np.array(translated)
    #
    #     new_left = max(0, new_w//2-256)
    #     new_top = max(0, new_h//2-256)
    #     new_right = min(new_w//2+256, new_w)
    #     new_bottom = min(new_h//2+256, new_h)
    #     final_w = new_right - new_left
    #     final_h = new_bottom - new_top
    #
    #     img_ret[256-final_h//2:256-final_h//2+final_h, 256-final_w//2:256-final_w//2+final_w,:] = translated_arr[new_top:new_top+final_h, new_left:new_left+final_w,:]
    #
    #     centerx = new_left + final_w//2
    #     centery = new_top + final_h//2
    #     return img_ret, centerx, centery, x, y, scale
    #
    # lis = sorted(glob.glob("/home/zhang/zydDataset/asianFaces/crop_512x512/*.jpg"))
    # for i in range(0, len(lis)):
    #     print(i, lis[i])
    #     name = lis[i].split("/")[-1]
    #
    #     crop_ret, centerx, centery, x, y, scale = get_crop_img_512x512(lis[i])
    #     Image.fromarray(crop_ret).save("/home/zhang/zydDataset/asianFaces/crop_512x512_2/" + name)
    #
    #
    #     new_left = max(0, centerx-256)
    #     new_top = max(0, centery-256)
    #     new_right = min(centerx+256, int(512*scale))
    #     new_bottom = min(centery+256, int(512*scale))
    #     final_w = new_right - new_left
    #     final_h = new_bottom - new_top
    #
    #     img_ret = np.ones((512, 512, 3), dtype='uint8') * 255
    #     PIL_img = Image.open("/home/zhang/zydDataset/asianFaces/foreground_512x512/"+name)
    #     PIL_img = PIL_img.resize((int(512 * scale), int(512 * scale)))
    #     translated = PIL_img.rotate(0, expand=1, fillcolor=(255, 255, 255), translate=(x, y))
    #     translated_arr = np.array(translated)
    #     img_ret[256 - final_h // 2:256 - final_h // 2 + final_h, 256 - final_w // 2:256 - final_w // 2 + final_w,:] = translated_arr[new_top:new_top + final_h, new_left:new_left + final_w, :]
    #
    #     Image.fromarray(img_ret).save("/home/zhang/zydDataset/asianFaces/foreground_512x512_2/" + name)

    # img = get_img_512x512("/home/zhang/zydDataset/faceRendererData/17.jpg")
    # img = cv2.imread("/home/zhang/zydDataset/faceRendererData/02680.jpg")
    # img2, ret = get_face_img(img)
    # cv2.imwrite("/home/zhang/zydDataset/faceRendererData/2680.jpg", img2)  # [:,:,::-1]

    # image_dir = "/home/zhang/zydDataset/faceRendererData/data512x512_2000/"
    # csv_name = "./dataset/celeba_train.csv"
    # # get_csv(image_dir, csv_name)
    #
    # path_csv_file = "./dataset/celeba_train.csv"
    #
    # tgt_dir = "/home/zhang/zydDataset/faceRendererData/data512x512_crop/"
    #
    # # with open(path_csv_file, 'r') as f:
    # #     reader = csv.reader(f)
    # #     image_paths = list(reader)
    # #
    # # for imagename in image_paths:
    # #     img = cv2.imread(os.path.join(image_dir, imagename[0]))
    # #
    # #     new_img, ret = get_face_img(img)
    # #     if ret < 0:
    # #         continue
    # #
    # #     cv2.imwrite(os.path.join(tgt_dir, imagename[0]), new_img)
    #
    #
    # get_csv("/home/zhang/zydDataset/faceRendererData/data512x512_crop/", "./dataset/celeba_train_crop2.csv")

    # batch_process()

    # lis = sorted(glob.glob("/home/zhang/zydDataset/faceRendererData/rawscan_masked/*/*/*.jpg"))
    # csv_name = "/home/zhang/PycharmProjects/facerenderer-pix2pix-hair/dataset/scan_test.csv"
    # f = open(csv_name, 'w', encoding='utf-8')
    # csv_writer = csv.writer(f)
    # for i in range(10000, 13000, 3):
    #     image_path = lis[i]
    #     # image_name = str(i).zfill(5) + ".jpg"
    #     # full_path = "/home/zhang/zydDataset/faceRendererData/data512x512/" + image_name
    #     if os.path.isfile(image_path):
    #         csv_writer.writerow([image_path])
    #         print(i)
    # f.close()

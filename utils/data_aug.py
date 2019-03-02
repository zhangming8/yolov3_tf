# coding: utf-8

import numpy as np
import random
import cv2
import glob
import xml.etree.cElementTree as ET


def random_translate(img, bboxes, p=0.5):
    if random.random() < p:
        h_img, w_img, _ = img.shape
        # 得到可以包含所有bbox的最大bbox
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w_img - max_bbox[2]
        max_d_trans = h_img - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w_img, h_img))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
    return img, bboxes


def random_crop(img, bboxes, p=0.5):
    if random.random() < p:
        h_img, w_img, _ = img.shape
        # 得到可以包含所有bbox的最大bbox
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w_img - max_bbox[2]
        max_d_trans = h_img - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        img = img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
    return img, bboxes


def random_horizontal_flip(img, bboxes, p=0.5):
    if random.random() < p:
        _, w_img, _ = img.shape
        img = img[:, ::-1, :]
        bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
    return img, bboxes


# 下面的增强来自https://zhuanlan.zhihu.com/p/54566524
# 随机变换亮度 (概率：0.5)
def random_bright(im, bboxes, p=0.5, delta=32):
    if random.random() < p:
        delta = random.uniform(-delta, delta)
        im += delta
        im = im.clip(min=0, max=255)
    return im, bboxes


# 随机变换通道
def random_swap(im, bboxes):
    perms = ((0, 1, 2), (0, 2, 1),
            (1, 0, 2), (1, 2, 0),
            (2, 0, 1), (2, 1, 0))
    if random.random() < 0.5:
        swap = perms[random.randrange(0, len(perms))]
        im = im[:, :, swap]
    return im, bboxes


# 随机变换对比度
def random_contrast(im, bboxes, lower=0.5, upper=1.5):
    if random.random() < 0.5:
        alpha = random.uniform(lower, upper)
        im *= alpha
        im = im.clip(min=0, max=255)
    return im, bboxes


# 随机变换饱和度
def random_saturation(im, bboxes, lower=0.5, upper=1.5):
    if random.random() < 0.5:
        im[:, :, 1] *= random.uniform(lower, upper)
    return im, bboxes


# 随机变换色度(HSV空间下(-180, 180))
def random_hue(im, bboxes, delta=18.0):
    if random.random() < 0.5:
        im[:, :, 0] += random.uniform(-delta, delta)
        im[:, :, 0][im[:, :, 0] > 360.0] -= 360.0
        im[:, :, 0][im[:, :, 0] < 0.0] += 360.0
    return im, bboxes


def rotate_aug(image, bboxes, angle, center=None, scale=1.0, p=0.5):
    '''
    :param img: img image
    :param bboxes: bboxes should be numpy array with [[x1,x2,x3,x4],
                                                    [y1,y2,y3,y4]]
    :param angle:
    :param center:
    :param scale:
    :return: the rotated image and the points
    '''
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    if bboxes is None:
        for i in range(image.shape[2]):
            image[:, :, i] = cv2.warpAffine(image[:, :, i], M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        return image
    else:
        box_x, box_y, box_label, box_tmp = [], [], [], []
        for box in bboxes:
            box_x.append(int(box[0]))
            box_x.append(int(box[2]))
            box_y.append(int(box[1]))
            box_y.append(int(box[3]))
            box_label.append(box[4])
        box_tmp.append(box_x)
        box_tmp.append(box_y)
        bboxes = np.array(box_tmp)
        ####make it as a 3x3 RT matrix
        full_M = np.row_stack((M, np.asarray([0,0,1])))
        img_rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

        ###make the bboxes as 3xN matrix
        full_bboxes = np.row_stack((bboxes, np.ones(shape=(1, bboxes.shape[1]))))
        bboxes_rotated = np.dot(full_M, full_bboxes)

        bboxes_rotated = bboxes_rotated[0:2, :]
        bboxes_rotated = bboxes_rotated.astype(np.int32)

        result = []
        for i in range(len(box_label)):
            x1, y1, x2, y2 = bboxes_rotated[0][2*i], bboxes_rotated[1][2*i], bboxes_rotated[0][2*i+1], bboxes_rotated[1][2*i+1]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
            x1, x2 = min(w, x1), min(w, x2)
            y1, y2 = min(h, y1), min(h, y2)
            one_box = [x1, y1, x2, y2, box_label[i]]
            result.append(one_box)
        return img_rotated, result


def readAnnotations(xml_path):
    et = ET.parse(xml_path)
    element = et.getroot()
    element_objs = element.findall('object')

    results = []
    for element_obj in element_objs:
        result = []
        class_name = element_obj.find('name').text

        obj_bbox = element_obj.find('bndbox')
        x1 = int(round(float(obj_bbox.find('xmin').text)))
        y1 = int(round(float(obj_bbox.find('ymin').text)))
        x2 = int(round(float(obj_bbox.find('xmax').text)))
        y2 = int(round(float(obj_bbox.find('ymax').text)))

        result.append(int(x1))
        result.append(int(y1))
        result.append(int(x2))
        result.append(int(y2))
        result.append(class_name)

        results.append(result)
    return results


if __name__ == "__main__":
    img_list = glob.glob("/Users/ming/Desktop/voc/*.jpg")
    for image_path in img_list:
        img_org = cv2.imread(image_path)
        img = img_org
        bboxes_org = readAnnotations(image_path[:-4] + ".xml")
        print("img: {}, box: {}".format(image_path, bboxes_org))

        img, bboxes = rotate_aug(img, bboxes_org, 180)
        print bboxes
        for box in bboxes:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img, box[4], (box[0], max(20, box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow(image_path, img)
        img = 0
        cv2.waitKey(0)
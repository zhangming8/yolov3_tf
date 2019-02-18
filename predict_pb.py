#coding:utf-8
import os
import cv2
import numpy as np
import glob
import shutil
import time
import tensorflow as tf
from tensorflow.python.platform import gfile
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from utils import utils
import config as cfg


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu 0
model_name = "model_yolov3.pb"
use_tensorrt = False #True
input_name = "input/input_data"
output_name = ["pred_lbbox/pred_bbox", "pred_mbbox/pred_bbox", "pred_sbbox/pred_bbox"]


def get_tf_graph():
    with gfile.FastGFile(model_name,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        print("[INFO] load .pb done...")
    return graph_def


def get_trt_graph(batch_size=32,workspace_size=1<<30):
    import tensorflow.contrib.tensorrt as trt
    precision_mode = "FP32" # 'FP32', 'FP16' and 'INT8'
    # conver pb to FP32pb
    with gfile.FastGFile(model_name,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        trt_graph = trt.create_inference_graph(input_graph_def=graph_def, outputs=output_name,
                                         max_batch_size=batch_size,
                                         max_workspace_size_bytes=workspace_size,
                                         precision_mode=precision_mode)  # Get optimized graph
    print("[INFO] ***********************************create tensorrt model done...")
    return trt_graph


def init_tf():
    global sess, input_data, pred_lbbox_, pred_mbbox_, pred_sbbox_
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if use_tensorrt:
        graph = get_trt_graph(batch_size=1)
    else:
        graph = get_tf_graph()
    tf.import_graph_def(graph, name='')

    # 获取输入tensor
    input_data = tf.get_default_graph().get_tensor_by_name(input_name + ":0")
    # 获取预测tensor
    pred_lbbox_ = tf.get_default_graph().get_tensor_by_name(output_name[0] + ":0")
    pred_mbbox_ = tf.get_default_graph().get_tensor_by_name(output_name[1] + ":0")
    pred_sbbox_ = tf.get_default_graph().get_tensor_by_name(output_name[2] + ":0")


class YoloTest(object):
    def __init__(self):
        self.__test_input_size = cfg.TEST_INPUT_SIZE
        self.__anchor_per_scale = cfg.ANCHOR_PER_SCALE
        self.__classes = cfg.CLASSES
        self.__num_classes = len(self.__classes)
        self.__class_to_ind = dict(zip(self.__classes, range(self.__num_classes)))
        self.__anchors = np.array(cfg.ANCHORS)
        self.__score_threshold = cfg.SCORE_THRESHOLD
        self.__iou_threshold = cfg.IOU_THRESHOLD
        self.__log_dir = os.path.join(cfg.LOG_DIR, 'test')
        self.__annot_dir_path = cfg.ANNOT_DIR_PATH
        self.__moving_ave_decay = cfg.MOVING_AVE_DECAY
        self.__dataset_path = cfg.DATASET_PATH
        self.__valid_scales = cfg.VALID_SCALES
        self.__input_data = input_data
        self.__pred_sbbox, self.__pred_mbbox, self.__pred_lbbox = pred_sbbox_, pred_mbbox_, pred_lbbox_

    def __get_bbox(self, image):
        """
        :param image: 要预测的图片
        :return: 返回NMS后的bboxes，存储格式为(xmin, ymin, xmax, ymax, score, class)
        """
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape
        yolo_input = utils.img_preprocess2(image, None, (self.__test_input_size, self.__test_input_size), False)
        yolo_input = yolo_input[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [self.__pred_sbbox, self.__pred_mbbox, self.__pred_lbbox], feed_dict={self.__input_data: yolo_input})

        sbboxes = self.__convert_pred(pred_sbbox, (org_h, org_w), self.__valid_scales[0])
        mbboxes = self.__convert_pred(pred_mbbox, (org_h, org_w), self.__valid_scales[1])
        lbboxes = self.__convert_pred(pred_lbbox, (org_h, org_w), self.__valid_scales[2])

        # sbboxes = self.__valid_scale_filter(sbboxes, self.__valid_scales[0])
        # mbboxes = self.__valid_scale_filter(mbboxes, self.__valid_scales[1])
        # lbboxes = self.__valid_scale_filter(lbboxes, self.__valid_scales[2])

        bboxes = np.concatenate([sbboxes, mbboxes, lbboxes], axis=0)
        bboxes = utils.nms(bboxes, self.__score_threshold, self.__iou_threshold, method='nms')
        return bboxes

    def __valid_scale_filter(self, bboxes, valid_scale):
        bboxes_scale = np.sqrt(np.multiply.reduce(bboxes[:, 2:4] - bboxes[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
        bboxes = bboxes[scale_mask]
        return bboxes

    def __convert_pred(self, pred_bbox, org_img_shape, valid_scale):
        """
        将yolo输出的bbox信息(x, y, w, h, confidence, probability)进行转换，
        其中(x, y, w, h)是预测bbox的中心坐标、宽、高，大小是相对于input_size的，
        confidence是预测bbox属于物体的概率，probability是条件概率分布
        (x, y, w, h) --> (xmin, ymin, xmax, ymax) --> (xmin_org, ymin_org, xmax_org, ymax_org)
        --> 将预测的bbox中超出原图的部分裁掉 --> 将分数低于score_threshold的bbox去掉
        :param pred_bbox: yolo输出的bbox信息，shape为(1, output_size, output_size, anchor_per_scale, 5 + num_classes)
        :param org_img_shape: 存储格式必须为(h, w)，输入原图的shape
        :return: bboxes
        假设有N个bbox的score大于score_threshold，那么bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
        其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
        """
        pred_bbox = np.array(pred_bbox)
        output_size = pred_bbox.shape[1]
        pred_bbox = np.reshape(pred_bbox, (output_size, output_size, self.__anchor_per_scale, 5 + self.__num_classes))
        pred_xywh = pred_bbox[:, :, :, 0:4]
        pred_conf = pred_bbox[:, :, :, 4:5]
        pred_prob = pred_bbox[:, :, :, 5:]

        # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :, :, :2] - pred_xywh[:, :, :, 2:] * 0.5,
                                    pred_xywh[:, :, :, :2] + pred_xywh[:, :, :, 2:] * 0.5], axis=-1)
        # (2)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
        # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * self.__test_input_size / org_w, 1.0 * self.__test_input_size / org_h)
        dw = (self.__test_input_size - resize_ratio * org_w) / 2
        dh = (self.__test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, :, :, 0::2] = 1.0 * (pred_coor[:, :, :, 0::2] - dw) / resize_ratio
        pred_coor[:, :, :, 1::2] = 1.0 * (pred_coor[:, :, :, 1::2] - dh) / resize_ratio

        # (3)将预测的bbox中超出原图的部分裁掉
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :, :, :2], [0, 0]),
                                    np.minimum(pred_coor[:, :, :, 2:], [org_w - 1, org_h - 1])], axis=-1)

        pred_coor = pred_coor.reshape((-1, 4))
        pred_conf = pred_conf.reshape((-1,))
        pred_prob = pred_prob.reshape((-1, self.__num_classes))

        # (4)去掉不在有效范围内的bbox
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # (4)将score低于score_threshold的bbox去掉
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.__score_threshold

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

        return bboxes

    def detect_image(self, image):
        bboxes = self.__get_bbox(image)
        res = []
        for i,box in enumerate(bboxes):
            x1, y1, x2, y2 = int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3]))
            score = box[4]
            cls = self.__classes[int(box[5])]
            res.append((cls, score, [x1, y1, x2, y2]))
        return res




if __name__ == '__main__':
    init_tf()
    save_dir = "result"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    images = ['./data/' + image for image in os.listdir('./data')
              if (image[-3:] == 'jpg') and (image[0] != '.')]
    yolo = YoloTest()
    for im in images:
        print("-------------------")
        img = cv2.imread(im)
        start = time.time()
        result = yolo.detect_image(img)
        print("time cost:", time.time() - start)
        print(result)
        for res in result:
            cls, prob, x1, y1, x2, y2 = res[0], res[1], res[2][0], res[2][1], res[2][2], res[2][3]
            cv2.putText(img, str(cls) + ": " + str(prob)[:5], (x1, max(y1, 15)), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
        print(im, result)
        cv2.imwrite(save_dir + "/" + os.path.basename(im), img)
        # cv2.imshow(os.path.basename(im), img)
    # cv2.waitKey(0)
    sess.close()

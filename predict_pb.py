#coding:utf-8
import os, cv2
import numpy as np
import tensorflow as tf
import glob

from tensorflow.python.platform import gfile


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def init_tf():
    global sess, pred, x
    sess = tf.Session(config=config)
    with gfile.FastGFile('./model_yolov3.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    # 获取输入tensor
    x = tf.get_default_graph().get_tensor_by_name("input/input_data:0")
    print("input:", x)
    # 获取预测tensor
    pred_lbbox = tf.get_default_graph().get_tensor_by_name("pred_lbbox/pred_bbox:0")
    pred_mbbox = tf.get_default_graph().get_tensor_by_name("pred_mbbox/pred_bbox:0")
    pred_sbbox = tf.get_default_graph().get_tensor_by_name("pred_sbbox/pred_bbox:0")
    print('load model done...')

def evaluate_image(img_dir):
    # read and process image
    image = cv2.imread(img_dir)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # pre-process the image for classification
    image = cv2.resize(image, (IMG_W, IMG_W))
    #image = image.astype("float") / 255.0
    im_mean = np.mean(image)
    stddev = max(np.std(image), 1.0/np.sqrt(IMG_W*IMG_W*3))
    image = (image - im_mean) / stddev #代替tf.image.per_image_standardization
    #image_array = np.array(image)
    image = np.expand_dims(image, axis=0)

    prediction = sess.run(pred, feed_dict={x: image})
    max_index = np.argmax(prediction)
    pred_label = label_dict_res[str(max_index)]
    print("%s, predict: %s(index:%d), prob: %f" %(img_dir, pred_label, max_index, prediction[0][max_index]))
    

if __name__ == '__main__':
    init_tf()
    data_path = "/media/lishundong/DATA2/docker/data/sku_for_classify2/sku_val"
    #data_path = "/media/lishundong/DATA2/docker/data/sku_for_classify2/sku_train"
    label = os.listdir(data_path)
    for l in label:
        if os.path.isfile(os.path.join(data_path, l)):
            continue
        for img in glob.glob(os.path.join(data_path, l, "*.jpg")):
            evaluate_image(img_dir=img)
    sess.close()

#coding:utf-8
import sys, shutil
import tensorflow as tf
from tensorflow.python.framework import graph_util

#import tensorflow.contrib.slim as slim
#sys.path.append("/home/lishundong/Desktop/models/research/slim")
#from nets.mobilenet import mobilenet_v2
#import model
from model.yolo_v3 import YOLO_V3


def print_pb(output_graph_path):
    tf.reset_default_graph()  # 重置计算图
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output_graph_def = tf.GraphDef()
    # 获得默认的图
    graph = tf.get_default_graph()
    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")
        # 得到当前图有几个操作节点
        print("%d ops in the final graph." % len(output_graph_def.node))
        #tensor_name = [tensor.name for tensor in output_graph_def.node]
        #print(tensor_name)
        print('---------------------------')
        for op in graph.get_operations():
            # print出tensor的name和值
            print(op.name, op.values())
    sess.close()


def save_new_ckpt(logs_train_dir, newckpt):
    #x = tf.placeholder(tf.float32, shape=[None, IMG_W, IMG_W, 3], name="input_1")
    x = tf.placeholder(dtype=tf.float32, shape=[None, 544, 544, 3], name='input_1')
    # predict
    #with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
    #with slim.arg_scope(mobilenet_v2.training_scope()):
    #    logit, _ = mobilenet_v2.mobilenet(x, num_classes=N_CLASSES, is_training=False)
    #logit = model.MobileNetV2(x, num_classes=N_CLASSES, is_training=False).output
    #logit = model.model4(x, N_CLASSES, is_trian=False)
    #logit = model.model5(x, N_CLASSES, is_trian=False)
    #logit = model.model7(x, N_CLASSES, is_trian=False)
    #logit = model.model8(x, N_CLASSES, is_trian=False)
    
    __training = tf.placeholder(dtype=tf.bool, name='training')
    _, _, _, __pred_sbbox, __pred_mbbox, __pred_lbbox = YOLO_V3(__training).build_nework(x)    
    #__conv_sbbox, __conv_mbbox, __conv_lbbox, __pred_sbbox, __pred_mbbox, __pred_lbbox = YOLO_V3(False).build_nework(x)

    #pred = tf.nn.softmax(logit, name="softmax_out")

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, logs_train_dir)
    print("load model done...")
    saver.save(sess, newckpt)
    sess.close()
    print('save new model done...')


def freeze_graph(input_checkpoint, output_graph):
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = ["input_1", "conv_lbbox/weight","conv_lbbox/bias", "conv_mbbox/weight", "conv_mbbox/bias", "conv_sbbox/weight", "conv_sbbox/bias"]
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def, # 等于:sess.graph_def
            output_node_names=output_node_names)
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点


def get_label(label_file):
    label_dict, label_dict_res = {}, {}
    with open(label_file, 'r') as f:
        for line in f.readlines():
            folder, label = line.strip().split(':')[0], line.strip().split(':')[1]
            label_dict[folder] = label
            label_dict_res[label] = folder
    print(label_dict)
    return label_dict


if __name__ == "__main__":
    #label_file = "label_mnist.txt"
    #label_file = "label_key.txt"
    #IMG_W = 28
    model_path = './weights/yolo.ckpt-16-6.0901' #load model
    pb_model = "model_yolov3.pb" #save final pb model

    #N_CLASSES = len(get_label(label_file))
    #print("num classes:", N_CLASSES)
    new_model_path = './model_temp/modelnew.ckpt' #save temp model
    save_new_ckpt(model_path, new_model_path)
    freeze_graph(new_model_path, pb_model)
    #print_pb(pb_model)
    print("create %s done..." %(pb_model))
    shutil.rmtree("./model_temp")

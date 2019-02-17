#coding:utf-8
import sys, shutil
import tensorflow as tf
from tensorflow.python.framework import graph_util

#import tensorflow.contrib.slim as slim
#sys.path.append("/home/lishundong/Desktop/models/research/slim")
#from nets.mobilenet import mobilenet_v2
#import model


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
    output_node_names = ["input/input_data", "pred_lbbox/pred_bbox", "pred_sbbox/pred_bbox", "pred_mbbox/pred_bbox"]
    import config as cfg
    from model.yolo_v3 import YOLO_V3
    __training = False
    with tf.name_scope('input'):
        __input_data = tf.placeholder(dtype=tf.float32, name='input_data')
        #__training = tf.placeholder(dtype=tf.bool, name='training')
    _, _, _, __pred_sbbox, __pred_mbbox, __pred_lbbox = YOLO_V3(__training).build_nework(__input_data)
    __moving_ave_decay = cfg.MOVING_AVE_DECAY
    with tf.name_scope('ema'):
        ema_obj = tf.train.ExponentialMovingAverage(__moving_ave_decay)
    saver = tf.train.Saver(ema_obj.variables_to_restore())


#    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True) #得到图、clear_devices ：Whether or not to clear the device field for an `Operation` or `Tensor` during import.
 
    graph = tf.get_default_graph() #获得默认的图
    input_graph_def = graph.as_graph_def()  #返回一个序列化的图代表当前的图
 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        print("loading model ...")
        
        # fix batch norm nodes
       # for node in input_graph_def.node:
       #   if node.op == 'RefSwitch':
       #     #print(node)
       #     node.op = 'Switch'
       #     for index in range(len(node.input)):
       #       if 'moving_' in node.input[index]:
       #         node.input[index] = node.input[index] + '/read'
       #   elif node.op == 'AssignSub':
       #     node.op = 'Sub'
       #     if 'use_locking' in node.attr: del node.attr['use_locking']

        for node in input_graph_def.node:
          if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
              if 'moving_' in node.input[index]:
                node.input[index] = node.input[index] + '/read'
          elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
          elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
          elif node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr: del node.attr['use_locking']
            if 'validate_shape' in node.attr: del node.attr['validate_shape']
            if len(node.input) == 2:
              # input0: ref: Should be from a Variable node. May be uninitialized.
              # input1: value: The value to be assigned to the variable.
              node.input[0] = node.input[1]
              del node.input[1]
 
        #print ("predictions : ", sess.run("predictions:0", feed_dict={"input_holder:0": [10.0]})) # 测试读出来的模型是否正确，注意这里传入的是输出 和输入 节点的 tensor的名字，不是操作节点的名字
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph.as_graph_def(), # 等于:sess.graph_def
            output_node_names=output_node_names)
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点


def freeze_graph2(input_checkpoint, pb_model):
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True) 
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, input_checkpoint)
        
        output_node_names = ["input/input_data", "pred_lbbox/pred_bbox", "pred_sbbox/pred_bbox", "pred_mbbox/pred_bbox"]
        
        # for fixing the bug of batch norm
        gd = sess.graph.as_graph_def()
        for node in gd.node:            
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']
        
        converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, output_node_names)
        tf.train.write_graph(converted_graph_def, "./", pb_model, as_text=False)
        sess.close()


if __name__ == "__main__":
    model_path = './weights/yolo.ckpt-1-6.4084' #load model
    pb_model = "model_yolov3.pb" #save final pb model

    freeze_graph(model_path, pb_model)
    #print_pb(pb_model)
    print("create %s done..." %(pb_model))

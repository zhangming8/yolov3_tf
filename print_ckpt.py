#coding:utf-8

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

#使用NewCheckpointReader来读取ckpt里的变量
checkpoint_path = "weights/yolo.ckpt-0-7.2752"
roi_name = "bbox"

reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()

a = {}
for key, value in var_to_shape_map.items():
    print("tensor_name: {}, shape: {}".format(key, value))
    if roi_name in key:
        a[key] = value
        #print("tensor_name: {}, shape: {}".format(key, value))

print("++++++++++++++++++++++++++")
for k,v in a.items():
    print(k, v)


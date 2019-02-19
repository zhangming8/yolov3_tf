#!/bin/bash

# convert yolov3.weights to yolov3_608_coco_pretrained.ckpt
python3.5 convert_weights.py --weights_file=yolov3.weights --dara_format=NHWC --ckpt_file=./saved_model/yolov3_608_coco_pretrained.ckpt

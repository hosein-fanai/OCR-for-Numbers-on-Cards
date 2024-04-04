import yaml

import multiprocessing

import time

import os


with open("./config.yaml") as stream:
    configs = yaml.safe_load(stream)

input_shape = tuple(int(dim) for dim in configs["input_shape"].split(' '))

num_poolings = configs["num_poolings"]
window_size = input_shape[0]//2**num_poolings, input_shape[1]//2**num_poolings

convert_bboxes_to_relative_bboxes = configs["convert_bboxes_to_relative_bboxes"]

num_anchors = configs["num_anchors"]
num_classes = configs["num_classes"]

batch_size = configs["batch_size"]
lr = float(configs["lr"])
lr_phase_2 = lr / float(configs["training_phase_2_lr_divisor"])
reg_coef = float(configs["reg_coef"])
dropout_rate = configs["dropout_rate"]
# class_weights_obj = 1. # float(window_size[0] * window_size[1])

use_data_aug = configs["use_data_aug"]

train_with_masks = configs["train_with_masks"]
training_phase_2 = configs["training_phase_2"]

threshold_conf = configs["threshold_conf"]
# threshold_nms = configs["threshold_nms"]

num_generating = configs["num_generating"]
generating_index = configs["generating_index"]

num_processes = multiprocessing.cpu_count()

num_generating_ID = num_generating // 4
num_generating_credit = num_generating - num_generating_ID
num_generating_en = num_generating_credit // 2

runtime_id = time.asctime().replace(":", "-")

log_dir = "./logs/" + runtime_id
monitor_metric = configs["monitor_metric"]

model_name = configs["model_name"]
model_path = "./models/" + model_name + "_" + runtime_id + ".h5"
model_plot_path = "./models/plots/" + model_name + "_" + runtime_id + ".png"

template_path = "./templates"

trainset_path = "./dataset/trainset"
testset_path = "./dataset/testset"

nums_en = list("0123456789")
nums_per = list("۰۱۲۳۴۵۶۷۸۹")

os.makedirs(os.path.join(trainset_path, "images"), exist_ok=True)
os.makedirs(os.path.join(trainset_path, "annotations"), exist_ok=True)
os.makedirs(testset_path, exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./models/plots", exist_ok=True)


import multiprocessing

import time


input_shape = (200, 320, 3)

num_poolings = 6
window_size = input_shape[0]//2**num_poolings, input_shape[1]//2**num_poolings

convert_bboxes_to_relative_bboxes = True

num_anchors = 10
num_classes = 10

batch_size = 64
lr = 1e-3
reg_coef = 4e-4
dropout_rate = 0.5
class_weights_obj = 1. # float(window_size[0] * window_size[1])

threshold_conf = 0.5
# threshold_nms = 0.5

num_generating = 200_000
generating_index = 0

num_processes = multiprocessing.cpu_count()

num_generating_ID = int(num_generating // 3)
num_generating_credit = int(num_generating // (3/2))
num_generating_en = int(num_generating_credit // 2)

runtime_id = time.asctime().replace(":", "-")

log_dir = "./logs/" + runtime_id
monitor_metric = "val_bboxes_mae"

model_name = "sliding_window_ocr_model"
model_path = "./models/" + model_name + "_" + runtime_id + ".h5"
model_plot_path = "./models/plots/" + model_name + "_" + runtime_id + ".png"

template_path = "./templates"

trainset_path = "./dataset/trainset"
testset_path = "./dataset/testset"

nums_en = list("0123456789")
nums_per = list("۰۱۲۳۴۵۶۷۸۹")



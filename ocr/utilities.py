import cv2

from PIL import Image

import numpy as np

from matplotlib import pyplot as plt

from constants import input_shape, window_size, num_anchors, threshold_conf, convert_bboxes_to_relative_bboxes


def read_img(img_path):
    img = Image.open(img_path).convert("RGB").resize(input_shape[1::-1])
    img = np.array(img).astype("uint8")

    return img


def IoU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = min(x1 + w1, x2 + w2) - x_intersection
    h_intersection = min(y1 + h1, y2 + h2) - y_intersection
    
    if w_intersection <= 0 or h_intersection <= 0:
        return 0.0
    
    intersection_area = w_intersection * h_intersection
    
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    
    union_area = area_box1 + area_box2 - intersection_area
    
    iou_score = intersection_area / union_area

    return iou_score


def calculate_corresponding_window(bbox):
    x_center = bbox[0] + bbox[2] / 2
    y_center = bbox[1] + bbox[3] / 2

    grid_x = int(x_center / (input_shape[1] / window_size[1]))
    grid_y = int(y_center / (input_shape[0] / window_size[0]))

    return grid_x, grid_y


def calculate_corresponding_anchor(bbox, window):
    grid_x, grid_y = window

    window_length = (input_shape[1] / window_size[1])
    window_height = (input_shape[0] / window_size[0])

    window_start_x = window_length * grid_x
    window_start_y = window_height * grid_y

    window_anchors = []
    for anchor in range(num_anchors):
        y = int(window_start_y)
        w = int(window_length / num_anchors)
        h = int(window_height)
        x = int(window_start_x + w * anchor)
        window_anchors.append((x, y, w, h))

    anchors_iou = []
    anchors_idx = []
    for i, anchor in enumerate(window_anchors):
        iou_score = IoU(bbox, anchor)
        anchors_iou.append(iou_score)
        anchors_idx.append(i)

    anchors_iou_and_idx = zip(anchors_iou, anchors_idx)
    anchors_sorted_iou_and_idx = sorted(anchors_iou_and_idx, key=lambda x: x[0], reverse=True)
    anchors_sorted_iou = [idx for _, idx in anchors_sorted_iou_and_idx]

    return anchors_sorted_iou


def normalize_bbox(bbox, input_shape=input_shape):
    return bbox[0]/input_shape[1], bbox[1]/input_shape[0], bbox[2]/input_shape[1], bbox[3]/input_shape[0]


def denormalize_bbox(bbox, input_shape=input_shape):
    return int(bbox[0]*input_shape[1]), int(bbox[1]*input_shape[0]), int(bbox[2]*input_shape[1]), int(bbox[3]*input_shape[0])


def convert_bbox_to_relative_bbox(bbox):
    x, y, w, h = bbox
    grid_x, grid_y = calculate_corresponding_window(bbox)

    window_length = (input_shape[1] / window_size[1])
    window_height = (input_shape[0] / window_size[0])

    window_start_x = window_length * grid_x
    window_start_y = window_height * grid_y

    window_x = (x - window_start_x) / window_length
    window_y = (y - window_start_y) / window_height

    window_w = w / window_length
    window_h = h / window_height

    bbox = (window_x, window_y, window_w, window_h)

    return bbox


def convert_relative_bbox_to_bbox(bbox, window):
    window_x, window_y, window_w, window_h = bbox
    grid_x, grid_y = window

    window_length = (input_shape[1] / window_size[1])
    window_height = (input_shape[0] / window_size[0])

    window_start_x = window_length * grid_x
    window_start_y = window_height * grid_y

    x = int(window_start_x + window_x * window_length)
    y = int(window_start_y + window_y * window_height)

    w = int(window_w * window_length)
    h = int(window_h * window_height)

    bbox = (x, y, w, h)

    return bbox


def read_annotation(annot_path):
    with open(annot_path, 'r') as file:
        lines = file.read().split('\n')

    info = lines[0].split(' ')
    card_type, cvv2, exp_date = int(info[0]), info[1], info[2]
    
    classes = []
    bboxes = []
    for line in lines[1:-1]:
        numbers = line.split(' ')
        class_id, *bbox = numbers
        
        classes.append(int(class_id))
        bboxes.append(denormalize_bbox([float(box) for box in bbox]))

    return classes, bboxes, card_type, cvv2, exp_date


def add_bbox_on_img(img, bboxes, labels=None):
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        img = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (255, 0, 0), 0)

        if labels is not None:
            text_position = (x + int(w / 2) - 5, y - 1)    
            img = cv2.putText(img.copy(), str(labels[i]), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 0)
    
    return img


def show_img(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def show_annotated_img(img_path, annot_path):
    classes, bboxes, _, _, _ = read_annotation(annot_path)
    
    img = read_img(img_path)
    img = add_bbox_on_img(img, bboxes, labels=classes)
    
    show_img(img)


def detect_and_crop_card(image_path):
    pass


def predict(img, model):
    results = model.predict(img[None])

    card_type = results.get("card_type", -1)
    cvv2 = [results.get(f"cvv2_digit_{i}", None) for i in range(4)]
    exp_date = [results.get(f"exp_date_digit_{i}", None) for i in range(8)]

    card_type = card_type[0][0] if card_type!=-1 else card_type
    cvv2 = "".join([str(digit[0].argmax()) for digit in cvv2]) if cvv2[0] is not None else -1
    exp_date = "".join([str(digit[0].argmax()) for digit in exp_date]) if exp_date[0] is not None else -1

    found_conf, found_bboxes, found_classes = [], [], []
    for i in range(window_size[0]):
        for j in range(window_size[1]):
            for k in range(num_anchors):
                conf = results[f"confs_anchor_{k}"][0][i, j, 0]
                if conf > threshold_conf:
                    cls = results[f"classes_anchor_{k}"][0][i, j]
                    bbox = results[f"bboxes_anchor_{k}"][0][i, j]

                    if convert_bboxes_to_relative_bboxes:
                        denormed_bbox = convert_relative_bbox_to_bbox(bbox, window=(j, i))
                    else:
                        denormed_bbox = (int(bbox[0]*input_shape[1]), int(bbox[1]*input_shape[0]), \
                            int(bbox[2]*input_shape[1]), int(bbox[3]*input_shape[0]))

                    found_conf.append(float(f'{conf:.2f}'))
                    found_bboxes.append(denormed_bbox)
                    found_classes.append(cls.argmax(axis=-1))

    return found_conf, found_bboxes, found_classes, card_type, cvv2, exp_date


def predict_and_show(model, img_path, crop=False):
    if crop:
        img = detect_and_crop_card(img_path)
    else:
        img = read_img(img_path)

    found_conf, found_bboxes, found_classes, card_type, cvv2, exp_date = predict(img, model)

    print(f"Confidence Scores: {found_conf}")
    print(f"Found Classes: {found_classes}")
    print(f"Found Objects #: {len(found_bboxes)}")
    print()
    print(f"Card Type: {card_type}")
    print(f"CVV2: {cvv2}")
    print(f"Expiration Date: {exp_date}")

    img = add_bbox_on_img(img, found_bboxes, found_classes)
    show_img(img)



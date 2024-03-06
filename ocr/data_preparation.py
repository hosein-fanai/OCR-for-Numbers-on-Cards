import numpy as np

import multiprocessing

import os

from ocr.constants import input_shape, window_size, trainset_path, num_anchors, convert_bboxes_to_relative_bboxes, num_processes
from ocr.utilities import read_annotation, calculate_corresponding_window, convert_bbox_to_relative_bbox, calculate_corresponding_anchor


def extract_annotation(annot_path):
    classes, bboxes, card_type, cvv2, exp_date = read_annotation(os.path.join(trainset_path, "annotations", annot_path))

    confs = np.zeros((*window_size, 1*num_anchors), dtype="float32")
    normed_windowed_bboxes = {f"bboxes_anchor_{i}": np.zeros((*window_size, 4), dtype="float32") for i in range(num_anchors)}
    classes_dict = {f"classes_anchor_{i}": np.zeros(window_size, dtype="float32") for i in range(num_anchors)}
    for bbox, cls in zip(bboxes, classes):
        grid_x, grid_y = calculate_corresponding_window(bbox)
        if grid_x >= window_size[1] or grid_y >= window_size[0]:
            continue

        if convert_bboxes_to_relative_bboxes:
            normed_bbox = convert_bbox_to_relative_bbox(bbox)
        else:
            normed_bbox = (bbox[0]/input_shape[1], bbox[1]/input_shape[0], bbox[2]/input_shape[1], bbox[3]/input_shape[0])

        anchor_iou_sorted = calculate_corresponding_anchor(bbox, (grid_x, grid_y))
        for anchor_i in anchor_iou_sorted:
            if (normed_windowed_bboxes[f"bboxes_anchor_{anchor_i}"][grid_y, grid_x] == 0).all():
                normed_windowed_bboxes[f"bboxes_anchor_{anchor_i}"][grid_y, grid_x] = normed_bbox
                break

        confs[grid_y, grid_x, anchor_i] = 1
        classes_dict[f"classes_anchor_{anchor_i}"][grid_y, grid_x] = cls

    classes_list = [classes_dict[key] for key in classes_dict.keys()]
    normed_windowed_bboxes_list = [normed_windowed_bboxes[key] for key in normed_windowed_bboxes.keys()]

    cvv2 = [int(digit) for digit in cvv2]
    if len(cvv2)==1:
        cvv2 = [0]*4
    
    exp_date = [int(digit) for digit in exp_date]
    if len(exp_date)==4:
        exp_date = [0]*4 + exp_date

    return confs, normed_windowed_bboxes_list, classes_list, card_type, cvv2, exp_date


def create_annotation_lists(annotation_paths, parallelize=True):
    confs_list = []
    bboxes_list  = []
    all_classes_list = []
    card_types_list  = []
    cvv2s_list  = []
    exp_dates_list = []

    if parallelize:
        print(f"Creating a pool of {num_processes} processes for {len(annotation_paths)} files.")

        pool = multiprocessing.Pool(processes=num_processes)
        results = pool.map(extract_annotation, annotation_paths)
        pool.close()
        pool.join()

        print("Closed the pool. Saving the results ...")

        for result in results:
            confs, normed_windowed_bboxes_list, classes_list, card_type, cvv2, exp_date= result

            confs_list.append(confs)
            bboxes_list.append(normed_windowed_bboxes_list)
            all_classes_list.append(classes_list)
            card_types_list.append(np.array(card_type, dtype="uint8")[None])
            cvv2s_list.append(cvv2)
            exp_dates_list.append(exp_date)
    else:
        for i, annot_path in enumerate(annotation_paths):
            print(f"\rAnnotation file#: {i}", end="")

            confs, normed_windowed_bboxes_list, classes_list, card_type, cvv2, exp_date = extract_annotation(annot_path)
        
            confs_list.append(confs)
            bboxes_list.append(normed_windowed_bboxes_list)
            all_classes_list.append(classes_list)
            card_types_list.append(np.array(card_type, dtype="uint8")[None])
            cvv2s_list.append(cvv2)
            exp_dates_list.append(exp_date)

        print()

    print()

    return confs_list, bboxes_list, all_classes_list, card_types_list, cvv2s_list, exp_dates_list



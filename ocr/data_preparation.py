import tensorflow as tf

from sklearn.model_selection import train_test_split

import numpy as np

import multiprocessing

from ocr.init import input_shape, window_size, num_anchors, convert_bboxes_to_relative_bboxes, num_processes, training_phase_2, batch_size, validation_split
from ocr.utilities import read_annotation, calculate_corresponding_window, convert_bbox_to_relative_bbox, calculate_corresponding_anchor, get_trainset_paths


def extract_annotation(annot_path):
    classes, bboxes, card_type, cvv2, exp_date = read_annotation(annot_path)

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
        print(f"\tCreating a pool of {num_processes} processes for {len(annotation_paths)} files.")

        pool = multiprocessing.Pool(processes=num_processes)
        results = pool.map(extract_annotation, annotation_paths)
        pool.close()
        pool.join()

        print("\tClosed the pool. Saving the results ...")

        for result in results:
            confs, normed_windowed_bboxes_list, classes_list, card_type, cvv2, exp_date = result

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


def split_dataset(trainset_images, trainset_annotations):
    x_train, x_val, y_train, y_val = train_test_split(trainset_images, trainset_annotations, 
                                                    test_size=validation_split, shuffle=True, 
                                                    random_state=np.random.get_state()[1][0])

    return x_train, x_val, y_train, y_val


def create_pipeline(trainset_files, valset_files=None, prefetch=5):
    x_train, train_extracted_annotation_lists = trainset_files
    train_confs, train_bboxes, train_classes, train_card_types, train_cvv2s, train_exp_dates = train_extracted_annotation_lists

    def preprocess_img_annot(img_path, confs, classes_list, bboxes, card_type, cvv2, exp_date):
        x = tf.io.read_file(img_path)
        x = tf.image.decode_image(x)

        y = {
            **{f"confs_anchor_{i}": confs[..., i, None] for i in range(num_anchors)}, 
            **{f"bboxes_anchor_{i}": bboxes[i] for i in range(len(bboxes))}, 
            **{f"classes_anchor_{i}": classes_list[i] for i in range(len(classes_list))}, 
        } | ({"card_type": card_type} if training_phase_2 else {}) | \
            ({f"cvv2_digit_{i}": cvv2[..., i] for i in range(4)} if training_phase_2 else {}) | \
            ({f"exp_date_digit_{i}": exp_date[..., i] for i in range(8)} if training_phase_2 else {})
        
        return x, y

    trainset = tf.data.Dataset.from_tensor_slices((x_train, train_confs, train_classes, train_bboxes, train_card_types, train_cvv2s, train_exp_dates))
    trainset = trainset.map(preprocess_img_annot, num_parallel_calls=tf.data.AUTOTUNE)
    trainset = trainset.shuffle(1_000).batch(batch_size).prefetch(prefetch)

    if valset_files:
        x_val, val_extracted_annotation_lists = valset_files
        val_confs, val_bboxes, val_classes, val_card_types, val_cvv2s, val_exp_dates = val_extracted_annotation_lists

        valset = tf.data.Dataset.from_tensor_slices((x_val, val_confs, val_classes, val_bboxes, val_card_types, val_cvv2s, val_exp_dates))
        valset = valset.map(preprocess_img_annot, num_parallel_calls=tf.data.AUTOTUNE)
        valset = valset.batch(batch_size).prefetch(prefetch)
    else:
        valset = None

    return trainset, valset


def create_dataset():
    trainset_images, trainset_annotations = get_trainset_paths()
    assert len(trainset_images) == len(trainset_annotations)
    print(f"* Found {len(trainset_images)} images in trainset directory.")
    print()

    x_train, x_val, y_train, y_val = split_dataset()

    print(f"* Splitted trainset images into two datasets: trainset #{len(x_train)}, validationset #{len(x_val)}")
    print()

    print("* Started the extraction of annotation files.")
    train_extracted_annotation_lists = create_annotation_lists(y_train)
    val_extracted_annotation_lists = create_annotation_lists(y_val)
    print()

    print("* Creating tf.data.Dataset pipeline for both train and validation data.")
    trainset, valset = create_pipeline((x_train, train_extracted_annotation_lists), (x_val, val_extracted_annotation_lists))
    print("* Successfully created the pipeline.")

    return trainset, valset



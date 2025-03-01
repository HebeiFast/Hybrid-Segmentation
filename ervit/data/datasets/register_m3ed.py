# Copyright (c) Facebook, Inc. and its affiliates.
import os
import json
import logging
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)

def _get_m3ed_meta():
    stuff_classes = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]
    assert len(stuff_classes) == 19

    stuff_colors = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32]]
    assert len(stuff_colors) == 19

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def load_event_sem_seg(gt_root, event_root, image_root, gt_ext="png", event_ext="h5", image_ext="png"):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.
    def file2id(folder_path, file_path):
        # extract relative path starting from `folder_path`
        image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
        # remove file extension
        image_id = os.path.splitext(image_id)[0]
        return image_id

    input_files = []
    event_files = []
    # timestamps = []
    gt_files = []
    for dir in os.listdir(image_root):
        input_dir_path = os.path.join(image_root, dir)
        part_input_files = sorted(
            (os.path.join(input_dir_path, f) for f in PathManager.ls(input_dir_path)),
            key=lambda file_path: file2id(input_dir_path, file_path),
        )
        input_files.extend(part_input_files)

        event_dir_path = os.path.join(event_root, dir)
        event_file = os.path.join(event_dir_path, 'events_rep' + '.' + event_ext)
        event_file = [event_file] * len(part_input_files)
        event_files.extend(event_file)

        gt_dir_path = os.path.join(gt_root, dir)
        part_gt_files = sorted(
            (os.path.join(gt_dir_path, f) for f in PathManager.ls(gt_dir_path) if f.endswith(gt_ext)),
            key=lambda file_path: file2id(gt_dir_path, file_path),
        )
        gt_files.extend(part_gt_files)

    assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)
    assert len(input_files) == len(gt_files), "The length of input_files is not equal to gt_files"
    assert len(input_files) == len(event_files), "The length of input_files is not equal to event_files"
    # assert len(input_files) == len(timestamps), "The length of input_files is not equal to timestamps"
    logger.info(
        "Loaded {} images with semantic segmentation from {}".format(len(input_files), image_root)
    )

    dataset_dicts = []

    for (img_path, event_path, gt_path) in zip(input_files, event_files, gt_files):
        record = {}
        record["file_name"] = img_path
        record["event_file_name"] = event_path
        record["sem_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts


def register_all_m3ed(root):
    event_dir_name = "event_voxel"

    root = os.path.join(root, "m3ed_HD")
    meta = _get_m3ed_meta()
    for name, dirname in [("train", "train"), ("val", "test")]:
        image_dir = os.path.join(root, "images", dirname)
        # event_dir = os.path.join(root, "events", dirname)
        event_dir = os.path.join(root, event_dir_name, dirname)
        gt_dir = os.path.join(root, "annotations", dirname)
        name = f"m3ed_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=event_dir, z=gt_dir: load_event_sem_seg(z, y, x, gt_ext="png", event_ext='h5', image_ext="png")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            event_root=event_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,  # different from other datasets, dsec sets ignore_label to 255
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "data")
register_all_m3ed(_root)

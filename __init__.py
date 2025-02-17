"""
iNaturalist 2021 dataset.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import logging
import os
import shutil

import fiftyone as fo

import eta.core.web as etaw
import eta.core.utils as etau
from fiftyone.utils.coco import COCOObject, _coco_objects_to_detections
import eta.core.serial as etas

logger = logging.getLogger(__name__)


def download_and_prepare(dataset_dir, split):
    """Downloads the dataset and prepares it for loading into FiftyOne.

    Args:
        dataset_dir: the directory in which to construct the dataset
        split: a specific split to download.  The supported
            values are ``("train", "train-mini", "validation", "test")``

    Returns:
        a tuple of

        -   ``dataset_type``: a ``fiftyone.types.Dataset`` type that the
            dataset is stored in locally
        -   ``num_samples``: the total number of downloaded samples for the
            dataset or split
        -   ``classes``: a list of classes in the dataset
    """

    def _download_and_extract(split):
        split_dir = os.path.join(dataset_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        img_tar_name = _SPLIT_TO_FILE_NAME[split] + ".tar.gz"
        anno_tar_name = _SPLIT_TO_FILE_NAME[split] + ".json.tar.gz"
        _download_archive(
            os.path.join(_S3_DIR, img_tar_name),
            os.path.join(scratch_dir, img_tar_name),
        )
        _download_archive(
            os.path.join(_S3_DIR, anno_tar_name),
            os.path.join(scratch_dir, anno_tar_name),
        )

        logger.info(f"Extracting archive for {split}")
        etau.extract_archive(
            os.path.join(scratch_dir, img_tar_name),
            os.path.join(split_dir, "data"),
        )
        etau.extract_archive(
            os.path.join(scratch_dir, anno_tar_name), split_dir
        )
        os.rename(
            os.path.join(split_dir, _SPLIT_TO_FILE_NAME[split] + ".json"),
            os.path.join(split_dir, "labels.json"),
        )

    scratch_dir = os.path.join(dataset_dir, "tmp-download")
    os.makedirs(scratch_dir, exist_ok=True)

    if split not in _SPLIT_TO_FILE_NAME.keys():
        raise IOError(f"Input split {split} is not supported.")
    _download_and_extract(split)

    dataset_type = None
    split_data_dir = os.path.join(dataset_dir, split, "data")
    num_samples = len(etau.list_files(split_data_dir, recursive=True))
    classes = len(etau.list_subdirs(split_data_dir, recursive=True)) - 1

    shutil.rmtree(scratch_dir, ignore_errors=True)
    shutil.rmtree(os.path.join(dataset_dir, "__pycache__"), ignore_errors=True)

    return dataset_type, num_samples, classes


def load_dataset(dataset, dataset_dir, split):
    """Loads the dataset into the given FiftyOne dataset.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset` to which to import
        dataset_dir: the directory to which the dataset was downloaded
        split: a split to load. The supported values are
            ``("train", "train-mini", "validation", "test")``
    """
    split_dir = os.path.join(dataset_dir, split)
    dataset.add_dir(
        dataset_dir=split_dir,
        dataset_type=fo.types.COCODetectionDataset,
    )

    labels_json = os.path.join(split_dir, "labels.json")
    labels = etas.load_json(labels_json)
    _add_classification_labels(dataset, labels=labels["annotations"])


def _add_classification_labels(dataset, labels, categories):
    coco_objects = []
    for d in labels:
        # Add psuedo bbox for detection to classification conversion
        d["bbox"] = [0, 0, 0, 0]
        coco_obj = COCOObject.from_anno_dict(d)
        coco_objects.append([coco_obj])

    dataset.compute_metadata()
    widths, heights = dataset.values(["metadata.width", "metadata.height"])
    classes_map = {c["id"]: c["name"] for c in categories}

    classification_labels = []
    for _coco_objects, width, height in zip(coco_objects, widths, heights):
        frame_size = (width, height)
        _detections = _coco_objects_to_detections(
            _coco_objects, frame_size, classes_map, None, False, False
        )
        for det in _detections["detections"]:
            _classification = fo.Classification(
                label=det.label, confidence=det.confidence
            )
            classification_labels.append(_classification)
    dataset.set_values("ground_truth", classification_labels)


def _download_archive(url, archive_path, overwrite=False):
    if os.path.isfile(archive_path) and not overwrite:
        logger.info(f"Using existing archive {archive_path}")
    else:
        logger.info("Downloading dataset archive")
        etaw.download_file(url, path=archive_path)


_S3_DIR = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021"

_SPLIT_TO_FILE_NAME = {
    "train": "train",
    "train-mini": "train_mini",
    "validation": "val",
    "test": "public_test",
}

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
from fiftyone.utils.coco import COCOObject, parse_coco_categories
import eta.core.serial as etas
import fiftyone.core.storage as fos

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
    classes = etau.list_subdirs(split_data_dir, recursive=True)[1:]

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
    data_path = os.path.join(fos.normalize_path(split_dir), "data")

    labels_json = os.path.join(split_dir, "labels.json")
    labels = etas.load_json(labels_json)
    classes_map, images, annotations = _parse_coco_classification_annotations(
        labels
    )

    samples = []
    for image_id in images:
        image = images[image_id]
        ground_truth = None
        if annotations:
            coco_object = annotations[image_id]
            ground_truth = fo.Classification(
                label=classes_map[coco_object.category_id]
            )
        file_path = os.path.join(data_path, fos.normpath(image["file_name"]))
        samples.append(
            fo.Sample(
                filepath=file_path,
                ground_truth=ground_truth,
            )
        )
    dataset.add_samples(samples)


def _parse_coco_classification_annotations(d):
    categories = d.get("categories", None)
    # Load classes
    if categories is not None:
        classes_map, _ = parse_coco_categories(categories)
    else:
        classes_map = None

    # Load image metadata
    images = {i["id"]: i for i in d.get("images", [])}

    # Load annotations
    _annotations = d.get("annotations", None)
    if _annotations is not None:
        annotations = {}
        for a in _annotations:
            if a["image_id"] in annotations:
                raise Exception(
                    f"{a['image_id']} has more than one classification label."
                )
            annotations[a["image_id"]] = COCOObject.from_anno_dict(a)

        annotations = dict(annotations)
    else:
        annotations = None

    return classes_map, images, annotations


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

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

logger = logging.getLogger(__name__)


def download_and_prepare(dataset_dir, split=None):
    """Downloads the dataset and prepares it for loading into FiftyOne.

    Args:
        dataset_dir: the directory in which to construct the dataset
        split (None): a specific split to download, if the dataset supports
            splits.  The supported values are
            ``("train", "train-mini", "validation", "test")``

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

    if split:
        if split not in _SPLIT_TO_FILE_NAME.keys():
            raise IOError(f"Input split {split} is not supported.")
        _download_and_extract(split)
    else:
        for _split in _SPLIT_TO_FILE_NAME:
            _download_and_extract(_split)

    dataset_type = fo.types.COCODetectionDataset
    num_samples = 0
    classes = 0
    if split:
        split_data_dir = os.path.join(dataset_dir, split, "data")
        num_samples += len(etau.list_files(split_data_dir, recursive=True))
        classes = max(
            classes, len(etau.list_subdirs(split_data_dir, recursive=True)) - 1
        )
    else:
        for d in etau.list_subdirs(dataset_dir):
            split_data_dir = os.path.join(dataset_dir, d, "data")
            if os.path.exists(split_data_dir):
                num_samples += len(
                    etau.list_files(split_data_dir, recursive=True)
                )
            classes = max(
                classes,
                len(etau.list_subdirs(split_data_dir, recursive=True)) - 1,
            )

    shutil.rmtree(scratch_dir, ignore_errors=True)
    shutil.rmtree(os.path.join(dataset_dir, "__pycache__"), ignore_errors=True)

    return dataset_type, num_samples, classes


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

import os
import subprocess
import shutil

import fiftyone as fo

# import eta.core.web as etaw
import eta.core.utils as etau


def download_file(url, output, overwrite=False):
    # TODO(manushree): Debug why file size is larger when using etaw
    # etaw.download_file(url, path=output)
    if os.path.isfile(output) and not overwrite:
        return
    return subprocess.run(["curl", "-o", output, url])


def download_and_prepare(dataset_dir, split=None, **kwargs):
    """Downloads the dataset and prepares it for loading into FiftyOne.

    Args:
        dataset_dir: the directory in which to construct the dataset
        split (None): a specific split to download, if the dataset supports
            splits.  The supported values are
            ``("train", "train-mini", "validation", "test")``
        **kwargs: optional keyword arguments that your dataset can define to
            configure what/how the download is performed

    Returns:
        a tuple of

        -   ``dataset_type``: a ``fiftyone.types.Dataset`` type that the
            dataset is stored in locally, or None if the dataset provides
            its own ``load_dataset()`` method
        -   ``num_samples``: the total number of downloaded samples for the
            dataset or split
        -   ``classes``: a list of classes in the dataset, or None if not
            applicable
    """

    def _download_and_extract(split):
        split_dir = os.path.join(dataset_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        img_tar_name = split_to_file_name[split] + ".tar.gz"
        anno_tar_name = split_to_file_name[split] + ".json.tar.gz"
        download_file(
            os.path.join(s3_dir, img_tar_name),
            os.path.join(scratch_dir, img_tar_name),
        )
        download_file(
            os.path.join(s3_dir, anno_tar_name),
            os.path.join(scratch_dir, anno_tar_name),
        )

        etau.extract_archive(
            os.path.join(scratch_dir, img_tar_name),
            os.path.join(split_dir, "data"),
        )
        etau.extract_archive(os.path.join(scratch_dir, img_tar_name), split_dir)
        os.rename(
            os.path.join(split_dir, split_to_file_name[split] + ".json"),
            os.path.join(split_dir, "labels.json"),
        )

    # Download files and organize them in `dataset_dir`
    s3_dir = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021"
    split_to_file_name = {
        "train": "train",
        "train-mini": "train_mini",
        "validation": "val",
        "test": "public_test",
    }

    scratch_dir = os.path.join(dataset_dir, "tmp-download")
    os.makedirs(scratch_dir, exist_ok=True)

    if split:
        if split not in split_to_file_name.keys():
            raise IOError(f"Input split {split} is not supported.")
        _download_and_extract(split)
    else:
        for _split in split_to_file_name:
            _download_and_extract(_split)

    dataset_type = None
    num_samples = 0
    classes = 0
    if split:
        split_data_dir = os.path.join(dataset_dir, split, "data")
        num_samples += len(etau.list_files(split_data_dir, recursive=True))
        classes = max(classes, etau.list_subdirs(split_data_dir))
    else:
        for d in etau.list_subdirs(dataset_dir):
            split_data_dir = os.path.join(dataset_dir, d, "data")
            if os.path.exists(split_data_dir):
                num_samples += len(
                    etau.list_files(split_data_dir, recursive=True)
                )
            classes = max(
                classes, etau.list_subdirs(split_data_dir, recursive=True) - 1
            )

    shutil.rmtree(scratch_dir, ignore_errors=True)
    return dataset_type, num_samples, classes


def load_dataset(dataset, dataset_dir, split=None, **kwargs):
    """Loads the dataset into the given FiftyOne dataset.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset` to which to import
        dataset_dir: the directory to which the dataset was downloaded
        split (None): a split to load. The supported values are
            ``("train", "train-mini", "validation", "test")``
        **kwargs: optional keyword arguments that your dataset can define to
            configure what/how the load is performed
    """

    split_dir = os.path.join(dataset_dir, split)
    dataset.add_dir(
        split_dir,
        dataset_type=fo.types.COCODetectionDataset,
    )

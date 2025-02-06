# iNaturalist 2021

The iNaturalist 2021 dataset.

The dataset consists images of 10,000 species, with nearly 2.7M training
images, 100K validation images and 500K test images. A "mini"
training split containing 50 images per species, with a total of 500K
images is also available.

## Details

-   Dataset name: ``voxel51/inaturalist-2021``
-   Dataset source: https://github.com/visipedia/inat_comp/tree/master/2021
-   Tags: ``image, classification``
-   Supported split: ``train, train-mini, validation, test``

## Example usage

```py
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("https://github.com/voxel51/inaturalist-2021")

session = fo.launch_app(dataset)
```

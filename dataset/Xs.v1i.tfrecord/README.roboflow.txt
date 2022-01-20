
Xs - v1 2022-01-19 7:37pm
==============================

This dataset was exported via roboflow.ai on January 20, 2022 at 3:39 AM GMT

It includes 1820 images.
X are annotated in Tensorflow TFRecord (raccoon) format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Fit (white edges))

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip

The following transformations were applied to the bounding boxes of each image:
* Random brigthness adjustment of between -5 and +5 percent
* Random exposure adjustment of between -5 and +5 percent



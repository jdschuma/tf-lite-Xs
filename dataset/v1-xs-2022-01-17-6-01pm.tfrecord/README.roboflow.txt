
xs - v1 Xs-2022-01-17 6:01pm
==============================

This dataset was exported via roboflow.ai on January 18, 2022 at 2:03 AM GMT

It includes 325 images.
Xs are annotated in Tensorflow TFRecord (raccoon) format.

The following pre-processing was applied to each image:
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random shear of between -10° to +10° horizontally and -28° to +28° vertically

The following transformations were applied to the bounding boxes of each image:
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Random rotation of between -15 and +15 degrees
* Salt and pepper noise was applied to 5 percent of pixels



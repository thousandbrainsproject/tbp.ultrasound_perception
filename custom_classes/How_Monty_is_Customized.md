# How Monty is Customized
> 📘 General Infos
> For more information on how to customize Monty for a robotics application, see our tutorial [here](https://thousandbrainsproject.readme.io/docs/using-monty-for-robotics).

To use Monty for Ultrasound, we had to customize several classes:
-  `UltrasoundEnvironment`: Loads ultrasound image from folder and returns it as observation dictionary. In our experiments we usually use one of two subclasses of this environment:
   -  `ProbeTriggeredUltrasoundEnvironment`: Streams observations from live server to Monty. Whenever the experimenter presses the main button on the ultrasound probe, a new image and tracker locations are sent to Monty.
   -  `JSONDatasetUltrasoundEnvironment`: Loads step-wise observations (image + tracker data) from .json files. This environment can be used for offline testing (doesn't require an ultrasound setup and live data streaming).
-  `UltrasoundDataloader`: Receives the full ultrasound image and extracts a patch to send to the sensor module. Also cycles between ultrasound objects in an experiment.
- `UltrasoundSM`: A sensor module specialize on extracting pose and features from an ultrasound image and combining it with tracker information to get a CMP message.
- `UltrasoundMotorPolicy`: Combines the `InformedPolicy` with the `JumpToGoalStateMixin`. Generally actions are executed by the human experimenter but Monty is able to suggest goal states (locations to move to) to the experimenter.
- `MontyUltrasoundSupervisedObjectPretrainingExperiment`, and `UltrasoundExperiment`: Experiment classes that support supervised learning and inference experiments with ultrasound data.

![Custom classes shown with dashed boarders. The ](./figures/CustomClassesOverview.png#width=300px)

Following is some more information on the inner workings of some of those components. 

## Ultrasound Dataloader
The dataloader receives the full image from the ultrasound probe and extracts a patch from that image. The patch should be centered on the surface of the object in the center (along x direction) of the image. 

Retrieving this is trickier than one might think as the ultrasound image can contain artifacts and shaddows. We developed some basic heuristics that retrieved the correct patch on a small test set.

As shown in the image below, the `find_patch_with_highest_gradient` function of the dataloader first runs a patch (half the size of desired patch size for SM) down the center column of the image and calculates the gradient for each of the y locations. The gradient is defined as the horizontal edge response to a 3x3 sobel filter.

![](./figures/Dataloader.png#width=200px)

We then look at the gradient distribution and pick the first y location with a significant peak. We pick the first location since the maximum peak can sometimes correspond to a shaddow in the ultrasound image and not the actual object surface. Below are some example images of the gradient distributions and the extracted patches.

![](./figures/DataloaderExamples.png)

You can see that the first peak doesn't always correspond to the gradient maximum and that there are significant shaddows in the ultrasound image. Sometimes, there are also artifact above the surface of the object (not shown in examples), which is why we define a local threshold to make sure we don't pick up on a fainter edge that is high up in the image.

The dataloader then takes the pixel location of the patch and calculates the depth in meters based on the probe depth settings.

The observations returns from the dataloader contain the image patch, the depth of the patch on the image, and the probe tracker location and orientation.

## Ultrasound Sensor Module

The sensor module needs to turn the observations from the dataloader (patch image, depth and tracker pose) into a CMP compliant message (pose relative to world, features). 

To get the pose at the current time step, the sensor module needs to extract the point normal direction on the surface of the edge and combine it with the patch depth and tracker pose. For features we extract the amount of curvature of the surface.

To retrieve the point normal and curvature at the center of the patch we first need to define the edge. For this we use the `extract_edge_points` function which moves down from top to bottom and determines for each column which pixel is part of the edge. Due to the artifact and shaddow complications mentioned earlier we need to use an adaptive threshold for this and only look in a certain area above and below the edge point in the center of the patch. This gives us a set of points on the surface of the object.

We then fit a circle through the edge points which is required to pass through the center point. The size of the fitted circle defines the curvature. We then use this fitted circle to calculate the point normal at the center point. As edges can sometimes be ragged or contain artifacts we use an itterative approach that gradually discards edge points towards the left and right side of the image if the MSE of the circle fit is too high.

![](./figures/SMExamples.png)

Finally, the sensor module needs to calculate the pose of the patch relative to a common coordinate system (we call it rel. world). For this, it combines the HTCVive tracker location with the sensor offset (distance between tracker to tip of the ultrasound probe) and the measured depth of the patch on the image.

![Locations on potted_meat_can.](./figures/CoordinateTransforms.png#width=200px)

> ❗️ Caution
> These transforms still don't seem quite right but we haven't been able to fully debug them yet.

The combined pose information plus the detected curvature are then sent to the learning module (standard `EvidenceLM`) in the form of a CMP message.
# Utilities

This module provides a list of helpful tools that can help you during the development of your project.

### Interactive color filtering

This example implements an interactive tool for determining color filtering parameters. The image is first blurred using a gaussian kernel, and adjusted for gamma. The color filtering depends on a lower and upper limit of HSV values which are used to threshold the image in HSV color-space. The script will open a window providing controls for:

Blurring --> Gaussian size (Kernel) and standard deviation (Sigma)
Gamma correction --> Target value (Gamma) (16 --> 1.6)
Lower and Upper thresholds for HSV image masking within  [0-255]
- More information: Gamma correction

### Multi-threading

This example demonstrates multithreading in Python. Multithreading is a powerful tool that allows multiple tasks to run concurrently or in the background. This keeps the main Python thread free to perform important work, such as maintaining UI responsiveness.

- More information: Threading

### ROI selection

In computer vision, a region of interest (ROI) describes a subregion of an image in pixel coordinates. An ROI is commonly represented as a vector with 4 elements [x ,y, w, h] where:

[x, y] are the pixel coordinates of the region. This can be the center or one of the corners. 
[w, h] are the width and height of the region.
Selecting an ROI is a key part of later image processing stages as it provides information about the image content and focus to an algorithm and often reduces the process cost involved.

OpenCV offers an ROI selection implementation that works on static images. Our own simple implementation of an ROI selector that allows working on live images is also made available.

- More information: OpenCV Examples

### RealSense cameras

This module demonstrates how to access the RealSense camera streams and process the images. The RealSense camera has multiple sensors that give different views of the world (RGB, depth, infrared)

To get to know the RealSense cameras, you can check out the Intel RealSense SDK to play around with the sensors and try out some filtering and processing steps. Camera stream access may be achieved manually using the Intel API or using our RealSenseCamera.py class which simplifies setup. 

The RealSense cameras enable advanced image processing and merging of sensor maps. The PointcloudViewer.py  provides a demo of the cameras highlighting spatial & temporal filtering, decimation, colorization and lighting.

- More information:  Examples, Filtering

### Handling mouse events 

This example illustrates how to handle mouse events and use them to draw objects on a screen.



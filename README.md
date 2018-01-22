# machineVision
This Repository regroups the course projects I had during the Machine Vision Course of my MSc


## Image Segmentation
Implementation of a segmentation algorithm training mixtures of Gaussians to detect apples in an image.
<br/>main.m shows an exemple about how to create a dataset, train a model and then plot the ROCCurve.
<br/>fitMixGauss.m contains all the subroutines to fit a mixture of Gaussians to a data set.
<br/>Homework.m regroupe all the following task : Create training, test and validation sets, trains several mixtures of Gaussians, compare their prediction to choose the best model, plot the ROC Curve, then run the chosen model on unseen data.

## Homography and Tracking

### Homography
Implementation of a function to find the Homography between 4 pairs of points.
<br/> practical1.m shows how to determine the homography between 4 pairs of points
<br/> practical1B.m reconstruct a panorama photography from 3 photos.
<br/> practical2b.m incorporate a 3D-wired cube into a photography

### Tracking
Implementation of a function doing tracking using particle filter.

### Tracking&Homography
Combine both by first tracking the corner of a black square in a photo and then construct a 3D cube bases on the square.
<br/> HW2_Practical9c.m show how to track the corners
<br/> HW2_TrackingAndHomographies reconstruct the 3D-wired cube

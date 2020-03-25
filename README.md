# computer_vision_object_detection

This is a U-Shape solution for detection problem.

Objections have its  shape and it is challenging for predicting the center. However we can make center distinguishable by
converting single coordinate to a Gaussian Heatmap with pixel level intensities. Then we change the task domain from regression to segmentation. We can get segmentation map from last layer of the network and format the maximum intensity as 
the detection location.

Author: Dimitris Koumoutsakos
Date: July 18, 2025 


This file serves as notes for the "gaussian_robust.py' script. 

This script has one big function that performs gaussian detection of circles, meaning it finds the centers of a blob. This blob is a mask which is an input to the detect_gaussian function. The function has two distinct methods, one is contour based detection meaning it uses the contours of the masks and image moments (summary of important statistics of the blobs -> in this case the masks of the circles or ball. This can include position, area, diameter, etc). Using the image moments the method can compute a center of mass. The other method, gaussian based detection, starts out with all the pixels that have a value > 0, meaning they are part of the mask, and treats them as data points in 2d (x,y). It then calculates the average position and the covariance matrix, which tells you how spread out and corelated the points are. It then models the pixel distribution as a bell curve with that mean and covariance and then calculates the probability density for each pixel, telling it how likely this pixel belongs to the blob. It iteratively removes the low-probaility pixels (edges or noise), recalculating everything and repeating until the probabilities stabilize or you hit max amount of iterations. The mean of the remaining pixels is taken and accepted as the centroid.  

Use Contous Based Detection if the blob is not very noisy. If it is, gaussian based detection will work better. 

So in summary:

Contour + moments: Uses the shape boundary, calculates centroid directly from pixel moments inside the contour.

Gaussian method: Uses a statistical model of the pixel distribution inside the mask, iteratively filters out outliers, and computes centroid from the refined pixel cloud.
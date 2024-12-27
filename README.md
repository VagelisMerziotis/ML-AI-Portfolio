# ML-AI-Portfolio
Kaggle and other ML &amp; AI Projects 

In this repository I post all the projects centered to machine learning.

Currently there are the following projects (and their respective commented and functional jupyter notebooks):
1)Electronic eye 
  This project is focused on computer vision implementation with the pre-trained model YOLO11. The goal is to develop a pipeline that takes .mp4 files as inputs, passes them to YOLO11 to measure independent variables and uses the mean values of the dependant variable predicted by suitable ML models to determine the deadline needed in hours to perform a coloring job on the B&W version of the clips. 
  Initially, the dataset destined to train the models, either ML or ANN, was limited (~50 entries). After rigorous working on obtaining a bigger sample, a KMeans clustering method was performed on both the original set and the new set. A time column was assigned based on the cluster assigned to each record (same amount of clusters for both datasets).

# ML-AI-Portfolio
Kaggle and other ML &amp; AI Projects 

In this repository I post all the projects centered to machine learning.

Currently there are the following projects (and their respective commented and functional jupyter notebooks):
1) Electronic eye 
  This project is focused on computer vision implementation with the pre-trained model YOLO11. The goal is to develop a pipeline that takes .mp4 files as inputs, passes them to YOLO11 to measure independent variables and uses the mean values of the dependant variable predicted by suitable ML models to determine the deadline needed in hours to perform a coloring job on the B&W version of the clips. 
  Initially, the dataset destined to train the models, either ML or ANN, was limited (~50 entries). After rigorous working on obtaining a bigger sample, a KMeans clustering method was performed on both the original set and the new set. A time column was assigned based on the cluster assigned to each record (same amount of clusters for both datasets). In the end, after keeping the actual training_set, data augmentation with GenAI CTGAN was performed. Duplicate recorded were shaked to convert to noise and reintroduced in the dataset.
  The second part was the cross-validation of most ML models in scikit-learn (and a few more) that I currently know. Best cases were kept in .pkl files
  Lastly, an ANN developmental attemp was made. Hardware limitations prevented its full potential. Nevertheless, the procedure was more than  
educational for me.

2) Car Regression
   A Kaggle competitionw as help on used cars. The traning set consisted of about ~180k entries. Data cleaning and preperation was performed and subsequently ML models and ANN development was conducted.
   The most educating part was the PCA practice to handle outliers and eventually increase the performance of both ML and ANN models.

# ML-projects

[![Language](https://img.shields.io/badge/language-Python-blue.svg)](https://www.python.org/)

Machine learning projects and challenges to practice skills.

## [Brain Tumor Classification using Deep Learning](https://github.com/jennbushey/ML-projects/blob/main/Tumour%20Classification)

This repository contains the final project for ENEL 645 Winter 2024 at the University of Calgary. The project focuses on classifying brain tumors using deep learning techniques. The dataset used is the [Kaggle Brain tumors 256x256](https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256), which includes images labeled with four classes.

The project includes the implementation of three deep learning models:

-   VGG model: An 8-layer convolutional neural network created based on original research (research cited in the paper).
-   Transfer learning models: Utilizing ResNet50 and Xception models.
-   GradCam analysis: Visualizing the feature map to understand what each model focuses on for classification.

TensorFlow served as the primary platform for developing and training the models, which were executed on the University's GPU cluster using a custom conda environment and slurm batch files.

The final deliverables of the project include a paper detailing the research and findings, as well as the trained models. The best model was selected based on validation loss, and the models were compared using ROC curves to determine the best in terms of accuracy and false negatives.

## [Linear Models](https://github.com/jennbushey/ML-projects/blob/main/Linear%20Models.ipynb)

Using linear models to perform classification and regression tasks.

Data source:

-   Yellowbrick [spam](https://www.scikit-yb.org/en/latest/api/datasets/spam.html)
-   Yellowbrick [concrete](https://www.scikit-yb.org/en/latest/api/datasets/concrete.html)

## [Non-Linear Models](https://github.com/jennbushey/ML-projects/blob/main/Non-Linear%20Models.ipynb)

Using non-linear models to perform classification and regression tasks.

Data source:

-   Yellowbrick [concrete](https://www.scikit-yb.org/en/latest/api/datasets/concrete.html)
-   Aeberhard,Stefan and Forina,M.. (1991). Wine. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J.

## [Pipelines](https://github.com/jennbushey/ML-projects/blob/main/Pipelines.ipynb)

Learning how to complete appropriate preprocessing, test different supervised learning models and evaluate the results. Using pipeline and grid search to tune hyperparameters.

Data source:

-   Kaggle [Tree Survival Prediction](https://www.kaggle.com/datasets/yekenot/tree-survival-prediction)

## [Principal Component Analysis and Clustering](https://github.com/jennbushey/ML-projects/blob/main/PCA%20and%20Clustering.ipynb)

Learning how to use PCA and clustering techniques for supervised and unsupervised learning.

Data source:

-   Wheat kernels. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5NG8N.

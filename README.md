# Machine Learning Projects

**Table of Contents**

-   [Deep Learning](#deep-learning)
-   [Machine Learning](#machine-learning)

<br>

# Deep Learning

## [Brain Tumor Classification using Deep Learning](./Tumor%20Classification/)

This repository contains the final project for ENEL 645 Winter 2024 at the University of Calgary. The project focuses on classifying brain tumors using deep learning techniques. The dataset used is the [Kaggle Brain tumors 256x256](https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256), which includes images labeled with four classes.

The project includes the implementation of three deep learning models:

-   VGG model: An 8-layer convolutional neural network created based on original research (research cited in the paper).
-   Transfer learning models: Utilizing ResNet50 and Xception models.
-   GradCam analysis: Visualizing the feature map to understand what each model focuses on for classification.

TensorFlow served as the primary platform for developing and training the models, which were executed on the University's GPU cluster using a custom conda environment and slurm batch files.

The final deliverables of the project include a paper detailing the research and findings, as well as the trained models. The best model was selected based on validation loss, and the models were compared using ROC curves to determine the best in terms of accuracy and false negatives.

<br>

# Machine Learning

## [Netflix Movies and TV Shows Analysis](./Netflix.ipynb)

This project explores a comprehensive dataset of movies and TV shows available on Netflix, covering various aspects such as title type, director, cast, country of production, release year, rating, duration, genres, and description. The dataset contains over 8,000 entries and provides valuable insights into trends in Netflix content, genre popularity, and distribution across different regions and time periods.

### Key Features

-   Exploratory Data Analysis: Analyzing the distribution of releases, genre trends over time, and genre co-occurrence.
-   Genre Analysis: One-hot encoding genres and examining genre distribution and correlation.
-   Visualization: Visualizing genre trends, distribution, and correlation using matplotlib and seaborn.
-   Machine Learning: Using KMeans clustering and PCA to identify clusters of similar genres and visualize genre overlaps.

### Findings

International Movies and Dramatic Movies are consistently popular genres on Netflix.
There is no strong linear correlation between genres, indicating diverse viewer preferences.
Genre clusters reveal interesting patterns in genre co-occurrence, such as International Movies being likely to be categorized as Dramatic Movies or Comedy.

### Next Steps

Future analysis could include:

-   Genre Clustering: Grouping similar genres together using clustering algorithms.
-   Genre Prediction: Building a genre prediction model based on movie features to predict the genre of a title.

## [Time Series Forecasting with Yahoo Stock Price](./time-series-stocks.ipynb)

The objective of this project was to predict stock prices using historical stock market data and machine learning techniques, specifically using a Ridge Regression model.

The dataset consisted of daily stock market data, including features such as High, Low, Open, Volume, Adj Close, Close, and Date. Data exploration and preprocessing steps were performed to prepare the data for modeling.

Exploratory Data Analysis (EDA) was conducted to understand the distribution and relationships between variables. Feature engineering techniques were used to create lagged features, rolling statistics, and other relevant features to improve model performance. A Ridge Regression model was selected and trained on the preprocessed dataset. The model was evaluated using metrics such as R2 Score, Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE).

The Ridge Regression model showed promising results in predicting stock prices. Evaluation metrics indicated that the model performed well in predicting stock prices based on the selected features.

Future steps for this project include fine-tuning the model's hyperparameters, exploring other machine learning algorithms, and incorporating additional features or data sources to further improve prediction accuracy. Overall, this project demonstrates the feasibility of using machine learning for stock price prediction, with potential applications in real-time market analysis and decision-making.

## [Visualizing Disaster Tweets with Natural Language Processing](./NLP-DisasterTweets.ipynb)

This project utilizes pandas for data manipulation and analysis, matplotlib and seaborn for visualization, and nltk, spacy, and wordcloud for natural language processing. The goal is to visualize the most popular words and locations tweeted during a disaster.

## [Linear Models](./Linear%20Models.ipynb)

Using linear models to perform classification and regression tasks.

Data source:

-   Yellowbrick [spam](https://www.scikit-yb.org/en/latest/api/datasets/spam.html)
-   Yellowbrick [concrete](https://www.scikit-yb.org/en/latest/api/datasets/concrete.html)

## [Non-Linear Models](./Non-Linear%20Models.ipynb)

Using non-linear models to perform classification and regression tasks.

Data source:

-   Yellowbrick [concrete](https://www.scikit-yb.org/en/latest/api/datasets/concrete.html)
-   Aeberhard,Stefan and Forina,M.. (1991). Wine. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J.

## [Pipelines](./Pipelines.ipynb)

Learning how to complete appropriate preprocessing, test different supervised learning models and evaluate the results. Using pipeline and grid search to tune hyperparameters.

Data source:

-   Kaggle [Tree Survival Prediction](https://www.kaggle.com/datasets/yekenot/tree-survival-prediction)

## [Principal Component Analysis and Clustering](./PCA%20and%20Clustering.ipynb)

Learning how to use PCA and clustering techniques for supervised and unsupervised learning.

Data source:

-   Wheat kernels. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5NG8N.

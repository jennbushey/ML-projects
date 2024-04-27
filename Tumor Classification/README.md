This project focuses on classifying brain tumors using deep learning techniques. The dataset used is the Kaggle Brain tumors 256x256, which includes images labeled with four classes.

The project includes the implementation of three deep learning models:

VGG model: An 8-layer convolutional neural network created based on original research (research cited in the paper).
Transfer learning models: Utilizing ResNet50 and Xception models.
GradCam analysis: Visualizing the feature map to understand what each model focuses on for classification.
TensorFlow served as the primary platform for developing and training the models, which were executed on the University's GPU cluster using a custom conda environment and slurm batch files.

The final deliverables of the project include a paper detailing the research and findings, as well as the trained models. The best model was selected based on validation loss, and the models were compared using ROC curves to determine the best in terms of accuracy and false negatives.

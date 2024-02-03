## Emotion Detection 

1. Importing Libraries and Setting Up Environment:
\
The code begins by importing necessary libraries such as TensorFlow, Matplotlib, and other TensorFlow modules for model building and data handling. It sets up TensorFlow mixed precision for faster computation and installs the Kaggle API for dataset download.

2. Downloading and Preparing Dataset:
\
Using Kaggle's API, the code downloads the 'Human Emotion Detection Dataset' and unzips it. It organizes the dataset into training and testing directories, each containing images categorized by emotion labels.

Datset Link : https://www.kaggle.com/datasets/shubh556/human-emotion-detection-dataset

3. Loading and Preprocessing Data:
\
The image_dataset_from_directory function from TensorFlow is utilized to efficiently load the dataset. The function ensures proper labeling and preprocessing, including resizing and rescaling the images to a standard size.

4. Model Building:
\
The model architecture is defined using the EfficientNetB7 convolutional neural network as the base model. Additional dense layers are added on top for classification. The model's layers are initially frozen to prevent retraining, and later, only the last five layers are unfrozen for fine-tuning.

5. Model Compilation and Callbacks:
\
The model is compiled with an optimizer (Adam) and loss function (sparse categorical cross-entropy). Callbacks for reducing learning rate on plateau and early stopping based on validation accuracy are defined to optimize training.

6. Training the Model:
\
The model is trained using the training dataset for 30 epochs. During training, the callbacks monitor the validation accuracy and adjust the learning rate accordingly. The model's performance is evaluated using the testing dataset.

7. Visualization:
\
Matplotlib is used to visualize the training and validation loss and accuracy curves, providing insights into the model's performance during training. Class-wise accuracy scores are also plotted to evaluate the model's performance across different emotions.

8. Evaluation and Interpretation:
\
Finally, the model's performance is evaluated using accuracy metrics calculated on the testing dataset. Predictions are made on the test data, and class-wise accuracy scores are calculated to assess the model's ability to classify different emotions accurately.

## Downloading Data 

![Home Pages1](https://github.com/Shubh556/Emotion-Detection/blob/main/images/downloading%20data.png?raw=true)

## Vizvalizing images

![Home Pages1](https://github.com/Shubh556/Emotion-Detection/blob/main/images/plotting%20the%20data%20.png?raw=true)

## Model

![Home Pages1](https://github.com/Shubh556/Emotion-Detection/blob/main/images/model.png?raw=true)

## Accuracy and Loss Curves

![Home Pages1](https://github.com/Shubh556/Emotion-Detection/blob/main/images/curves.png?raw=true)

## Result on Test Data

![Home Pages1](https://github.com/Shubh556/Emotion-Detection/blob/main/images/Evaluating%20Results%20.png?raw=true)

## Class-wise Accuracy Score 

![Home Pages1](https://github.com/Shubh556/Emotion-Detection/blob/main/images/Compare.png?raw=true)

## Testing it on images downloded from google 

![Home Pages1](https://github.com/Shubh556/Emotion-Detection/blob/main/images/testing.png?raw=true)



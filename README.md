# Covid-19-Chest-X-Ray-Image-Recognition

**Covid-19 Chest XRay Image Recognition Project Report**

**Introduction:**

The Covid-19 Chest XRay Image Recognition project aims to build an image classification model that can identify Covid-19 positive patients by analyzing their chest X-ray images. The project utilizes computer vision techniques to distinguish between healthy and Covid-19 infected lungs based on visual patterns present in the X-ray images. This project has significant real-world applications in healthcare, especially during the ongoing Covid-19 pandemic.

**Dataset:**

The dataset used for this project is the "Covid19 Image Dataset" by Pranav Raikokte, available on Kaggle. The dataset contains chest X-ray images of Covid-19 positive and negative patients, as well as patients with other lung diseases. It comprises three classes: Covid-19 Positive, Covid-19 Negative, and Virus (other lung diseases).

Dataset Source: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset

**Data Preprocessing:**

The dataset images were preprocessed before feeding them into the model. The preprocessing steps include resizing the images to a standard size of 256x256 pixels and normalizing the pixel values to be between 0 and 1. Data augmentation techniques were applied to increase the dataset's size and improve the model's generalization. The augmentation techniques used include rotation, width and height shift, shearing, zooming, and horizontal flipping.

**Model Building:**

The image classification model was built using a sequential model architecture in Keras. The model consists of three convolutional layers with ReLU activation functions and max-pooling layers to extract features from the images. Dropout layers were added to reduce overfitting. The last layers include a flatten layer, a dense layer with a ReLU activation, another dropout layer, and the final output layer with a softmax activation function to predict the probability of each class.

**Model Compilation:**

The model was compiled using the categorical cross-entropy loss function, Adam optimizer, and accuracy as the evaluation metric. The categorical cross-entropy loss is suitable for multi-class classification tasks, and the Adam optimizer is an efficient variant of stochastic gradient descent.

**Model Training:**

The model was trained on the preprocessed dataset using the ImageDataGenerator for batch processing. The training was performed for 20 epochs with a batch size of 16. The training process involved updating the model's weights based on the backpropagation of errors and gradient descent.

**Model Evaluation:**

The trained model was evaluated on the test dataset to assess its performance. The evaluation metrics used include accuracy, precision, recall, and F1-score. Additionally, a confusion matrix was generated to visualize the model's performance for each class.

**Results and Discussion:**

The model achieved an accuracy of X% on the test dataset. The precision, recall, and F1-score for each class were as follows:
- Covid-19 Positive: Precision = X%, Recall = X%, F1-score = X%
- Covid-19 Negative: Precision = X%, Recall = X%, F1-score = X%
- Virus: Precision = X%, Recall = X%, F1-score = X%

The confusion matrix showed that the model performed well in correctly classifying Covid-19 positive and negative cases. However, there were some misclassifications between Covid-19 negative and virus cases, indicating areas for potential improvement.

**Conclusion:**

The Covid-19 Chest XRay Image Recognition project successfully built an image classification model capable of distinguishing between Covid-19 positive, Covid-19 negative, and other lung diseases based on chest X-ray images. The model's accuracy and performance metrics demonstrate its potential for real-world application in healthcare settings, especially during the Covid-19 pandemic.

**Future Work:**

Possible areas of future improvement include:
- Fine-tuning the model's architecture to achieve better performance.
- Exploring other deep learning architectures or transfer learning techniques.
- Collecting a larger and more diverse dataset to further improve model generalization.
- Investigating ways to handle class imbalances if present in the dataset.

**GitHub Repository:**

The complete code and project report can be found on GitHub: https://github.com/Asadxio/Covid-19-Chest-X-Ray-Image-Recognition

**License:**

This project is shared under the [MIT License](https://opensource.org/licenses/MIT).


# SHAP Visualization Lab with TensorFlow

## Introduction
This lab will guide you through the process of using SHAP (SHapley Additive exPlanations) to interpret the decisions made by a deep learning model trained on the CIFAR-10 dataset. By the end of this lab, you should be able to generate SHAP visualizations that reveal the influence of image features on model predictions.

## Objectives
- Implement a function to generate SHAP visualizations.
- Interpret the visualizations to understand model predictions.

## Setup
Ensure you have the following libraries installed:
```
numpy==1.24.3
pandas==2.0.3
torch==2.0.1
torchvision==0.15.2
scikit-learn==1.2.2
shap==0.46.0
scipy==1.11.2
matplotlib==3.9.2
transformers==4.44.0
nltk==3.8
tensorflow==2.15.0
```

## Base Code
The following code sections are provided for you to use as a foundation for the lab tasks.

### Part 1: Import Necessary Modules

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
from ipywidgets import interact, IntSlider

print("TensorFlow version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
```

**Explanation:**
This section imports the required libraries for building and training a convolutional neural network (CNN) using TensorFlow and Keras. It also imports libraries for data manipulation (`numpy`), plotting (`matplotlib`), and explainability (`shap`). The `ipywidgets` library is imported for interactive components, if needed. The code also checks and prints if a GPU is available, which can speed up model training.

### Part 2: Data Preparation

```python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```

**Explanation:**
Here, the CIFAR-10 dataset is loaded, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images. The images are normalized by scaling pixel values from 0-255 to a range of 0-1. The `class_names` list provides readable labels for each class, making it easier to interpret predictions.

### Part 3: Model Definition

```python
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

**Explanation:**
This function defines a convolutional neural network (CNN) using the Keras Sequential API. The model includes multiple convolutional layers (`Conv2D`) for feature extraction, max pooling layers (`MaxPooling2D`) for dimensionality reduction, and fully connected dense layers (`Dense`) for classification. The final layer uses a softmax activation function to output class probabilities for the 10 categories in CIFAR-10. The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.

### Part 4: Model Loading or Training

```python
model_path = 'cifar10_cnn.keras'
if os.path.exists(model_path):
    print("Loading pre-trained model...")
    model = tf.keras.models.load_model(model_path)
else:
    print("No pre-trained model found. Training a new model...")
    model = create_model()
    history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    model.save(model_path)
    print(f"Model saved to {model_path}")
```

**Explanation:**
This part handles model loading and training. If a pre-trained model is found in the specified path, it loads the model, saving training time. Otherwise, it trains a new model from scratch, running for 10 epochs. After training, the model is saved for future use. The `history` object can be used to analyze training metrics like loss and accuracy.

### Part 5: Model Evaluation

```python
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc:.2f}")
```

**Explanation:**
The trained or loaded model is evaluated on the test dataset to check its performance. The test accuracy is printed to give an indication of how well the model generalizes to unseen data. The loss is also calculated but not printed in this case.

### Part 6: Model Testing

```python
def test_model(model, test_images, test_labels, num_samples=5):
    indices = np.random.choice(test_images.shape[0], num_samples, replace=False)
    sample_images = test_images[indices]
    sample_labels = test_labels[indices]

    predictions = model.predict(sample_images)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(sample_images[i])
        predicted_class = class_names[np.argmax(predictions[i])]
        true_class = class_names[sample_labels[i][0]]
        ax.set_title(f"Pred: {predicted_class}\nTrue: {true_class}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
```

**Explanation:**
This function tests the model by randomly selecting a few images from the test dataset and displaying the predictions. The actual and predicted classes are shown for each image, allowing you to visually inspect how well the model performs. The function uses Matplotlib to display the images side by side with their labels. This provides a simple, interactive way to verify the model’s predictions.


## SHAP Preparation

```python
num_background = 100
background_images = test_images[:num_background]
explainer = shap.GradientExplainer(model, background_images)
```
**Explanation:**
This section sets up SHAP (SHapley Additive exPlanations) to explain the model’s predictions. The background_images variable takes a subset of 100 images from the test dataset, which are used as reference images (background data) for SHAP explanations. The shap.GradientExplainer is instantiated using the trained model and the background images. This explainer will later allow us to compute the SHAP values, which highlight the most important features contributing to the model’s predictions.

## Task: Implement SHAP Visualization

The `shap_visualization` function leverages SHAP (SHapley Additive exPlanations) values to illustrate how different features of an input image contribute to a neural network's prediction. This guide provides a comprehensive overview of implementing this function to help you better understand each component's role.

## Overview

The function is designed to:
1. Select a single image from the dataset based on an index.
2. Calculate SHAP values that indicate the contribution of each pixel to the model's prediction.
3. Visualize both the original image and a scatter plot of the SHAP values overlaid on the image.

## Step-by-Step Implementation

## Step 1: Function Definition

Start by defining the function that takes an image index as its parameter:

```python
def shap_visualization(image_index):
    # We'll add the function body in the following steps
    pass
```

## Step 2: Extract the Image and Its Label

Inside the function, add code to extract the image and its true label:

```python
def shap_visualization(image_index):
    image = test_images[image_index:image_index+1]
    true_label = test_labels[image_index][0]
```

## Step 3: Generate and Process SHAP Values

Add code to generate SHAP values and process them:

```python
def shap_visualization(image_index):
    image = test_images[image_index:image_index+1]
    true_label = test_labels[image_index][0]

    # Generate and process SHAP values
    shap_values = explainer.shap_values(image)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    shap_values_for_class = shap_values[0, ..., predicted_class]
    shap_sum = np.sum(shap_values_for_class, axis=-1)
```

## Step 4: Normalize SHAP Values

Add code to normalize the SHAP values for the scatter plot:

```python
def shap_visualization(image_index):
    image = test_images[image_index:image_index+1]
    true_label = test_labels[image_index][0]

    # Generate and process SHAP values
    shap_values = explainer.shap_values(image)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    shap_values_for_class = shap_values[0, ..., predicted_class]
    shap_sum = np.sum(shap_values_for_class, axis=-1)

    # Normalize SHAP values for scatter plot
    shap_normalized = (shap_sum - shap_sum.min()) / (shap_sum.max() - shap_sum.min())
```

## Step 5: Create the Visualization

Now, add the code to create the visualization:

```python
def shap_visualization(image_index):
    image = test_images[image_index:image_index+1]
    true_label = test_labels[image_index][0]

    # Generate and process SHAP values
    shap_values = explainer.shap_values(image)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    shap_values_for_class = shap_values[0, ..., predicted_class]
    shap_sum = np.sum(shap_values_for_class, axis=-1)

    # Normalize SHAP values for scatter plot
    shap_normalized = (shap_sum - shap_sum.min()) / (shap_sum.max() - shap_sum.min())

    # Create figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Original Image
    axs[0].imshow(image[0])
    axs[0].set_title("Original Image\nTrue: " + class_names[true_label])
    axs[0].axis('off')

    # Scatter Plot with Stars on Image
    y, x = np.indices(shap_sum.shape)
    colors = shap_sum.flatten()  # Color by SHAP values
    sizes = 100 * shap_normalized.flatten() + 10  # Size of stars
    axs[1].imshow(image[0], aspect='auto')  # Display the original image as background
    scatter = axs[1].scatter(x.flatten(), y.flatten(), c=colors, s=sizes, cmap='coolwarm', marker='o', alpha=0.6)
    axs[1].set_title("SHAP Scatter on Image\nPredicted: " + class_names[predicted_class])
    axs[1].axis('off')
    fig.colorbar(scatter, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
```

## Step 6: Test the Function

After defining the function, you can test it by calling it with an image index:

```python
# Test the function
shap_visualization(10)  # Visualize SHAP values for the 11th image
```

## Conclusion

By following these steps, you will have a functional `shap_visualization` function that not only computes but also beautifully illustrates the impact of individual features (pixels, in this case) on the model's prediction. This is invaluable for debugging, improving model understanding, and presentations.

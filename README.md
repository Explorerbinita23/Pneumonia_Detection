# Pneumonia_Detection
Pneumonia Detection Using CNN

📌 Project Overview

This project focuses on developing a Convolutional Neural Network (CNN) model to detect pneumonia from chest X-ray images. The goal is to build a reliable medical image classification model that can distinguish between normal and pneumonia cases with high accuracy and robust performance.

🚀 Problem Statement

Early and accurate detection of pneumonia is crucial in healthcare, as delayed diagnosis can lead to severe complications. This project aims to classify chest X-ray images into two categories:

Normal

Pneumonia

The challenge lies in dealing with imbalanced data—there are almost three times more pneumonia cases than normal ones, which is a typical scenario in medical datasets.

📊 Dataset

Source: Chest X-ray dataset from Kaggle

Classes: Normal and Pneumonia

Data preprocessing:

Rescaling pixel values to [0, 1]

Augmentation techniques applied to mitigate class imbalance

🔍 Data Augmentation

Data augmentation was crucial for balancing the imbalanced dataset by generating new samples for the minority class (normal cases). The following augmentations were applied:

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

Rotation: Randomly rotate images by 15 degrees

Shifts: Horizontal and vertical shifts

Zoom: Random zoom up to 20%

Horizontal flip: To enhance variety in training samples

🏗️ Model Architecture

We built a hybrid approach by combining transfer learning with custom layers:

Base model: VGG19 (pre-trained on ImageNet)

Fine-tuning: Unfrozen the last 5 layers of VGG19 to allow learning task-specific features

Custom layers:

Global Average Pooling (reduces overfitting by minimizing the number of parameters)

Dense layers with Batch Normalization and Dropout

for layer in base_model.layers[-5:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

📈 Training and Evaluation

Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy

Callbacks:

Early stopping: Stop training if validation loss stops improving

Learning rate scheduler: Gradually reduce learning rate for fine-tuning

Model checkpoint: Save the best model

✅ Results

Accuracy: 92%

Recall: 94% (ensuring minimal false negatives — crucial for medical AI)

Precision: 93% (reducing false positives)
Note: While the model's performance is promising, it's not yet optimal. There’s still room for improvement, and we plan to further enhance its accuracy and robustness.

🔥 Future Enhancements

Dataset Expansion: Include more diverse X-ray images to improve model generalization.

Advanced Models: Experiment with ResNet, EfficientNet, or Inception for better performance.

Explainability: Integrate Grad-CAM to visualize which areas of the X-ray influenced the model’s decisions.

Medical Validation: Collaborate with healthcare professionals for validation and feedback.

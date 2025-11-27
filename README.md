# EXP 04: Implement a Transfer Learning concept in Image Classification
## AIM:
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.
DESIGN STEPS:
1.Load the dataset and resize all images to 224×224.

2.Import the pre-trained VGG19 model with ImageNet weights.

3.Freeze the VGG19 layers so they are not trained again.

4.Add new Dense layers on top for classifying the given dataset.

5.Compile the model using Adam optimizer and categorical cross-entropy loss.

6.Train the model using training and validation data.

7.Evaluate the model on test images and display final accuracy.

PROGRAM:
# Step 1: Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models

# Step 2: Load and Preprocess Dataset (CIFAR-10)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to range [0,1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Step 3: Load Pre-trained Model (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # Freeze the base layers

# Step 4: Add Custom Classification Layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')  # 10 classes in CIFAR-10
])

# Step 5: Compile the Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 6: Train the Model
history = model.fit(
    train_images, train_labels,
    epochs=5,
    validation_data=(test_images, test_labels)
)

# Step 7: Evaluate the Model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Step 8: Save the Model
model.save("transfer_learning_model.h5")
print("\nModel saved successfully!")
output:
<img width="848" height="218" alt="image" src="https://github.com/user-attachments/assets/88440fde-027f-4dbb-a504-ed1455fa18a4" />

result:
Thus,the program was implemented and executed successfully.




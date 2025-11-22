# **MACHINE LEARNING WITH COMPUTER VISION ASSESSMENT**

Create a handwriting recognition system which is capable of distinguishing specific handwriting numbers (0-9) from a group of given images and videos captured by scanned handwriting from a given test dataset

I will first build and train a convolutional neural network model using Mnist dataset in tensorflow

# **Building AI Model**

###  **Importing Libraries**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
```

```python
## mounting drive
from google.colab import drive
drive.mount('/content/drive')
```

### **Load MNIST dataset**

```python
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

### **Explore the data**

```python
get_img = X_train[0]
print(f"The name of the image number is: {y_train[0]}")
plt.axis('off')
plt.imshow(get_img, cmap='gray')
```

```python
# showing more numbers

plt.figure(figsize=(9,9))

for i in range(9):
    img = X_train[i+18].reshape((28,28)) / 255
    ax = plt.subplot(3, 3, i+1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


    ax.set_title(f'{y_train[i+18]}')
    plt.imshow(img, cmap='gray')

plt.tight_layo
```

### **Data Preprocessing**

```python
# Normalize images the image to 0,1 range
X_train, X_test = X_train / 255.0, X_test / 255.0
```

```python
#Reshaping of data
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
print(f"X_train shape is: {X_train.shape}")
print(f"X_test shape is: {X_test.shape}")
```

```python
# Convert labels to categorical (one-hot encoding)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
```

### **Building the Model**

```python
# Build an exceptional CNN model
def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        la
```

```python
# Compile the model
model = build_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

```python
# Train the model with early stopping and learning rate reduction
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

history = model.fit(X_train,
```

```python
## Saving the model
model.save("/content/drive/MyDrive/Assessment/Computer vision/models/Advance_model.h5")

```

```python
model.save('/content/drive/MyDrive/Assessment/Computer vision/models/Advance_model.keras')
```

```python
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

### **Testing Prediction**

```python
def predict_images(image_paths):
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = img.reshape(1, 28, 28, 1) / 255.0
        predict
```

# Summary of Final Prediction1.ipynb

# **Predicting and Testing of Numbers**

## **Importing Libraries**

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
```

```python
# mount the google drive
from google.colab import drive
drive.mount('/content/drive')
```

### **Loading my trained Model**

```python
model = load_model("/content/drive/MyDrive/Assessment/Computer vision/models/Advance_model.keras")
```

### **Loading my testing data**



```python
image1 = "/content/drive/MyDrive/Assessment/Computer vision/Number_Test_Data/000.png"
image2 = "/content/drive/MyDrive/Assessment/Computer vision/Number_Test_Data/001.png"
image3 = "/content/drive/MyDrive/Assessment/Computer vision/Number_Test_Data/002.png"
image4 = "/content/drive/MyDrive/Assessmen
```

# **Predicting the first 10 images**

```python
def predict_images(image_paths):
    num_images = len(image_paths)
    num_cols = 5
    num_rows = (num_images + num_cols - 1)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, img_path in enumerate(image_paths):
        img = cv2.imr
```

### **Functions for numbers segmentation and feature extraction**

```python
def predict_digits_from_image(image_path):
    try:
        #Load the image to predict
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")

        #Segment the image into individual numbers
 
```

### **Other Image Predictions**

```python
predict_digits_from_image(image11)
```

```python
predict_digits_from_image(image12)
```

```python
predict_digits_from_image(image13)
```

```python
predict_digits_from_image(image14)
```

```python
predict_digits_from_image(image15)
```

### **Working on the video**

```python
from google.colab.patches import cv2_imshow
# Open the video file
video_path = "/content/drive/MyDrive/Assessment/Computer vision/Number_Test_Data/015.avi"
cap = cv2.VideoCapture(video_path)

frame_count = 0
predictions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        
```


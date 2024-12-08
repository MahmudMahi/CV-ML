# Import necessary libraries
import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Step 1: Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Reduce the dataset to 100 samples for both train and test
X_train = X_train[:100]
y_train = y_train[:100].ravel()
X_test = X_test[:100]
y_test = y_test[:100].ravel()

# Step 2: Normalize the images by scaling pixel values to [0, 1]
X_train_normalized = X_train / 255.0
X_test_normalized = X_test / 255.0

# Step 3: Flatten the images to 1D vectors (CIFAR-10 images are 32x32x3, so each image becomes a vector of size 3072)
X_train_flattened = X_train_normalized.reshape(-1, 32*32*3)
X_test_flattened = X_test_normalized.reshape(-1, 32*32*3)

# Step 4: Initialize and train the K-Nearest Neighbors (K-NN) classifier
n_neighbors=5
knn = KNeighborsClassifier(n_neighbors) 
knn.fit(X_train_flattened, y_train) #fit means train

# Step 5: Make predictions on the test set
y_pred = knn.predict(X_test_flattened)

# Step 6: Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of K-NN on CIFAR-10 dataset with 100 samples: {accuracy * 100:.2f}%")

#Input an image from a file
def predict_image(image_path):
    # Load the image file, resizing it to 32x32 (the size of CIFAR-10 images)
    img = image.load_img(image_path, target_size=(32, 32))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Normalize the image
    img_array = img_array / 255.0

    # Flatten the image
    img_array = img_array.reshape(1, -1)

    # Make a prediction using the trained model
    prediction = knn.predict(img_array)
    
    # Show the image and prediction
    plt.imshow(img)
    plt.title(f"Predicted: {prediction}")
    plt.axis('off')
    plt.show()

# Example: Pass the path of your image here
image_path = 'C:\\Users\\mahmu\\Downloads\\frog.jpeg'  # Change this to your image path
predict_image(image_path)


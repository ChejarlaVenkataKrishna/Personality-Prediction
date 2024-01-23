import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Function to extract features from an image
def extract_features(image_path):
    # Use OpenCV to read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Perform feature extraction (e.g., histogram of oriented gradients)
    # Replace this with the actual feature extraction method you want to use
    
    # For demonstration, let's use the pixel values as features
    features = image.flatten()
    
    return features

# Load image paths and corresponding labels
# Modify this part according to your dataset structure
image_paths = ['Anayya.jpg', 'My photo.jpg', ...]
labels = ['personality_type1', 'personality_type2', ...]

# Extract features for each image
X = [extract_features(image_path) for image_path in image_paths]
y = labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with PCA and SVM
clf = make_pipeline(StandardScaler(), PCA(n_components=100), SVC())

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



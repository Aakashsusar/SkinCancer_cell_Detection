import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import os

from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import layers
from functools import partial

AUTO = tf.data.AUTOTUNE  # Updated AUTO, optimize the performance of data pipelines when working with TensorFlow's

import warnings
warnings.filterwarnings('ignore')

# Load dataset
# Retrieve all image file paths from the dataset directory
image_paths = glob(r'C:\Users\Aakash\Downloads\Cancer cell\Cancer_cells\cancer_cells\*\*.jpg')
# Print the total number of images found
print(f"Total images found: {len(image_paths)}")

# Ensure paths are correctly formatted
image_paths = [path.replace('\\', '/') for path in image_paths]

# Create DataFrame
df = pd.DataFrame({'filepath': image_paths})

# Extract label from the directory name
df['label'] = df['filepath'].apply(lambda x: os.path.basename(os.path.dirname(x)))

# Convert labels to binary (0: benign, 1: malignant)
df['label_bin'] = df['label'].map({'benign': 0, 'malignant': 1})

# Display first few rows
print(df.head())

# Image preprocessing function
from PIL import Image, ImageFilter
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    # Apply noise reduction (Gaussian Blur)
    image = image.filter(ImageFilter.GaussianBlur(radius=1))  # Adjust radius as needed
    image = image.resize((128, 128))  # Resize image
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return image_array

# Visualizing before and after preprocessing
def visualize_preprocessing(image_path):
    original = Image.open(image_path) # Load the original image
    processed = preprocess_image(image_path)  # Apply preprocessing function

    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) #Creates a 1-row, 2-column figure layout to display two images side by side.

    axes[0].imshow(original) ## Display original image

    axes[0].set_title("Original Image") # Set title
    axes[0].axis("off") # Remove axis for better visualization

    axes[1].imshow(processed) # Display preprocessed image
    axes[1].set_title("Preprocessed Image")
    axes[1].axis("off") # Remove axis

    plt.show() # Display the figure

    #visualization
sample_image_path = df['filepath'].iloc[35] # Select the image from location (iloc[]) from the dataset
visualize_preprocessing(sample_image_path) # Call the function to visualize the original and preprocessed image

x = df['label'].value_counts() # Count the number of images in each class (benign vs malignant)

# Create a pie chart to visualize class distribution
plt.pie(x.values,
        labels=x.index,
        autopct='%1.1f%%')
plt.show() # Show the pie chart

for cell in df['label'].unique():
  # Loop through each unique class label in the dataset (benign and malignant)
    temp = df[df['label'] == cell]
    # Filter the DataFrame to get all images belonging to the current class

    index_list = temp.index
    # Get the index positions of the images in the current class
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
     # Create a figure with 4 subplots to display 4 images
    fig.suptitle(f'Images for {cell} category . . . .', fontsize=20)
    # Set a title for the figure indicating the current class being visualized
    for i in range(4):
       # Loop to select and display 4 random images from the current class
        index = np.random.randint(0, len(index_list))
        # Generate a random index within the range of available images in this class
        index = index_list[index]
        data = df.iloc[index]

        image_path = data[0]  # Get the image file path from the DataFrame

        img = np.array(Image.open(image_path))
         # Open the image and convert it into a NumPy array for visualization
        ax[i].imshow(img)
        # Display the image in the corresponding subplot
plt.tight_layout()
# This function is used to adjust layout to prevent overlapping
plt.show()

def decode_image(filepath, label=None):

    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0

    # Compare label with integer 0 or 1 instead of string
    if label == 0:
        Label = 0
    else:
        Label = 1

    return img, Label
# Split the DataFrame into training and validation sets
from sklearn.model_selection import train_test_split

X = df['filepath']  # Image file paths
Y = df['label_bin']  # Corresponding labels

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y  # Stratify for balanced split
)

train_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_train, Y_train))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO)
)

val_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_val, Y_val))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO)
)

from tensorflow.keras.applications.efficientnet import EfficientNetB7

pre_trained_model = EfficientNetB7(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False
)

for layer in pre_trained_model.layers:
    layer.trainable = False

    from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Step 1: Convert all images into flattened arrays
X = []
y = []

for i in tqdm(df.index):
    image_array = preprocess_image(df.loc[i, 'filepath'])  # shape: (128, 128, 3)
    X.append(image_array.flatten())  # Flatten to 1D
    y.append(df.loc[i, 'label_bin'])

X = np.array(X)
y = np.array(y)

# Step 2: Split data
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

# Step 4: Evaluate
y_pred_rf = rf_model.predict(X_test_rf)
print("Random Forest Accuracy:", accuracy_score(y_test_rf, y_pred_rf))
print("Classification Report:\n", classification_report(y_test_rf, y_pred_rf))

#importing necessar libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Split dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_bin'])
#Training set (train_df): Used to train the CNN.
#Validation set (val_df): Used to evaluate the CNN's performance during training.
# Image Data Generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='filepath', y_col='label_bin', target_size=(150, 150),
    batch_size=32, class_mode='raw')

val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col='filepath', y_col='label_bin', target_size=(150, 150),
    batch_size=32, class_mode='raw')

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# Function to classify new images
def classify_image(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Malignant" if prediction > 0.5 else "Benign"

# Save the trained model to the current folder
model.save('cancer_model.h5')

print("Model saved successfully.")

import pandas as pd
hist_df = pd.DataFrame(history.history)
hist_df.head()

def classify_image(img_path, model):
  #loading and preprocess_image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0 #converts loaded image into numpy array
    # divides pixel value by 255 to normalize them into range 0 to 1
    img_array = np.expand_dims(img_array, axis=0)
    #Adds an extra dimension to the array along the axis=0. This is necessary because the model expects input in batches, even if you're classifying a single image.
    prediction = model.predict(img_array)[0][0]
    return "Malignant" if prediction > 0.5 else "Benign"

hist_df['loss'].plot()
hist_df['val_loss'].plot()
plt.title('Loss v/s Validation Loss')
plt.legend()
plt.show()

#input image
# Plot the image with classification result
def plot_classification(img_path, result):
    img = image.load_img(img_path, target_size=(150, 150))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Classification: {result}')
    plt.show()
    
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide the main tkinter window
Tk().withdraw()

# Open file picker dialog
imgpath = askopenfilename(
    title="Select an Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if imgpath:
    print(f"Selected image: {imgpath}")
    result = classify_image(imgpath, model)
    plot_classification(imgpath, result)
    print(result)
else:
    print("No image selected.")


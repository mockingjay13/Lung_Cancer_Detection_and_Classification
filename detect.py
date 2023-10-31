import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from segmentation import calculate_tumor_area

class CancerDetector:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, custom_objects={'F1': self.F1})

    def F1(self, y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+tf.keras.backend.epsilon())
        return f1_val
     

    def determine_stage(self, tumor_area):
        if tumor_area <= 300:  
            return "Stage I"
        elif tumor_area <= 600:  
            return "Stage II"
        else:
            return "Stage III"
        return "--"
        
    
    def predict(self, image_path):
        # data preprocessing
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Image couldn't be loaded. Check the path or the image format.")
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalizedImage = cv2.equalizeHist(gray_image)
        e, segmentedImage = cv2.threshold(equalizedImage, 128, 255, cv2.THRESH_TOZERO)
        stacked_img = cv2.merge([segmentedImage, segmentedImage, segmentedImage])
        resized_img = cv2.resize(stacked_img, (224, 224))
        img_array = resized_img / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        tumor_area = calculate_tumor_area(image_path)
        stage = self.determine_stage(tumor_area)

        prediction = self.model.predict(img_array)

        # Get the class with the highest probability
        predicted_class = np.argmax(prediction, axis=1)
        
        # Create a mapping from the label indices to their names
        class_mapping = {
            0: 'adenocarcinoma',
            1: 'large.cell.carcinoma',
            2: 'normal',
            3: 'squamous.cell.carcinoma'
        }
        
        # Return the class name
        return class_mapping[predicted_class[0]], tumor_area, stage

if __name__ == "__main__":
    detector = CancerDetector('lung_cancer_model.h5')
    image_path = 'path_to_your_image.jpg' # Replace with the path to the image which will be used when this file is run independently.
    result = detector.predict(image_path)
    print("Prediction:", result)

######################################################################################################################################

# Area per pixel =0.00264 sqcm
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense,Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers, optimizers
import tensorflow as tf
from keras.applications import ResNet50,VGG16,ResNet101, VGG19, DenseNet201, EfficientNetB4, MobileNetV2
from keras.applications import vgg16 
from keras import Model
from keras.optimizers.legacy import Adam
import keras.backend as K
import matplotlib.pyplot as plt
import pandas as pd
import PIL
import os
import cv2
import zipfile
import splitfolders


##############################################################################################################################################

# Data Pre-processing

##############################################################################################################################################

# unzip to extract raw data files
rawData_directory = './Raw_Data'
categories = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
processed_directory = './Processed_Data'

def unzip_folder(zip_path, output_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)

# Specify the path to the zipped file and the output directory
zipFile_path = 'Raw_Data_Zipped.zip'
unzip_directory = 'Raw_Data'  

# Create the output directory if it doesn't exist
if not os.path.exists(unzip_directory):
    os.makedirs(unzip_directory)

# Unzip the folder
unzip_folder(zipFile_path, unzip_directory)

for category in categories:
    path = os.path.join(rawData_directory, category)
    for image in os.listdir(path):
        image_Path = os.path.join(path, image)
        readImage = cv2.imread(image_Path, 0)
        equalizedImage = cv2.equalizeHist(readImage)
        e, segmentedImage = cv2.threshold(equalizedImage, 128, 255, cv2.THRESH_TOZERO)
        if category == 'normal':
            imageFinal = image_Path.replace('Raw_Data/normal', 'Processed_Data/normal')
            cv2.imwrite(imageFinal, segmentedImage)
        elif category == 'adenocarcinoma':
            imageFinal = image_Path.replace('Raw_Data/adenocarcinoma', 'Processed_Data/adenocarcinoma')
            cv2.imwrite(imageFinal, segmentedImage)
        elif category == 'large.cell.carcinoma':
            imageFinal = imageFinal.replace('Raw_Data/large.cell.carcinoma', 'Processed_Data/large.cell.carcinoma')
            cv2.imwrite(imageFinal, segmentedImage)
        elif category == 'squamous.cell.carcinoma':
            imageFinal = imageFinal.replace('Raw_Data/squamous.cell.carcinoma', 'Processed_Data/squamous.cell.carcinoma')
            cv2.imwrite(imageFinal, segmentedImage)
print("Successfully created processed data directory at:", processed_directory)


###############################################################################################################################################

# Splitting the processed images

###############################################################################################################################################

split_processed_directory = './Processed_Data_Split'

splitfolders.ratio(processed_directory, output=split_processed_directory, seed=6942, ratio=(.7, 0.1, 0.2)) 


###############################################################################################################################################

# Neural networks

###############################################################################################################################################

N_CLASSES = 4
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(dtype='float32', rescale= 1.0/255.0)

valid_datagen = ImageDataGenerator(dtype='float32', rescale= 1.0/255.0)

test_datagen = ImageDataGenerator(dtype='float32', rescale = 1.0/255.0)

train_DataSet  = train_datagen.flow_from_directory(directory = 'Processed_Data_Split/train',
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (224,224),
                                                   class_mode = 'categorical')

validate_DataSet = valid_datagen.flow_from_directory(directory = 'Processed_Data_Split/val',
                                                    batch_size = BATCH_SIZE,
                                                    target_size = (224,224),
                                                    class_mode = 'categorical')

test_DataSet = test_datagen.flow_from_directory(directory = 'Processed_Data_Split/test',
                                                 batch_size = BATCH_SIZE,
                                                 target_size = (224,224),
                                                 class_mode = 'categorical')


##############################################################################################################################################

# Resnet-50

##############################################################################################################################################

base_model = VGG16(include_top=False, pooling='avg', weights='imagenet', input_shape = (224,224,3))

for layer in base_model.layers:
    layer.trainable = False

model=Sequential()

model.add(base_model)
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(N_CLASSES, activation='softmax'))

optimizer = Adam(lr=0.001, decay=1e-6)
model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics = ['acc'])

checkpointer = ModelCheckpoint(filepath='./chest_CT_SCAN-/vgg16.hdf5',
                            monitor='val_loss', verbose = 1,
                            save_best_only=True)
early_stopping = EarlyStopping(verbose=1, patience=15)

model_history = model.fit(train_DataSet,
                    steps_per_epoch = 20,
                    epochs = 100,
                    verbose = 1,
                    validation_data = validate_DataSet,
                    callbacks = [checkpointer, early_stopping])

model_scores = model.evaluate(test_DataSet)

# Save the model
model.save('lung_cancer_model.h5')

accuracy_val = model_scores[1]
print('Accuracy: ', accuracy_val)

with open('accuracy.txt', 'w') as file:
    file.write(str(accuracy_val))
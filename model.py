
import csv 
import cv2
import numpy as np
import keras 
import imageutil as iu

# ----------------------------
# Tensorflow manual setting 
# to avoid 'CUBLAS_STATUS_ALLOC_FAILED' error
import tensorflow as tf
from keras import backend 
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
backend.set_session(sess)
#-----------------------------

#NVIDIA Model 
def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3), name='Normalization'))
    model.add(Conv2D(24, (5, 5), subsample=(2,2), activation='elu', name='conv1'))
    model.add(Conv2D(36, (5, 5), subsample=(2,2), activation='elu', name='conv2'))
    model.add(Conv2D(48, (5, 5), subsample=(2,2), activation='elu', name='conv3'))
    model.add(Conv2D(64, (3, 3), activation='elu', name='conv4'))
    model.add(Conv2D(64, (3, 3), activation='elu', name='conv5')) 
    model.add(Dropout(0.5))    
    model.add(Flatten())       
    model.add(Dropout(0.5))        
    model.add(Dense(100, activation='elu', name='fc1'))
    model.add(Dense(50, activation='elu', name='fc2'))
    model.add(Dense(10, activation='elu', name='fc3'))
    model.add(Dense(1, name='output'))
    return(model)

# Load Simulator data files 
lines = []

# with open('../Simulator/data_t1_1/driving_log.csv') as csvfile:
#      reader = csv.reader(csvfile)
#      for line in reader:        
#          lines.append(line)        
# print("CSV #1 read count: {}".format(len(lines)))

with open('../Simulator/data_t1_2_r/driving_log.csv') as csvfile:
     reader = csv.reader(csvfile)
     for line in reader:        
         lines.append(line)        
print("CSV #2 read count: {}".format(len(lines))) 

# with open('../Simulator/data/driving_log.csv') as csvfile:
#      reader = csv.reader(csvfile)
#      for line in reader:        
#          lines.append(line)        
# print("CSV #3 read count: {}".format(len(lines)))
      
with open('../Simulator/data2/driving_log.csv') as csvfile:
     reader = csv.reader(csvfile)
     for line in reader: 
         lines.append(line)        
print("CSV #4 read count: {}".format(len(lines)))

with open('../Simulator/data3/driving_log.csv') as csvfile:
      reader = csv.reader(csvfile)
      for line in reader: 
          lines.append(line)        
print("CSV #5 read count: {}".format(len(lines)))        

with open('../Simulator/data4_r/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:        
        lines.append(line)        
print("CSV #7 read count: {}".format(len(lines)))


images = []
measurements = [] 
sampling = False 

#Prepare Image Data for training from given inputs
images,measurements = iu.prepare_images(lines)
X_train = np.array(images)
y_train = np.array(measurements)
import statistics as st

# Store Data Histogram
st.generate_histogram(y_train)

print("Start Training with keras v{}.".format(keras.__version__))
print("X_train[0] Shape : {}".format(X_train[1].shape))
print("X_train Shape : {}".format(X_train.shape))

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
 
model = build_model()
from keras.optimizers import Adam
model.compile(loss='mse', optimizer=Adam(lr=0.001))

print("Model Ready. Start to training...")
model.fit(X_train,y_train,validation_split=0.2, shuffle=True,epochs=30)
model.save('model.h5')  

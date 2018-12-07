
#Simple Image Pre-Processing Library 
#@Author Bryan Baek 

from PIL import Image 
import numpy as np
import cv2

# Remove upper 60 pixels 
def preprocess(img1,img2,img3):
    image = concatenate(img1,img2,img3)
    dim = (320,160)
    processed_image = resize(image,dim) 
    return(processed_image)

#Concatenate 3 image as 1 horizontally.  
def concatenate(img1,img2,img3):
    con_img = np.concatenate((img1,img2,img3), axis=1) 
    return(con_img)

#Example : resize(image, (320,160))
def resize(image, dim=(200,66)):
    return cv2.resize(image,dim,interpolation = cv2.INTER_AREA )


#Crop Image
#NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]        
def crop(image):
    return image[60:135,20:300]  

    
#Color Change 
def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

#Flip Image 
def flip(image, steering_angle):
    image = cv2.flip(image,1)
    steering_angle = -steering_angle 
    return image,steering_angle

#Adjust Left Camera Image to utilize as training data. 
def adjust_left(image,steering_angle):
    steering_angle = steering_angle + 0.22 
    return image, steering_angle 

#Adjust Right Camera Image to utilize as training data. 
def adjust_right(image, steering_angle):
    steering_angle = steering_angle - 0.22 
    return image, steering_angle 

def to_array(image):
    return np.asarray(image)

def brightning_image(image):
    imageb = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    random_bright = 0.1+ np.random.uniform()
    imageb[:,:,2] = imageb[:,:,2]*random_bright
    imager = cv2.cvtColor(imageb, cv2.COLOR_HSV2BGR)
    return imager    

# add CV2 image, measurement to data
# CV2 image is converted to array, then added. 
def add_to_data(images, measurements, image, measurement):
    images.append(to_array(image))
    measurements.append(measurement)
    return images,measurements


# pince once for debugging 
alreadyPrinted = {}
def print_once(printKey,printStr):
    if printKey in alreadyPrinted:
        return
    alreadyPrinted[printKey] = True
    print("[{}] {}".format(printKey, printStr))
    return

# generate additional training data with given input
# randomly select by count
# modify it via iaa augment function 
# return it 
from imgaug import augmenters as iaa
import random
def generate_data(Xs,Ys,count, augfunc = iaa.Multiply((1,1.5) )):    
    genXs = []
    genYs = []
    # random sampling by count     
    for i in range(0,count):
        ri = random.randint(0, len(Xs))
        genXs.append(Xs[ri])
        genYs.append(Ys[ri])

    #Augment by augment function 
    augImages = augfunc.augment_images(genXs)
    return(augImages,genYs)

#Image preprocess for driving mode 
def process_image_drive(image):
    image2 = crop(image)        
    image2 = resize(image2)            
    image2 = rgb2yuv(image2)
    return(image2)

#Image preprocess for taining mode 
#Apply brightness change to image randomly 
def process_image(image):
    image2 = crop(image) 
    if(random.randint(0, 1000)<120):
        image2 = brightning_image(image2)       
    image2 = resize(image2)            
    image2 = rgb2yuv(image2)
    return(image2)

 
# There are so many '0' data, remove those by propotion. 
def adjust_distribution(lines):
    adjusted_lines = [] 
    for line in lines:
        measurement = float(line[3])        
        if(measurement == 0 ):
            r = random.randint(0, 1000)
            if(r<40):
                adjusted_lines.append(line)
        else:
            adjusted_lines.append(line)    
    return(adjusted_lines)                    

# Input - lines from 'driving_log.csv' 
# Output - Argumented Image ( FLIP, Left, Right )
def prepare_images(lines):
    images = []
    measurements = [] 
    lines = adjust_distribution(lines)
    print("Adjusted line count: {}".format(len(lines)))    
    print("Loading Images & Cropping....")
    sampling = False
    for line in lines:
        #Center Image 
        imageC = cv2.imread(line[0])  #BGR by default 
        imageCC = process_image(imageC)
        measurement = float(line[3])
        images, measurements = add_to_data(images,measurements, imageCC,measurement)         

        #Flip & Add 
        imageCCF,measurementF = flip(imageCC,measurement)
        images, measurements = add_to_data(images,measurements, imageCCF,measurementF)         

        #Left Image 
        imageL = cv2.imread(line[1])
        imageLC = process_image(imageL)
        imageLC, measurementA = adjust_right(imageLC, measurement)
        images, measurements = add_to_data(images,measurements, imageLC,measurementA)         

        #Flip & Add 
        imageLCF,measurementAF = flip(imageLC,measurementA)
        images, measurements = add_to_data(images,measurements, imageLCF,measurementAF)                 

        #Right Image 
        imageR = cv2.imread(line[2])    
        imageRC = process_image(imageR)
        imageRC, measurementA = adjust_right(imageRC,measurement)
        images, measurements = add_to_data(images,measurements,imageRC,measurementA)

        #Flip & Add
        imageRCF, measurementAF  = flip(imageRC, measurementA)
        images,measurements = add_to_data(images,measurements, imageRCF, measurementAF)
    
    print("Train data prepared images {}, measurements {} from org {}.".format(len(images),len(measurements),len(lines)))

    return images,measurements        

        


















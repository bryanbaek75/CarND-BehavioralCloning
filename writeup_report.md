#**Behavioral Cloning** 

##Bryan Baek - Self-Driving Car Term 3 

###Summarization of the experiment executed. 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/DataDistribution.png "Model Visualization"
[image2]: ./examples/nvidia_model.png "NVidia CNN Model"
[image3]: ./examples/driving_01.jpg "Driving foward"
[image4]: ./examples/driving_02.jpg "Driving foward 2"
[image5]: ./examples/reverse_01.jpg "Reverse direction"
[image6]: ./examples/reverse_02.jpg "Reverse direction 2"
[image7]: ./examples/recovery_01.jpg "Recovery"
[image8]: ./examples/recovery_02.jpg "Recovery 2"
[image9]: ./examples/original-01.jpg "Brightening before"
[image10]: ./examples/brightend-01.jpg "Brightening After"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* imageutil.py containing the utility script to manipulate image data 
* statistics.py containing the utility script to generate statics of given training data
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* output_video.mp3 for recorded autonomous driving

Note that I updated the KERAS to newest version. 

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. The imageutil.py contains the code for image data augmentation and proprocessing, and it is organizaed as a simple image utility library. 

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is NVIDIA behavioral cloning model (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). 

![alt text][image2]

It is utilized as the same as original. 

The input image is cropped from 160x320 to 75*300 and resize to 66x200 in order to fit the model. The model includes RELU layers and drop out layer to reduce the possibility of overfitting.  

####2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting (model.py lines 28~30). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 46-74). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an Adam optimizer, but the learning rate was set to 0.001 by manually (model.py line 101).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and reverse direction driving. Also, both drive by keyboard and drive by mouse is applied. Both data set were compensating each other by means of the steering value, and showed better distribution. 

![alt text][image1]


For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to adopt N-Vidia model and appropriate image data set. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set by utilizing KERAS fit function. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model and added dropout layer. 

But It easily went to off-load, then I tried to increase the training data by supplying more the manual drived image data such as reverse driving and recovery to the center. But all of those were not work well. 

So, I augmented image data by flipping and utilizing left, right camera images and flipping with that image also. It seems to work a little more but easily jumped into the lake while on the autonomous driving in the simulator. It seems to be a similar color data for the N-Vidia Model, so I added a random brighting augmentation. 

The I ran the simulator to see how well the car was driving around track one. As I mentioned, there were a few spots where the vehicle fell off the track. I augmented image data by flipping and utilizing left/right camera images but the problem was the largest portion of 0'degree steering data. After removing 96% of center images, it drive quite better, but still it jumped into the lake. 

Improve the driving behavior in these cases, I augmented ramdom brighting augmentation. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 20-35) consisted of a convolution neural network. 
Layer size and channel depth is described above model. 


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one. Here is an example image of center lane driving:

![alt text][image3]
![alt text][image4]

In order to capture more data, I recorded four laps by driving to reverse direction. Here is an example image of center lane driving:

![alt text][image5]
![alt text][image6]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to center. These images show what a recovery looks like starting from ... :

![alt text][image7]
![alt text][image8]

To augment the data set, I also flipped images and adjusted brightening. here is an image that has then been brightening:

![alt text][image9]
![alt text][image10]

After the collection process, I had X number of data points. I then preprocessed this data by 3 way. 

1. crop image to 75x280
2. resize image to 66x200 
3. change color coordinate from BGR2YUV

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by error rate. 

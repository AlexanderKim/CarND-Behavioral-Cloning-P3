# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I've used NVIDIA's neural network architecture mentioned in training videos and described here: https://devblogs.nvidia.com/deep-learning-self-driving-cars/

My model consists of the following layers:
* Normalization
* Cropping
* Convolutional layers x5
* Fully connected layers x3

![Neural net layers from NVIDIA's website](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

Since we frame the sterring prediction as regression problem, model is trained with MSE loss function, using Adam optimizer.


#### 2. Attempts to reduce overfitting in the model

Essentially for the sake of reducing overfitting I've produced more data using joystick and keyboard. 

#### 3. Model parameter tuning

Experimentally and according to training materials I've figured out that optimal number of epochs for this model architecture is 3

Below is the output out of model training with validation loss provided:
```sh
62272/62256 [==============================] - 159s - loss: 0.0288 - val_loss: 0.0273
Epoch 2/3
62304/62256 [==============================] - 158s - loss: 0.0252 - val_loss: 0.0250
Epoch 3/3
62272/62256 [==============================] - 158s - loss: 0.0258 - val_loss: 0.0260
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road both with joystick and keyboard, forward and reverse directions of the track. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Throughtout the process of building a final model architecture the approach evolved from simplistic neural net with 1 fully-connected layer to LeNet to NVIDIA's SDC neural net.

Interestingly enough, validation loss grew after certain numbers of epochs, and as I've described above, the optimal value for the final architecture is 3.

In regards of traingin data generation, I've driven a number of laps in the center of the lane with joystick and keyboard, then one lap on each side of the road. Repeated the same for the reverse direction.

Even though the output of training was promising as shown above, I've found the car to be in deficite of steering. I've palyed with driving parameters, fist and foremost the speed, and it went fine.

My assumption is that joystic is far more precise and continuous than keyboard. Imagine you're steering leaner, but continously as opposed to discrete, but with higher magnitude.
Accumulated steering is going to be the same, but timing matters. So the assumption is that in training mode input is provided from joystick directly, and it is more frequent than from drive.py in autonomous mode, hence the steering in unnderaccumulating if it makes sense.
Hence I've multiplied predicted steering by 1.5 in drive.py to compensate it.

Another interesting observation is that over time my driving manner tended to be corner-cutting or closer to a racing line as if I was playing a racing game. Combined with a handful of laps on the side it resulted in using both sides and the center lane of the road. You can see it in the video below

#### 2. Output video
Find video with the result in output_video.mp4

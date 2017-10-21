# Project: Follow Me

This project is on deep learning. In this project, a neural network is trained to identify and track a target (a hero in red) in the simulator.

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 


### Walkthrough
The model structure used is a 5 layer full convolutional network model. It has 2 encoders, 1x1 convolution layer and 2 decoders.

![model](./images/FCN_model.png)

#### Encoder
Separable convolution is used for the encoders. Separable convolutions is a technique that reduces the number of parameters needed, thus increasing efficiency for the encoder network. It requires less parameters, making it more efficient and reducing runtime performance.
The first encoder outputs a depth of 32 from 3, and the second encoder outputs a depth of 64.

#### 1x1 Convolution layer
The 1x1 convolution has a kernel size of 1 and stride of 1. It outputs a depth of 128.

#### Decoder
The decoder bilinear sampling, where it takes the input and first layer as inputs to be fed into to do upsampling. The decoder will upsample and concatenate (refer to image above), followed by passing through 2 separable convolution layers.


### Training and validation data
More training data were added to increase the training accuracy. There were different scenarions taken into account:
1. Hero in a dense area, where there are other people and can be mistaken for the hero
2. Different angles of the hero, the hero would be zigzagging
3. Some overlapping between the patrol area and hero walking area

![model](./images/training_zigzag.png)


![model](./images/training_random.png)

### Pitfalls
Convolution layer is determined by kernel size. 1x1 convolution layer would require a kernel size of 1.
Number of output is determined by both kernel_size and strides. For a kernel_size 3x3 means these 9 numbers in inputs, multiplied by w1, we can get 1 value. If the kernel_size is 5x5, one value of output would be calculated with 25 values from input.

## Future enhancements



## Scenarios

There are different hyperparameters that can be used to obtain different results. 
The final score is based of this equation: average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))

#### Scenario 1
The default training and validation data that were provided were used for training. There were about 4000 data for training and 1000 data for validation. The following params has been used:

	learning_rate = 0.01
	batch_size = 50
	num_epochs = 20
	steps_per_epoch = 50
	validation_steps = 50
	workers = 2

With the score of 0.3889

#### Scenario 2
A new training and validation data that was self trained were used for training. There were about 2000 data for training and 1000 data for validation. The following params has been used.

	learning_rate = 0.005
	batch_size = 50
	num_epochs = 20
	steps_per_epoch = 80
	validation_steps = 50
	workers = 2

With the score of 0.0354653

There is a signicant drop in result. This might be due to poor quality training data, or bad batch_size/steps_per_epoch. Also note in this case, the total number (batch_size*steps_per_epochs = 4000) is higher than the total number of training data (2000). 


#### Scenario 3
A new training and validation data that was self trained were used for training. There were about 2000 data for training and 1000 data for validation. The following params has been used. With half the learning rate.

	learning_rate = 0.01
	batch_size = 50
	num_epochs = 20
	steps_per_epoch = 80
	validation_steps = 50
	workers = 2

With the score of 0.1348
Comparing the learning rate (double the scenario 2), the learning rate has doubled, the final score has risen significantly. 

#### Scenario 4
A new training and validation data that was self trained were used for training. There were about 2000 data for training and 1000 data for validation. The following params has been used. The total number of steps has halved to ensure that the same data is not trained twice.

	learning_rate = 0.01
	batch_size = 50
	num_epochs = 20
	steps_per_epoch = 50
	validation_steps = 50
	workers = 2

With the score of 0.106
The steps of per epoch was nearly halved. There is a drop in the score as it meant probably lack of total training as some of the training data was not trained. The step_per_epoch should be approximately `total_sample_num/batch_size`. If it's too large, there are some pictures are used over one time for one epoch, if it's too small, some figures didn't be used in one epoch

#### Scenario 5
A new training and validation data that was self trained were used for training. There were about 2000 data for training and 1000 data for validation. The following params has been used. The following params has been used. The number of epochs doubled.

	learning_rate = 0.01
	batch_size = 50
	num_epochs = 40
	steps_per_epoch = 50
	validation_steps = 50
	workers = 2

With the score of 0.091006864897
As the smaller steps needs more epochs to train, to ensure most data has been trained at least once. However as it trains further, there is an odd spike. There is a drop in the result, which probably meant the data is overtrained.

#### Scenario 6
Training and validation data are combined with the sample default and self trained to increase the number of training data. There were about 6000 data for training and 2000 data for validation. 

	learning_rate = 0.01
	batch_size = 40
	num_epochs = 40
	steps_per_epoch = 50
	validation_steps = 50
	workers = 2

With the score of 0.4289
A better training set definitely increased the score.

#### Scenario 7
Training and validation data are combined with the sample default and self trained to increase the number of training data. There were about 6000 data for training and 2000 data for validation. 

	learning_rate = 0.01
	batch_size = 40
	num_epochs = 20
	steps_per_epoch = 80
	validation_steps = 50
	workers = 2

With the score of 0.4538235
Playing around with the steps per epoch, to ensure most of the training data has been trained by the model. An increase of steps per epoch has a slight increment in the final score.


#### Scenario 8
Training and validation data are combined with the sample default and self trained to increase the number of training data. There were about 6000 data for training and 2000 data for validation. 

	learning_rate = 0.01
	batch_size = 40
	num_epochs = 40
	steps_per_epoch = 80
	validation_steps = 50
	workers = 2

With the score of 0.439556
Increasing the number of epochs seems to have a drop in final score. This may indicate that the model may the overfitted. Thus a drop in score.

#### Scenario 9
Training and validation data are combined with the sample default and self trained to increase the number of training data. There were about 6000 data for training and 2000 data for validation. 

	learning_rate = 0.005
	batch_size = 40
	num_epochs = 20
	steps_per_epoch = 80
	validation_steps = 50
	workers = 2

With the score of 0.4647
Halving the learning rate seems to work better.

#### Scenario 10
Training and validation data are combined with the sample default and self trained to increase the number of training data. There were about 6000 data for training and 2000 data for validation. 

	learning_rate = 0.005
	batch_size = 40
	num_epochs = 40
	steps_per_epoch = 80
	validation_steps = 50
	workers = 2

With the score of 0.45320


#### Scenario 11
Training and validation data are combined with the sample default and self trained to increase the number of training data. There were about 6000 data for training and 2000 data for validation. 

	learning_rate = 0.005
	batch_size = 40
	num_epochs = 20
	steps_per_epoch = 100
	validation_steps = 50
	workers = 2

With the score of 0.46090

#### Scenario 12
Training and validation data are combined with the sample default and self trained to increase the number of training data. There were about 6000 data for training and 2000 data for validation. 
Adding another layer for encoding and decoding. It increases training time (slightly)

	learning_rate = 0.005
	batch_size = 40
	num_epochs = 20
	steps_per_epoch = 80
	validation_steps = 50
	workers = 2

With the score of 0.388

#### Scenario 13
Training and validation data are combined with the sample default and self trained to increase the number of training data. There were about 6000 data for training and 2000 data for validation. 
Adding another layer for encoding and decoding ( 3 each in total). And the convolution layer in the middle would have double the amount compared to last encoder.

	learning_rate = 0.005
	batch_size = 40
	num_epochs = 20
	steps_per_epoch = 80
	validation_steps = 50
	workers = 2

With the score of 0.303
The score has dropped, and this is due to an increase in encoding/decoding layer. This may be due to overfitting.



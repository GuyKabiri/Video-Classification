# Video Classification
The third exercise in the course Neural Networks for Computer Vision.  
UCF101 dataset was used to train CNN classification architecture to classify videos.  
There are two parts to the exercise:  
First, an early fusion video classification architecture is implemented.  
Second, a SlowFast video classifier using a ResNet-50 architecture and PytorchVideo.

Submitted By:

* Tal Goldengoren
* Guy Kabiri

Table of Contents:
* [Data](#data)
    * [Data Samples](#data-samples)
        * [Basketball](#basketball-samples)
        * [Rope Climbing](#ropeclimbing-samples)
        * [Biking](#biking-samples)
        * [Diving](#diving-samples)
        * [Pizza Tossing](#pizzatossing-samples)
    * [Data Exploration](#data-exp)
* [Part 1 - Early Methods](#part1)
    * [Early Fusion](#early-fusion)
    * [Training](#part1-training)
    * [Results](#part1-results)
* [Part 2 - Modern Methods](#part2)
    * [SlowFast](#slowfast)
    * [Training](#part2-training)
    * [Results](#part2-results)
        * [ResNet-50](#part2-resnet50)
* [References](#refs)

# Data <span id="data"><span>

The dataset used in this task is [UCF101](https://www.crcv.ucf.edu/data/UCF101.php).  
UCF 101 is a set of action recognition videos from YouTube, with 101 action categories.  
In this exercise, five classes were used:  
Basketball, Rope Climbing, Biking, Diving, Pizza Tossing.

## Data Samples <span id="data-samples"></span>

### Basketball <span id="basketball-samples"></span>
![ezgif com-gif-maker](https://user-images.githubusercontent.com/52006798/146648149-7b2a60dd-c4dc-486d-bf74-6ee36b51e24b.gif)
![ezgif com-gif-maker(1)](https://user-images.githubusercontent.com/52006798/146648151-e9b93f3f-75f9-4ba8-af38-b9360b709563.gif)
![ezgif com-gif-maker(2)](https://user-images.githubusercontent.com/52006798/146648154-893fdd3f-9f9b-4348-9e62-1ee6f4d24994.gif)

### Rope Climbing <span id="ropeclimbing-samples"></span>

![ezgif com-gif-maker(12)](https://user-images.githubusercontent.com/52006798/146648141-c1d88761-cb3c-44fc-bd1d-adc078c49eb7.gif)
![ezgif com-gif-maker(13)](https://user-images.githubusercontent.com/52006798/146648144-bcd00631-6c35-4d2b-8bf8-dbe949401702.gif)
![ezgif com-gif-maker(14)](https://user-images.githubusercontent.com/52006798/146648146-e1c00dd1-80d8-4f62-82bd-b861782ea643.gif)

### Biking <span id="biking-samples"></span>

![ezgif com-gif-maker(3)](https://user-images.githubusercontent.com/52006798/146648156-c329b5d9-271f-49d9-b420-0d88f1d0cad6.gif)
![ezgif com-gif-maker(4)](https://user-images.githubusercontent.com/52006798/146648158-ec8451ae-8bba-4808-8783-edfe74f3cec9.gif)
![ezgif com-gif-maker(5)](https://user-images.githubusercontent.com/52006798/146648160-2b1e509a-d324-4739-8081-268f6ad2c229.gif)

### Diving <span id="diving-samples"></span>

![ezgif com-gif-maker(8)](https://user-images.githubusercontent.com/52006798/146648134-1f1960fb-0d1c-40f4-8bde-68e9b07e917a.gif)
![ezgif com-gif-maker(6)](https://user-images.githubusercontent.com/52006798/146648167-f6eb1d56-9c7a-4146-95dc-5a49fdc54fd7.gif)
![ezgif com-gif-maker(7)](https://user-images.githubusercontent.com/52006798/146648171-d3c9ba37-78b4-4f12-af1f-2bdd28eb22a8.gif)

### Pizza Tossing <span id="pizzatossing-samples"></span>

![ezgif com-gif-maker(9)](https://user-images.githubusercontent.com/52006798/146648135-2ebf956e-e239-4d8d-9d52-57c34070be83.gif)
![ezgif com-gif-maker(10)](https://user-images.githubusercontent.com/52006798/146648137-fdb8c9b9-f80b-4bf9-94d2-2a6188312040.gif)
![ezgif com-gif-maker(11)](https://user-images.githubusercontent.com/52006798/146648140-6b541150-45c9-4de5-914b-f74867f7a74e.gif)

## Data Exploration <span id="data-exp"></span>


```python
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import os
```


```python
phases = [ 'train', 'valid', 'test' ]
paths = { p: 'data/{}'.format(p) for p in phases }
```


```python
df = pd.DataFrame(columns=[ 'name', 'class', 'frames', 'phase' ])
```


```python
for p in phases:
    for cls in os.listdir(paths[p]):     #   iterate over the classes of each phase
        class_path = os.path.join(paths[p], cls)
        videos = [ f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) ]
        for vid in videos:
            video_path = os.path.join(class_path, vid)

            cap = cv2.VideoCapture(video_path)
            df = df.append({
                'name': vid,
                'class': cls,
                'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'phase': p
            }, ignore_index=True)
            cap.release()
            
```


```python
for p in phases:
    print('Samples in {} set: {}'.format(p, len(df[df['phase']==p])))
```

    Samples in train set: 522
    Samples in valid set: 128
    Samples in test set: 18



```python
def plot_class_count(df):
    fig = plt.figure(figsize=(15, 7))
    ax = sns.countplot(x='class', hue='phase', data=df)
    for p in ax.patches:
        ax.annotate(p.get_height(), (p.get_x()+0.1, p.get_height()+0.15))


plot_class_count(df)
```


    
![png](submission_files/submission_16_0.png)
    



```python
def plot_frames_dist(df):
    # fig = plt.figure(figsize=(20, 10))
    ax = sns.displot(df, x='frames', hue='phase', kde=True, element='poly', col='class', )

plot_frames_dist(df)
```


    
![png](submission_files/submission_17_0.png)
    


# Part 1 - Early methods <span id="part1"><span>
The first part of the exercise includes two early video processing methods, Early Fusion, and Late Fusion.  

Unlike image classification, where a single image has three dimensions, video classification adds a fourth dimension - Time/Frame.  
In order to make a CNN model suitable for video classification, a modification is required.  
Several methods exist for dealing with this issue, including Early Fusion, Late Fusion, Slow Fusion, and Fast Fusion.  
![image](https://user-images.githubusercontent.com/52006798/146635639-db3cd74f-7889-405c-ae82-1e3aaab9c0e8.png)  
Implementing this exercise required either Early Fusion or Late Fusion, and Early Fusion was selected.

## Early Fusion <span id="early-fusion"><span>
In a CNN model, the first convolution layer has an input of three channels, representing the three color channels in the image.  
During video-CNN processing, the first layer handles 3 channels of color simultaneously in each frame.  

How it works:  
Video samples representing batch will be [ B, F, 3, H, W ]. Image CNN architectures, however, utilize 2D CNN blocks.  
Additionally, all of the frames must be analyzed simultaneously. There is a reshape required, so the first convolution layer will be performed on all frames in the architecture.  
Therefore, the new batch size dimensions are [ B, 3F, H, W ].  
It means two things, each batch should be modified at runtime, and the first CNN layer in the architecture should be changed in advance to be in the shape of [3F, H, W].  
Finally, when dealing with pre-trained architecture, the last fully connected layer should be modified as well to the desired number of classes.  

Legend:  
* `B` - batch size
* `F` - number of frames
* `C` - number of channels
* `H` - height
* `W` - width

## Training <span id="part1-training"><span>
Various configurations were evaluated to determine which was the best.  
EfficientNet, ResNet, and ResNeXt architectures were used.  
The Adam and AdamW optimizers were tried, as well as different learning rates and schedulers.  
Additionally, two methods of converting video into frames were used: 

1. Static-Batch: Data preprocessing by selecting a subset of frames from each video and training only on these frames.  
As the frames were already stored in the storage and only needed to be loaded, training progress should be incredibly rapid. However, a small dataset may quickly overfit.  
Each frame was sampled by a fixed step in this method. step = len(total_frames) / len(subset_frames).  

<img src="https://user-images.githubusercontent.com/52006798/147848832-1f1e3890-9aa0-4789-b054-97c17a1d8a84.png" width="30%"/>    

2. Dynamic-Batch: Sample a random group of frames each time.  
This would be slower, but it could prevent overfitting since every time a video is loaded, a different set of frames is processed by the network.  

<img src="https://user-images.githubusercontent.com/52006798/147848886-b4e54060-7d7c-402a-9fba-ad86eb46df5c.png" width="30%"/>

## Results <span id="part1-results"><span>
The following graphs show a sample of different configurations that were tried.


<img src="https://user-images.githubusercontent.com/52006798/147827538-4083f42a-1b44-4592-a3ab-043ddd42911d.png" width="30%"/><img src="https://user-images.githubusercontent.com/52006798/147827539-1c5831b5-adb5-4d7d-884b-ebb22a231001.png" width="30%"/>

<img src="https://user-images.githubusercontent.com/52006798/147827535-844af00b-dfe3-43e9-81e3-afc93cd5c12c.png" width="30%"/><img src="https://user-images.githubusercontent.com/52006798/147827536-9aef5827-7bac-4869-ace6-36e728cb8a6b.png" width="30%"/><img src="https://user-images.githubusercontent.com/52006798/147827540-7b63f025-0fc1-448e-b5ed-5beea13a3cac.png" width="30%"/>

Finally, the preferred configuration was as follows:  

*   Pre-trained ResNet-50.
*   Pre-processed data (static dataset - the same frames were used in each video during training, instead of random ones).
*   32 frames for each video.
*   Optimizer: Adam.
*   Scheduler: ReduceLROnPlateau (factor=0.8, patience=2).
*   Learning rate: 1e-5.
*   Batch-size: 8.
*   Epochs: 100.

<img src="https://user-images.githubusercontent.com/52006798/147827800-eb667b7d-d310-4b30-aeab-c0da599c29fe.png" width="45%"/><img src="https://user-images.githubusercontent.com/52006798/147827802-d2ae0176-278b-4359-b85c-497eacaf0ad8.png" width="45%"/>

<img src="https://user-images.githubusercontent.com/52006798/147827797-b543147a-7484-4f89-a196-881043f2abf9.png" width="30%"/><img src="https://user-images.githubusercontent.com/52006798/147827798-8d0ca0ac-9f3e-436b-882d-ce220d58152b.png" width="30%"/><img src="https://user-images.githubusercontent.com/52006798/147827799-8c85523d-1104-48cb-8121-db125124737e.png" width="30%"/>

As can see above, we achieved accuracy of almost 0.9 and loss a little less than 0.7.

# Part 2 - Modern Methods <span id="part2"></span>

## SlowFast <span id="slowfast"></span>

The SlowFast architecture handles video in two ways.  
In order to capture spatial semantics, the first approach uses a low frame rate, which will be called the Slow pathway.  
A high frame rate is used in the second method, called the Fast pathway, to capture motion.  

![](https://feichtenhofer.github.io/pubs/slowfast_concept.gif)  

While both pathways use ResNet as their backbone, unlike the previous model, these use 3D CNN blocks.  
A small number of channels is used to keep the Fast pathway fast since it samples many more frames than the Slow pathway.  
In addition, the Fast pathway results are fed into the Slow pathway to improve prediction (taking into account the different shapes in each pathway).  
The results from the Slow pathway are not fed into the Fast pathway since the researchers found no improvements while executing it.

## Training <span id="part2-training"></span>

SlowFast's training phase required some modifications in the data processing.  
Firstly, SlowFast splits into 2 pathways, each of which consumes a different number of frames.  
Second, the tensor shape differs from the early method.  
While the early methods architecture was fed by [ B, F, 3, H, W ] tensors, the SlowFast architecture was fed by [ B, 3, F, H, W ].  
In order to use the same dataset, when handling a SlowFast model, the tensor needs to be reshaped into the correct dimensions and then split into two tensors.  
Getting the [ B, 3, F, H, W ] shape for the fast pathway, and the [ B, 3, F/4, H, W ] shape for the slow pathway.

## Results <span id="part2-results"></span>

Configuration used:  

*   Pre-trained SlowFast.
*   Dynamic data (sample random frames from a video at each epoch).
*   32 frames for each video.
*   Optimizer: ASGD.
*   Scheduler: ReduceLROnPlateau (factor=0.9, patience=4).
*   Scheduler: CosineAnnealingLR
*   Learning rate: 1e-3.
*   Batch-size: 4.
*   Epochs: 100.

Despite trying several different configurations, we were unable to achieve good results with the SlowFast model.  
Therefore, we can't compare the models since SlowFast should yield better results than the simpler methods.

<img src="https://user-images.githubusercontent.com/52006798/149008482-d1fa2847-8539-4e8f-818d-edce2274f427.png" width="45%"/><img src="https://user-images.githubusercontent.com/52006798/149008486-76272fc2-bbc6-420e-8964-c69227e58ea6.png" width="45%"/>

<img src="https://user-images.githubusercontent.com/52006798/149008476-4e424da6-f4c3-4029-9a31-a0315df3f4ee.png" width="30%"/><img src="https://user-images.githubusercontent.com/52006798/149008492-071af52c-0e36-4082-a422-53a921e8c12e.png" width="30%"/><img src="https://user-images.githubusercontent.com/52006798/149008494-3c0f927b-ef92-434b-ab86-55575f3828ab.png" width="30%"/>

# References <span id="refs"><span>
* [Video Classification with CNN, RNN, and PyTorch](https://medium.com/howtoai/video-classification-with-cnn-rnn-and-pytorch-abe2f9ee031)
* [Video Classification with CNN+LSTM](https://discuss.pytorch.org/t/video-classification-with-cnn-lstm/113413/4)
* [Introduction to Video Classification](https://towardsdatascience.com/introduction-to-video-classification-6c6acbc57356)
* [Justin Johnson's Lecture](https://web.eecs.umich.edu/~justincj/slides/eecs498/FA2020/598_FA2020_lecture18.pdf)
* [Christoph Feichtenhofer](https://feichtenhofer.github.io/)
* [SlowFast Explained: Dual-mode CNN for Video Understanding](https://towardsdatascience.com/slowfast-explained-dual-mode-cnn-for-video-understanding-8bf639960256)

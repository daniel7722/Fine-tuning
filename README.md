# Fine-tuning

## Progress

### 1. **Implement ResNet20**
   - Foundation Architecture:
  

| Layer (type)               | Output Shape         | Param       |
| -------------------------- | -------------------- | ----------- |
| input_1 (InputLayer)       | [(None, 32, 32, 3)]  | 0           |
| conv2d (Conv2D)            | (None, 32, 32, 16)   | 448         |
| batch_normalization        | (None, 32, 32, 16)   | 64          |
| activation                 | (None, 32, 32, 16)   | 0           |
| conv2d_1 (Conv2D)          | (None, 32, 32, 16)   | 2320        |
| batch_normalization_1      | (None, 32, 32, 16)   | 64          |
| activation_1               | (None, 32, 32, 16)   | 0           |
| conv2d_2 (Conv2D)          | (None, 32, 32, 16)   | 2320        |
| batch_normalization_2      | (None, 32, 32, 16)   | 64          |
| add                        | (None, 32, 32, 16)   | 0           |
| activation_2               | (None, 32, 32, 16)   | 0           |
| conv2d_3 (Conv2D)          | (None, 32, 32, 16)   | 2320        |
| batch_normalization_3      | (None, 32, 32, 16)   | 64          |
| activation_3               | (None, 32, 32, 16)   | 0           |
| conv2d_4 (Conv2D)          | (None, 32, 32, 16)   | 2320        |
| batch_normalization_4      | (None, 32, 32, 16)   | 64          |
| add_1                      | (None, 32, 32, 16)   | 0           |
| activation_4               | (None, 32, 32, 16)   | 0           |
| conv2d_5 (Conv2D)          | (None, 32, 32, 16)   | 2320        |
| batch_normalization_5      | (None, 32, 32, 16)   | 64          |
| activation_5               | (None, 32, 32, 16)   | 0           |
| conv2d_6 (Conv2D)          | (None, 32, 32, 16)   | 2320        |
| batch_normalization_6      | (None, 32, 32, 16)   | 64          |
| add_2                      | (None, 32, 32, 16)   | 0           |
| activation_6               | (None, 32, 32, 16)   | 0           |
| conv2d_7 (Conv2D)          | (None, 16, 16, 32)   | 4640        |
| batch_normalization_7      | (None, 16, 16, 32)   | 128         |
| activation_7               | (None, 16, 16, 32)   | 0           |
| conv2d_8 (Conv2D)          | (None, 16, 16, 32)   | 9248        |
| batch_normalization_8      | (None, 16, 16, 32)   | 128         |
| lambda                     | (None, 16, 16, 32)   | 0           |
| add_3                      | (None, 16, 16, 32)   | 0           |
| activation_8               | (None, 16, 16, 32)   | 0           |
| conv2d_9 (Conv2D)          | (None, 16, 16, 32)   | 9248        |
| batch_normalization_9      | (None, 16, 16, 32)   | 128         |
| activation_9               | (None, 16, 16, 32)   | 0           |
| conv2d_10 (Conv2D)         | (None, 16, 16, 32)   | 9248        |
| batch_normalization_10     | (None, 16, 16, 32)   | 128         |
| add_4                      | (None, 16, 16, 32)   | 0           |
| activation_10              | (None, 16, 16, 32)   | 0           |
| conv2d_11 (Conv2D)         | (None, 16, 16, 32)   | 9248        |
| batch_normalization_11     | (None, 16, 16, 32)   | 128         |
| activation_11              | (None, 16, 16, 32)   | 0           |
| conv2d_12 (Conv2D)         | (None, 16, 16, 32)   | 9248        |
| batch_normalization_12     | (None, 16, 16, 32)   | 128         |
| add_5                      | (None, 16, 16, 32)   | 0           |
| activation_12              | (None, 16, 16, 32)   | 0           |
| conv2d_13 (Conv2D)         | (None, 8, 8, 64)     | 18496       |
| batch_normalization_13     | (None, 8, 8, 64)     | 256         |
| activation_13              | (None, 8, 8, 64)     | 0           |
| conv2d_14 (Conv2D)         | (None, 8, 8, 64)     | 36928       |
| batch_normalization_14     | (None, 8, 8, 64)     | 256         |
| lambda_1                   | (None, 8, 8, 64)     | 0           |
| add_6                      | (None, 8, 8, 64)     | 0           |
| activation_14              | (None, 8, 8, 64)     | 0           |
| conv2d_15 (Conv2D)         | (None, 8, 8, 64)     | 36928       |
| batch_normalization_15     | (None, 8, 8, 64)     | 256         |
| activation_15              | (None, 8, 8, 64)     | 0           |
| conv2d_16 (Conv2D)         | (None, 8, 8, 64)     | 36928       |
| batch_normalization_16     | (None, 8, 8, 64)     | 256         |
| add_7                      | (None, 8, 8, 64)     | 0           |
| activation_16              | (None, 8, 8, 64)     | 0           |
| conv2d_17 (Conv2D)         | (None, 8, 8, 64)     | 36928       |
| batch_normalization_17     | (None, 8, 8, 64)     | 256         |
| activation_17              | (None, 8, 8, 64)     | 0           |
| conv2d_18 (Conv2D)         | (None, 8, 8, 64)     | 36928       |
| batch_normalization_18     | (None, 8, 8, 64)     | 256         |
| add_8                      | (None, 8, 8, 64)     | 0           |
| activation_18              | (None, 8, 8, 64)     | 0           |
| global_average_pooling2d   | (None, 64)           | 0           |
| flatten                    | (None, 64)           | 0           |
| dense                      | (None, 10)           | 650         |


#### Training & Validation Loss and Accuracy
<img src='https://github.com/daniel7722/Fine-tuning/assets/74921405/2a25e066-9ac6-4e1a-8f52-364739a43170' width='600'>

Total params: 271786 (1.04 MB) <br>
Trainable params: 270410 (1.03 MB) <br>
Non-trainable params: 1376 (5.38 KB) <br>

config: 
- learning rate: <br>
boundaries = [32000, 48000]<br>
values = [0.1, 0.01, 0.001]<br>

#### Results
Test loss: 0.576532244682312 / Test accuracy: 0.8963000178337097

### 2. **Add Dropout layer after residual block and implement Early Stopping**: result isn't great
   - Drop out layer is added after Flatten with drop out rate 0.5
   - It consists of a Dense, a BatchNorm, an Activation, and a Dropout
   - Early stopping is added to the callback, hence it stops at epoch 60 something hindering further progress

#### Training & Validation Loss and Accuracy
<img src='https://github.com/daniel7722/Fine-tuning/assets/74921405/beddb697-d5a8-4c41-8b41-1d2220aef47f' width='600'>

config: 
- learning rate: <br>
boundaries = [32000, 48000]<br>
values = [0.1, 0.01, 0.001]<br>
- Dropout rate: 0.5
- Early Stopping
- One more block

### 3. **Remove Early Stopping**:

#### Training & Validation Loss and Accuracy
<img src='https://github.com/daniel7722/Fine-tuning/assets/74921405/d1c390c8-5d04-41e6-be13-a6a323061798' width='600'>

config: 
- learning rate: <br>
boundaries = [32000, 48000]<br>
values = [0.1, 0.01, 0.001]<br>
- Dropout rate: 0.5
- One more block

#### Results
Test loss: 0.5847798585891724 / Test accuracy: 0.8770999908447266

### 4. **Remove final layer that was added previously**
   - Drop out rate remains 0.5
   - Now it's Flatten $\rightarrow$ Dropout $\rightarrow$ Output
  
Total params: 272170 (1.04 MB)<br>
Trainable params: 270602 (1.03 MB)<br>
Non-trainable params: 1568 (6.12 KB)<br>

#### Training & Validation Loss and Accuracy
<img src='https://github.com/daniel7722/Fine-tuning/assets/74921405/64682f7e-2383-4fcd-a96e-1d5c6059a971' width='600'>

#### Results
Test loss: 0.5499668717384338 / Test accuracy: 0.8998000025749207

config: 
- learning rate: <br>
boundaries = [32000, 48000]<br>
values = [0.1, 0.01, 0.001]<br>
- Dropout rate: 0.5

### 5. **Adjust learning rate scheduler**
   - In view of the validation loss' pattern observed in previous graph, I adjust learning rate scheduling so it is smoother
#### Training & Validation Loss and Accuracy
<img src='https://github.com/daniel7722/Fine-tuning/assets/74921405/4ce941ed-ea5b-46e1-8aab-82ea6965e6d1' width='600'>

config: 
- learning rate: 
boundaries = [20000, 32000, 56000]<br>
values = [0.1, 0.02, 0.005, 0.001]<br>
- Drop out rate: 0.5

#### Results
Test loss: 0.6009443402290344 / Test accuracy: 0.8934999704360962


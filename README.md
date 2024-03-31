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

#### Training & Validation Accuracy
<img src='https://github.com/daniel7722/Fine-tuning/assets/74921405/3922508b-5419-45d6-a38d-fa2ec769677b' width='600'>

#### Training & Validation Loss
<img src='https://github.com/daniel7722/Fine-tuning/assets/74921405/7276a14f-e136-4403-ae54-2e187edc376b' width='600'>

Total params: 271786 (1.04 MB) <br>
Trainable params: 270410 (1.03 MB) <br>
Non-trainable params: 1376 (5.38 KB) <br>

### 2. **Add Dropout layer after residual block and implement Early Stopping**: result isn't great
   - Drop out layer is added after Flatten with drop out rate 0.5
   - It consists of a Dense, a BatchNorm, an Activation, and a Dropout
   - Early stopping is added to the callback, hence it stops at epoch 60 something hindering further progress

#### Training & Validation Accuracy
<img src='https://github.com/daniel7722/Fine-tuning/assets/74921405/c01e9bc7-b5c7-4ee3-89ed-ebf31838c66b' width='600'>

#### Training & Validation Loss
<img src='https://github.com/daniel7722/Fine-tuning/assets/74921405/f89ace7b-58f8-4044-9de1-5867d508a62b' width='600'>

### 3. **Remove Early Stopping**:

#### Training & Validation Accuracy
<img src='https://github.com/daniel7722/Fine-tuning/assets/74921405/b21ae649-b292-4707-aa04-a7a533d243ea' width='600'>

#### Training & Validation Loss
<img src='https://github.com/daniel7722/Fine-tuning/assets/74921405/57e8828d-ceac-4cda-8fd5-eab42558ca40' width='600'>

#### Results
Test loss: 0.5847798585891724 / Test accuracy: 0.8770999908447266

### 4. **Remove final layer that was added previously**
   - Drop out rate remains 0.5
   - Now it's Flatten $\rightarrow$ Dropout $\rightarrow$ Output
  
Total params: 272170 (1.04 MB)<br>
Trainable params: 270602 (1.03 MB)<br>
Non-trainable params: 1568 (6.12 KB)<br>





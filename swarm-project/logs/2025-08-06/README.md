## 08-06
I try just plop the whole foreign dataset to the system, hoping both models will learn something during the process and become a better model, and fusion unit, though will experience some hard time during the beginning, will pick up when agents start to produce good result. But turns out, the final result suck and there are several reasons that cause this. For a primary reason, the agents never learn good features. They are producing random result the whole time, making fusion unit really hard to understand what's going on. Therefore, I decided to first produce a good agent, then we can put verdict on the performance of the fusion unit. 

Real dataset of AVE is implemented with vision agent having efficientnetb0 as its based and LSTM based model for audio agent. Vision is pretrained with imagenet while audio agent has a randomly initialised model. Data distribution is a three-fold split with training/validating/testing datasets with 6/2/2 split. The setup is like we pre-train both models with training dataset in 5 epoch. Then, use validationg set to see check the quality of the resulting pretrained models. Finally, using testing data to train fusion unit to see if it has learned anything. 

So far the result is very demotivating. We have good pretrained accuracy to 0.93 for vision model. Reasonable since it has pre-trained weight on imagenet. But bad validation loss implies a significant overfit, leading to a not so ideal model. Audio model learns nothing as expected since it's random initialisation. Hence, from this pov, I was expecting maybe fusion unit would trust vision agent fully and make a good result of that. But it turns out fusion unit learns nothing again with really bad result. 

Hence, the next run name "att_2" will be adapting to this situation where we 
- freeze vision model backbone and only train the classifier to avoid overfitting
- don't pretrain audio model to make it a fully randomised model
- manually assign hedge weight so I know vision model is always more trusted than audio model
- freeze the continual improvement as well during sim
- log fusion loss every 100 rounds

Cool, this run is at least going somewhere. Here's the output:
``` 
[]
Loading data (with cache)...
Done loading data.
Data loaded in 344.77 seconds
Pre-training agents...
Epoch 1/5
78/78 ━━━━━━━━━━━━━━━━━━━━ 388s 338ms/step - accuracy: 0.3383 - loss: 2.6270  
Epoch 2/5
78/78 ━━━━━━━━━━━━━━━━━━━━ 26s 337ms/step - accuracy: 0.7436 - loss: 0.9275
Epoch 3/5
78/78 ━━━━━━━━━━━━━━━━━━━━ 25s 316ms/step - accuracy: 0.8103 - loss: 0.6387
Epoch 4/5
78/78 ━━━━━━━━━━━━━━━━━━━━ 24s 303ms/step - accuracy: 0.8579 - loss: 0.4873
Epoch 5/5
78/78 ━━━━━━━━━━━━━━━━━━━━ 24s 305ms/step - accuracy: 0.8958 - loss: 0.3693
Pre-training complete.
Validating pre-trained agents...
Agent 0 validation loss: 0.9069
...
Round 2400: Loss: 0.0293, Correct: 75.43%
```

By overcoming overfitting for vision agent, leaving audio agent random, and leverage that with hard-coded hedge weights that always trust vision more, there's accuracy improvement with training data. Although, vision always produce accurate result, the fusion accuracy is at quite low 75%, which is great improvement from random 4% but it is still affected by audio agent, which is expected. But the level of influence has no quantifiable way to explain. That is, fixing hedge weight at 0.95 vs 0.05 will always leave some level of trust towards audio agent, an amount that cannot explained by the difference between fully trusting vision agent, which accuracy is roughly 0.9 but strictly speaking unknown, and 95% trusting. This is still area to dig deeper.
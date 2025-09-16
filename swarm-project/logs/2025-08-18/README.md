## 08-18
### new split
[]
Pre-training agents...
Pre-training agent 0...
Epoch 1/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 16s 299ms/step - accuracy: 0.1717 - loss: 3.0395
Epoch 2/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 295ms/step - accuracy: 0.6335 - loss: 1.6233
31/31 ━━━━━━━━━━━━━━━━━━━━ 10s 296ms/step - accuracy: 0.6338 - loss: 1.6162
Epoch 3/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 10s 309ms/step - accuracy: 0.7858 - loss: 0.9175
Epoch 4/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 297ms/step - accuracy: 0.8677 - loss: 0.6033
31/31 ━━━━━━━━━━━━━━━━━━━━ 10s 298ms/step - accuracy: 0.8670 - loss: 0.6020
Epoch 5/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 10s 302ms/step - accuracy: 0.9257 - loss: 0.4079
Epoch 6/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 10s 303ms/step - accuracy: 0.9557 - loss: 0.2890
Epoch 7/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 10s 307ms/step - accuracy: 0.9654 - loss: 0.2247
Epoch 8/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 323ms/step - accuracy: 0.9731 - loss: 0.1764
31/31 ━━━━━━━━━━━━━━━━━━━━ 11s 324ms/step - accuracy: 0.9723 - loss: 0.1759
Epoch 9/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 11s 319ms/step - accuracy: 0.9805 - loss: 0.1276
Epoch 10/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 10s 304ms/step - accuracy: 0.9895 - loss: 0.1112
Saved pre-trained model for agent 0 to models/pretrained_agent_0.keras
Pre-training agent 1...
Epoch 1/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 52s 16ms/step - accuracy: 0.1589 - loss: 3.1078 - learning_rate: 0.0010
Epoch 2/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.5080 - loss: 1.9315 - learning_rate: 0.0010
Epoch 3/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - accuracy: 0.6583 - loss: 1.3483 - learning_rate: 0.0010
Epoch 4/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 24ms/step - accuracy: 0.6787 - loss: 1.1944 - learning_rate: 0.0010
Epoch 5/10
30/31 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - accuracy: 0.7338 - loss: 0.9712
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 22ms/step - accuracy: 0.7328 - loss: 0.9742 - learning_rate: 0.0010
Epoch 6/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 29ms/step - accuracy: 0.7356 - loss: 0.9472 - learning_rate: 0.0010
Epoch 7/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 23ms/step - accuracy: 0.7692 - loss: 0.8480 - learning_rate: 0.0010
Epoch 8/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 26ms/step - accuracy: 0.8072 - loss: 0.7388 - learning_rate: 0.0010
Epoch 9/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 26ms/step - accuracy: 0.8186 - loss: 0.6782 - learning_rate: 0.0010
Epoch 10/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 27ms/step - accuracy: 0.8186 - loss: 0.6430 - learning_rate: 0.0010
Saved pre-trained model for agent 1 to models/pretrained_agent_1.keras
Pre-training complete.
Validating pre-trained agents...
7/7 ━━━━━━━━━━━━━━━━━━━━ 4s 332ms/step - accuracy: 0.6918 - loss: 0.8799
Agent 0 validation loss: 0.7878
7/7 ━━━━━━━━━━━━━━━━━━━━ 13s 2s/step - accuracy: 0.7297 - loss: 0.8916
Agent 1 validation loss: 0.9382
Continue training? (y/n): y
Starting training...
[Round 0] GT label = 7
  Agent 0 pred=7 correct=True hedge=0.0357
  Agent 1 pred=7 correct=True hedge=0.0357
[Round 1] GT label = 6
  Agent 0 pred=6 correct=True hedge=0.4988
  Agent 1 pred=6 correct=True hedge=0.5012
[Round 2] GT label = 11
  Agent 0 pred=11 correct=True hedge=0.5085
  Agent 1 pred=15 correct=False hedge=0.4915
[EVAL:val@500] n=200 | a0=0.705  a1=0.705  b0=0.725  m1=0.795
[EVAL:val@1000] n=200 | a0=0.705  a1=0.705  b0=0.705  m1=0.815
[EVAL:val_final] n=829 | a0=0.713  a1=0.697  b0=0.697  m1=0.840
[EVAL:test_final] n=829 | a0=0.688  a1=0.690  b0=0.691  m1=0.819
(fine-tuning-env) (base) danielhuang@Daniels-MacBook-Pro-2 swarm-project % python python_sims/metric/plotting.py
{'Agent0': np.float64(0.6967015285599356), 'Agent1': np.float64(0.7135961383748994), 'B0': np.float64(0.7562349155269509), 'M1': np.float64(0.7851971037811746)}
McNemar: pvalue      0.0030960205499590185
statistic   8.75
  grp = df.groupby("cos_bin")[["acc_B0","acc_M1"]].mean()
                       acc_B0    acc_M1
cos_bin                                
(0.000745, 0.06301]  0.448000  0.440000
(0.06301, 0.1777]    0.411290  0.540323
(0.1777, 0.3151]     0.524194  0.556452
(0.3151, 0.567]      0.588710  0.685484
(0.567, 0.7865]      0.768000  0.800000
(0.7865, 0.9195]     0.927419  0.927419
(0.9195, 0.9807]     0.935484  0.943548
(0.9807, 0.997]      0.975806  0.975806
(0.997, 0.9998]      0.983871  0.983871
(0.9998, 1.0]        1.000000  1.000000

### Additional metrics: 
[]
Pre-training agents...
Loaded pre-trained model for agent 0 from disk.
Loaded pre-trained model for agent 1 from disk.
Pre-training complete.
Validating pre-trained agents...
7/7 ━━━━━━━━━━━━━━━━━━━━ 5s 307ms/step - accuracy: 0.6918 - loss: 0.8799
Agent 0 validation loss: 0.7878
7/7 ━━━━━━━━━━━━━━━━━━━━ 13s 2s/step - accuracy: 0.7297 - loss: 0.8916
Agent 1 validation loss: 0.9382
Continue training? (y/n): y
Starting training...
[Round 0] GT label = 18
  Agent 0 pred=18 correct=True hedge=0.5000
  Agent 1 pred=25 correct=False hedge=0.5000
[Round 1] GT label = 2
  Agent 0 pred=2 correct=True hedge=0.5202
  Agent 1 pred=2 correct=True hedge=0.4798
[Round 2] GT label = 22
  Agent 0 pred=22 correct=True hedge=0.5244
  Agent 1 pred=22 correct=True hedge=0.4756
[EVAL:val@500::agent0] micro-acc=0.705  macro-F1=0.694
[EVAL:val@500::agent0] top confusions: [(27, 17, 4), (14, 27, 3), (6, 3, 3), (17, 14, 2), (9, 27, 2)]
[EVAL:val@500::agent1] micro-acc=0.705  macro-F1=0.684
[EVAL:val@500::agent1] top confusions: [(3, 11, 3), (9, 14, 3), (20, 11, 2), (17, 27, 2), (10, 24, 2)]
[EVAL:val@500::B0] micro-acc=0.815  macro-F1=0.797
[EVAL:val@500::B0] top confusions: [(9, 14, 4), (2, 22, 2), (20, 11, 2), (17, 27, 2), (6, 3, 2)]
[EVAL:val@500::M1] micro-acc=0.795  macro-F1=0.776
[EVAL:val@500::M1] top confusions: [(9, 14, 4), (27, 17, 3), (6, 3, 3), (20, 13, 1), (23, 3, 1)]
[EVAL:val@500] Top-5 per-class gains (acc, f1): [(15, 0.4285714285713673, 0.20606060606057897), (21, 0.09999999999998999, 0.10526315789472573), (4, 0.07142857142856629, 0.03968253968253854), (13, 0.0, -0.14285714285708861), (14, 0.0, 0.0)]
[EVAL:val@500] n=200 | a0=0.705  a1=0.705  b0=0.815  m1=0.795
2025-08-18 15:43:50.209051: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
[EVAL:val@1000::agent0] micro-acc=0.705  macro-F1=0.694
[EVAL:val@1000::agent0] top confusions: [(27, 17, 4), (14, 27, 3), (6, 3, 3), (17, 14, 2), (9, 27, 2)]
[EVAL:val@1000::agent1] micro-acc=0.705  macro-F1=0.684
[EVAL:val@1000::agent1] top confusions: [(3, 11, 3), (9, 14, 3), (20, 11, 2), (17, 27, 2), (10, 24, 2)]
[EVAL:val@1000::B0] micro-acc=0.705  macro-F1=0.684
[EVAL:val@1000::B0] top confusions: [(3, 11, 3), (9, 14, 3), (20, 11, 2), (17, 27, 2), (10, 24, 2)]
[EVAL:val@1000::M1] micro-acc=0.810  macro-F1=0.782
[EVAL:val@1000::M1] top confusions: [(9, 14, 4), (17, 27, 2), (20, 11, 2), (6, 3, 2), (20, 13, 1)]
[EVAL:val@1000] Top-5 per-class gains (acc, f1): [(3, 0.5714285714284897, 0.39999999999989555), (24, 0.4285714285713673, 0.3179487179486875), (4, 0.35714285714283156, 0.2329192546583555), (10, 0.2499999999999687, 0.29411764705877974), (16, 0.22222222222219756, 0.16959064327484197)]
[EVAL:val@1000] n=200 | a0=0.705  a1=0.705  b0=0.705  m1=0.810
[EVAL:val_final::agent0] micro-acc=0.713  macro-F1=0.713
[EVAL:val_final::agent0] top confusions: [(1, 5, 9), (14, 27, 9), (9, 27, 9), (6, 3, 8), (27, 17, 8)]
[EVAL:val_final::agent1] micro-acc=0.697  macro-F1=0.667
[EVAL:val_final::agent1] top confusions: [(17, 27, 12), (24, 10, 9), (3, 11, 7), (5, 1, 7), (3, 6, 6)]
[EVAL:val_final::B0] micro-acc=0.697  macro-F1=0.667
[EVAL:val_final::B0] top confusions: [(17, 27, 12), (24, 10, 9), (3, 11, 7), (5, 1, 7), (3, 6, 6)]
[EVAL:val_final::M1] micro-acc=0.805  macro-F1=0.779
[EVAL:val_final::M1] top confusions: [(6, 3, 7), (5, 1, 7), (9, 14, 6), (12, 8, 4), (18, 2, 4)]
[EVAL:val_final] Top-5 per-class gains (acc, f1): [(3, 0.40540540540539444, 0.27368421052626846), (24, 0.33333333333332404, 0.16153846153846096), (13, 0.24999999999998757, 0.19367588932806035), (17, 0.24324324324323665, 0.15539906103283652), (25, 0.22727272727271697, 0.27999999999997816)]
[EVAL:val_final] n=829 | a0=0.713  a1=0.697  b0=0.697  m1=0.805
[EVAL:test_final::agent0] micro-acc=0.688  macro-F1=0.679
[EVAL:test_final::agent0] top confusions: [(6, 3, 10), (1, 5, 9), (11, 4, 7), (14, 27, 7), (9, 27, 6)]
[EVAL:test_final::agent1] micro-acc=0.690  macro-F1=0.650
[EVAL:test_final::agent1] top confusions: [(6, 3, 10), (3, 6, 10), (17, 27, 9), (20, 11, 7), (6, 11, 7)]
[EVAL:test_final::B0] micro-acc=0.691  macro-F1=0.651
[EVAL:test_final::B0] top confusions: [(6, 3, 10), (3, 6, 10), (17, 27, 9), (20, 11, 7), (6, 11, 7)]
[EVAL:test_final::M1] micro-acc=0.791  macro-F1=0.759
[EVAL:test_final::M1] top confusions: [(6, 3, 11), (20, 11, 7), (3, 6, 7), (17, 27, 6), (13, 4, 5)]
[EVAL:test_final] Top-5 per-class gains (acc, f1): [(13, 0.3999999999999801, 0.2880523731587573), (3, 0.3243243243243155, 0.22996515679442425), (24, 0.27777777777777, 0.16347687400318567), (25, 0.2727272727272604, 0.3199999999999766), (19, 0.24999999999998446, 0.07407407407406996)]
[EVAL:test_final] n=829 | a0=0.688  a1=0.690  b0=0.691  m1=0.791
(fine-tuning-env) (base) danielhuang@Daniels-MacBook-Pro-2 swarm-project % python3 python_sims/metric/plotting.py
{'Agent0': np.float64(0.6875753920386007), 'Agent1': np.float64(0.6899879372738239), 'B0': np.float64(0.6911942098914354), 'M1': np.float64(0.7913148371531966)}
McNemar: pvalue      5.6078418266289875e-14
statistic   56.50420168067227
  grp = df.groupby("cos_bin")[["acc_B0","acc_M1"]].mean()
                        acc_B0    acc_M1
cos_bin                                 
(0.0003844, 0.07219]  0.349398  0.445783
(0.07219, 0.1724]     0.385542  0.554217
(0.1724, 0.3206]      0.397590  0.650602
(0.3206, 0.556]       0.457831  0.734940
(0.556, 0.7286]       0.590361  0.843373
(0.7286, 0.8891]      0.865854  0.817073
(0.8891, 0.9758]      0.891566  0.891566
(0.9758, 0.9965]      0.975904  0.975904
(0.9965, 0.9999]      1.000000  1.000000
(0.9999, 1.0]         1.000000  1.000000
Final cum NLL: {'_cum_b0': 795.7053007633231, '_cum_m1': 660.966834425238}
ECE(M1): 0.10198587539579959

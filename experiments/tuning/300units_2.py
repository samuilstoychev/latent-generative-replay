RAM AT BEGINNING: 0.22404861450195312
Latent replay turned on
CUDA is used
RAM BEFORE LOADING DATA: 0.228546142578125

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.2895088195800781
RAM BEFORE CLASSIFER: 2.242687225341797
RAM AFTER CLASSIFER: 2.242687225341797
RAM BEFORE PRE-TRAINING 2.242687225341797
RAM AFTER PRE-TRAINING 2.2572975158691406
RAM BEFORE GENERATOR: 2.2572975158691406
RAM AFTER DECLARING GENERATOR: 2.2572975158691406
MACs of root classifier 772000
MACs of top classifier: 39680
RAM BEFORE REPORTING: 2.2579116821289062

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([300, 300, 300])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([300, 300, 300])--z100-c10)-s5445

----------------------------------------TOP----------------------------------------
CNNTopClassifier(
  (dropout2): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=300, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 39818 parameters (~0.0 million)
      of which: - learnable: 39818 (~0.0 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------ROOT----------------------------------------
CNNRootClassifier(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 10, kernel_size=(5, 5), stride=(1, 1))
  (dropout1): Dropout(p=0.25, inplace=False)
  (fc0): Linear(in_features=1440, out_features=300, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 435070 parameters (~0.4 million)
      of which: - learnable: 435070 (~0.4 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------GENERATOR----------------------------------------
AutoEncoderLatent(
  (fcE): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=300, out_features=300)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=300, out_features=300)
      (nl): ReLU()
    )
  )
  (toZ): fc_layer_split(
    (mean): fc_layer(
      (linear): LinearExcitability(in_features=300, out_features=100)
    )
    (logvar): fc_layer(
      (linear): LinearExcitability(in_features=300, out_features=100)
    )
  )
  (classifier): fc_layer(
    (linear): LinearExcitability(in_features=300, out_features=10)
  )
  (fromZ): fc_layer(
    (linear): LinearExcitability(in_features=100, out_features=300)
    (nl): ReLU()
  )
  (fcD): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=300, out_features=300)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=300, out_features=300)
      (nl): Sigmoid()
    )
  )
)
------------------------------------------------------------------------------------------
--> this network has 454610 parameters (~0.5 million)
      of which: - learnable: 454610 (~0.5 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------
RAM BEFORE TRAINING: 2.2579116821289062
CPU BEFORE TRAINING: (27.55, 2.77)
INITIALISING GPU TRACKER

Training...
PEAK TRAINING RAM: 2.2601470947265625
Peak mem and init mem: 979 953
GPU BEFORE EVALUATION: (11.428571428571429, 26)
RAM BEFORE EVALUATION: 2.2601470947265625
CPU BEFORE EVALUATION: (120.13, 5.4)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 1.0000
 - Task 2: 0.9995
 - Task 3: 0.9968
 - Task 4: 0.9950
 - Task 5: 0.9894
=> Average precision over all 5 tasks: 0.9961

=> Total training time = 65.6 seconds

RAM AT THE END: 2.260150909423828
CPU AT THE END: (121.94, 5.4)

RAM AT BEGINNING: 0.2232818603515625
Latent replay turned on
CUDA is used
RAM BEFORE LOADING DATA: 0.22785186767578125

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.2888526916503906
RAM BEFORE CLASSIFER: 2.2380828857421875
RAM AFTER CLASSIFER: 2.2380828857421875
RAM BEFORE PRE-TRAINING 2.2380828857421875
RAM AFTER PRE-TRAINING 2.2539138793945312
RAM BEFORE GENERATOR: 2.2539138793945312
RAM AFTER DECLARING GENERATOR: 2.2539138793945312
MACs of root classifier 412000
MACs of top classifier: 7680
RAM BEFORE REPORTING: 2.2539138793945312

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([50, 50, 50])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([50, 50, 50])--z100-c10)-s3949

----------------------------------------TOP----------------------------------------
CNNTopClassifier(
  (dropout2): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=50, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 7818 parameters (~0.0 million)
      of which: - learnable: 7818 (~0.0 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------ROOT----------------------------------------
CNNRootClassifier(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 10, kernel_size=(5, 5), stride=(1, 1))
  (dropout1): Dropout(p=0.25, inplace=False)
  (fc0): Linear(in_features=1440, out_features=50, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 74820 parameters (~0.1 million)
      of which: - learnable: 74820 (~0.1 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------GENERATOR----------------------------------------
AutoEncoderLatent(
  (fcE): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=50, out_features=50)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=50, out_features=50)
      (nl): ReLU()
    )
  )
  (toZ): fc_layer_split(
    (mean): fc_layer(
      (linear): LinearExcitability(in_features=50, out_features=100)
    )
    (logvar): fc_layer(
      (linear): LinearExcitability(in_features=50, out_features=100)
    )
  )
  (classifier): fc_layer(
    (linear): LinearExcitability(in_features=50, out_features=10)
  )
  (fromZ): fc_layer(
    (linear): LinearExcitability(in_features=100, out_features=50)
    (nl): ReLU()
  )
  (fcD): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=50, out_features=50)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=50, out_features=50)
      (nl): Sigmoid()
    )
  )
)
------------------------------------------------------------------------------------------
--> this network has 25860 parameters (~0.0 million)
      of which: - learnable: 25860 (~0.0 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------
RAM BEFORE TRAINING: 2.2539138793945312
CPU BEFORE TRAINING: (27.29, 2.54)
INITIALISING GPU TRACKER

Training...
PEAK TRAINING RAM: 2.257434844970703
Peak mem and init mem: 965 953
GPU BEFORE EVALUATION: (10.571428571428571, 12)
RAM BEFORE EVALUATION: 2.257434844970703
CPU BEFORE EVALUATION: (122.64, 5.06)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.9943
 - Task 2: 1.0000
 - Task 3: 0.9961
 - Task 4: 0.9947
 - Task 5: 0.9985
=> Average precision over all 5 tasks: 0.9967

=> Total training time = 68.0 seconds

RAM AT THE END: 2.257434844970703
CPU AT THE END: (124.47, 5.07)

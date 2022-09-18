RAM AT BEGINNING: 0.22371673583984375
Latent replay turned on
CUDA is used
RAM BEFORE LOADING DATA: 0.22826385498046875

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.2893638610839844
RAM BEFORE CLASSIFER: 2.241443634033203
RAM AFTER CLASSIFER: 2.241443634033203
RAM BEFORE PRE-TRAINING 2.241443634033203
RAM AFTER PRE-TRAINING 2.2579879760742188
RAM BEFORE GENERATOR: 2.2579879760742188
RAM AFTER DECLARING GENERATOR: 2.2579879760742188
MACs of root classifier 628000
MACs of top classifier: 26880
RAM BEFORE REPORTING: 2.2579879760742188

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([200, 200, 200])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([200, 200, 200])--z100-c10)-s18758

----------------------------------------TOP----------------------------------------
CNNTopClassifier(
  (dropout2): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=200, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 27018 parameters (~0.0 million)
      of which: - learnable: 27018 (~0.0 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------ROOT----------------------------------------
CNNRootClassifier(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 10, kernel_size=(5, 5), stride=(1, 1))
  (dropout1): Dropout(p=0.25, inplace=False)
  (fc0): Linear(in_features=1440, out_features=200, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 290970 parameters (~0.3 million)
      of which: - learnable: 290970 (~0.3 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------GENERATOR----------------------------------------
AutoEncoderLatent(
  (fcE): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=200, out_features=200)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=200, out_features=200)
      (nl): ReLU()
    )
  )
  (toZ): fc_layer_split(
    (mean): fc_layer(
      (linear): LinearExcitability(in_features=200, out_features=100)
    )
    (logvar): fc_layer(
      (linear): LinearExcitability(in_features=200, out_features=100)
    )
  )
  (classifier): fc_layer(
    (linear): LinearExcitability(in_features=200, out_features=10)
  )
  (fromZ): fc_layer(
    (linear): LinearExcitability(in_features=100, out_features=200)
    (nl): ReLU()
  )
  (fcD): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=200, out_features=200)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=200, out_features=200)
      (nl): Sigmoid()
    )
  )
)
------------------------------------------------------------------------------------------
--> this network has 223110 parameters (~0.2 million)
      of which: - learnable: 223110 (~0.2 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------
RAM BEFORE TRAINING: 2.2579879760742188
CPU BEFORE TRAINING: (30.57, 2.64)
INITIALISING GPU TRACKER

Training...
PEAK TRAINING RAM: 2.2603836059570312
Peak mem and init mem: 969 951
GPU BEFORE EVALUATION: (10.285714285714286, 18)
RAM BEFORE EVALUATION: 2.2603836059570312
CPU BEFORE EVALUATION: (125.41, 5.3)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.9985
 - Task 2: 0.9980
 - Task 3: 0.9972
 - Task 4: 0.9952
 - Task 5: 0.9931
=> Average precision over all 5 tasks: 0.9964

=> Total training time = 67.9 seconds

RAM AT THE END: 2.2603836059570312
CPU AT THE END: (127.31, 5.31)

RAM AT BEGINNING: 0.22328948974609375
Latent replay turned on
CUDA is used
RAM BEFORE LOADING DATA: 0.2279205322265625

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.2888832092285156
RAM BEFORE CLASSIFER: 2.2372283935546875
RAM AFTER CLASSIFER: 2.238208770751953
RAM BEFORE PRE-TRAINING 2.238208770751953
RAM AFTER PRE-TRAINING 2.2537994384765625
RAM BEFORE GENERATOR: 2.2537994384765625
RAM AFTER DECLARING GENERATOR: 2.2537994384765625
MACs of root classifier 412000
MACs of top classifier: 7680
RAM BEFORE REPORTING: 2.2537994384765625

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([50, 50, 50])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([50, 50, 50])--z100-c10)-s13544

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
RAM BEFORE TRAINING: 2.2537994384765625
CPU BEFORE TRAINING: (27.66, 2.64)
INITIALISING GPU TRACKER

Training...
PEAK TRAINING RAM: 2.257415771484375
Peak mem and init mem: 965 953
GPU BEFORE EVALUATION: (10.857142857142858, 12)
RAM BEFORE EVALUATION: 2.257415771484375
CPU BEFORE EVALUATION: (116.92, 4.63)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.9954
 - Task 2: 0.9986
 - Task 3: 0.9936
 - Task 4: 0.9919
 - Task 5: 0.9830
=> Average precision over all 5 tasks: 0.9925

=> Total training time = 60.9 seconds

RAM AT THE END: 2.257476806640625
CPU AT THE END: (118.72, 4.65)

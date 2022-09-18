RAM AT BEGINNING: 0.2234954833984375
Latent replay turned on
CUDA is used
RAM BEFORE LOADING DATA: 0.22805023193359375

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.28932952880859375
RAM BEFORE CLASSIFER: 2.239727020263672
RAM AFTER CLASSIFER: 2.239727020263672
RAM BEFORE PRE-TRAINING 2.239727020263672
RAM AFTER PRE-TRAINING 2.255390167236328
RAM BEFORE GENERATOR: 2.255390167236328
RAM AFTER DECLARING GENERATOR: 2.255390167236328
MACs of root classifier 484000
MACs of top classifier: 14080
RAM BEFORE REPORTING: 2.255390167236328

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([100, 100, 100])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([100, 100, 100])--z100-c10)-s29590

----------------------------------------TOP----------------------------------------
CNNTopClassifier(
  (dropout2): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=100, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 14218 parameters (~0.0 million)
      of which: - learnable: 14218 (~0.0 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------ROOT----------------------------------------
CNNRootClassifier(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 10, kernel_size=(5, 5), stride=(1, 1))
  (dropout1): Dropout(p=0.25, inplace=False)
  (fc0): Linear(in_features=1440, out_features=100, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 146870 parameters (~0.1 million)
      of which: - learnable: 146870 (~0.1 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------GENERATOR----------------------------------------
AutoEncoderLatent(
  (fcE): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=100, out_features=100)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=100, out_features=100)
      (nl): ReLU()
    )
  )
  (toZ): fc_layer_split(
    (mean): fc_layer(
      (linear): LinearExcitability(in_features=100, out_features=100)
    )
    (logvar): fc_layer(
      (linear): LinearExcitability(in_features=100, out_features=100)
    )
  )
  (classifier): fc_layer(
    (linear): LinearExcitability(in_features=100, out_features=10)
  )
  (fromZ): fc_layer(
    (linear): LinearExcitability(in_features=100, out_features=100)
    (nl): ReLU()
  )
  (fcD): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=100, out_features=100)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=100, out_features=100)
      (nl): Sigmoid()
    )
  )
)
------------------------------------------------------------------------------------------
--> this network has 71610 parameters (~0.1 million)
      of which: - learnable: 71610 (~0.1 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------
RAM BEFORE TRAINING: 2.255390167236328
CPU BEFORE TRAINING: (28.19, 2.64)
INITIALISING GPU TRACKER

Training...
PEAK TRAINING RAM: 2.2586288452148438
Peak mem and init mem: 967 955
GPU BEFORE EVALUATION: (10.714285714285714, 12)
RAM BEFORE EVALUATION: 2.2586288452148438
CPU BEFORE EVALUATION: (118.72, 4.93)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.9980
 - Task 2: 0.9949
 - Task 3: 0.9925
 - Task 4: 0.9879
 - Task 5: 0.9970
=> Average precision over all 5 tasks: 0.9941

=> Total training time = 62.5 seconds

RAM AT THE END: 2.258636474609375
CPU AT THE END: (120.49, 4.96)

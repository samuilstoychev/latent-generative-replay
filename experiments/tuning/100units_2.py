RAM AT BEGINNING: 0.22267532348632812
Latent replay turned on
CUDA is used
RAM BEFORE LOADING DATA: 0.227325439453125

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.2882270812988281
RAM BEFORE CLASSIFER: 2.2365074157714844
RAM AFTER CLASSIFER: 2.2372283935546875
RAM BEFORE PRE-TRAINING 2.2372283935546875
RAM AFTER PRE-TRAINING 2.252735137939453
RAM BEFORE GENERATOR: 2.252735137939453
RAM AFTER DECLARING GENERATOR: 2.252735137939453
MACs of root classifier 484000
MACs of top classifier: 14080
RAM BEFORE REPORTING: 2.252735137939453

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([100, 100, 100])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([100, 100, 100])--z100-c10)-s13714

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
RAM BEFORE TRAINING: 2.252735137939453
CPU BEFORE TRAINING: (28.23, 2.62)
INITIALISING GPU TRACKER

Training...
PEAK TRAINING RAM: 2.256122589111328
Peak mem and init mem: 967 955
GPU BEFORE EVALUATION: (9.75, 12)
RAM BEFORE EVALUATION: 2.256122589111328
CPU BEFORE EVALUATION: (127.86, 5.94)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.9990
 - Task 2: 0.9986
 - Task 3: 0.9895
 - Task 4: 0.9914
 - Task 5: 0.9912
=> Average precision over all 5 tasks: 0.9939

=> Total training time = 70.1 seconds

RAM AT THE END: 2.2562408447265625
CPU AT THE END: (129.74, 5.97)

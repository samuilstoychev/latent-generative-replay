RAM AT BEGINNING: 0.22316360473632812
Latent replay turned on
CUDA is used
RAM BEFORE LOADING DATA: 0.22772979736328125

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.28879547119140625
RAM BEFORE CLASSIFER: 2.2383460998535156
RAM AFTER CLASSIFER: 2.2397384643554688
RAM BEFORE PRE-TRAINING 2.2397384643554688
RAM AFTER PRE-TRAINING 2.254749298095703
RAM BEFORE GENERATOR: 2.254749298095703
RAM AFTER DECLARING GENERATOR: 2.254749298095703
MACs of root classifier 628000
MACs of top classifier: 26880
RAM BEFORE REPORTING: 2.254749298095703

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([200, 200, 200])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([200, 200, 200])--z100-c10)-s5694

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
RAM BEFORE TRAINING: 2.254749298095703
CPU BEFORE TRAINING: (29.7, 2.92)
INITIALISING GPU TRACKER

Training...
PEAK TRAINING RAM: 2.257476806640625
Peak mem and init mem: 969 951
GPU BEFORE EVALUATION: (10.142857142857142, 18)
RAM BEFORE EVALUATION: 2.257476806640625
CPU BEFORE EVALUATION: (123.02, 5.61)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.9883
 - Task 2: 0.9869
 - Task 3: 0.9907
 - Task 4: 0.9986
 - Task 5: 0.9995
=> Average precision over all 5 tasks: 0.9928

=> Total training time = 67.1 seconds

RAM AT THE END: 2.2575950622558594
CPU AT THE END: (124.9, 5.62)

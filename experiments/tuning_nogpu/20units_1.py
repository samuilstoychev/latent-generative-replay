RAM AT BEGINNING: 0.22374343872070312
Latent replay turned on
CUDA is NOT(!!) used
RAM BEFORE LOADING DATA: 0.2285308837890625

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.2894783020019531
RAM BEFORE CLASSIFER: 0.29033660888671875
RAM AFTER CLASSIFER: 0.29033660888671875
RAM BEFORE PRE-TRAINING 0.29033660888671875
RAM AFTER PRE-TRAINING 0.3155326843261719
RAM BEFORE GENERATOR: 0.3155326843261719
RAM AFTER DECLARING GENERATOR: 0.3155326843261719
MACs of root classifier 368800
MACs of top classifier: 3840
RAM BEFORE REPORTING: 0.3155326843261719

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([20, 20, 20])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([20, 20, 20])--z100-c10)-s24849

----------------------------------------TOP----------------------------------------
CNNTopClassifier(
  (dropout2): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=20, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 3978 parameters (~0.0 million)
      of which: - learnable: 3978 (~0.0 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------ROOT----------------------------------------
CNNRootClassifier(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 10, kernel_size=(5, 5), stride=(1, 1))
  (dropout1): Dropout(p=0.25, inplace=False)
  (fc0): Linear(in_features=1440, out_features=20, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 31590 parameters (~0.0 million)
      of which: - learnable: 31590 (~0.0 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------GENERATOR----------------------------------------
AutoEncoderLatent(
  (fcE): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=20, out_features=20)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=20, out_features=20)
      (nl): ReLU()
    )
  )
  (toZ): fc_layer_split(
    (mean): fc_layer(
      (linear): LinearExcitability(in_features=20, out_features=100)
    )
    (logvar): fc_layer(
      (linear): LinearExcitability(in_features=20, out_features=100)
    )
  )
  (classifier): fc_layer(
    (linear): LinearExcitability(in_features=20, out_features=10)
  )
  (fromZ): fc_layer(
    (linear): LinearExcitability(in_features=100, out_features=20)
    (nl): ReLU()
  )
  (fcD): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=20, out_features=20)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=20, out_features=20)
      (nl): Sigmoid()
    )
  )
)
------------------------------------------------------------------------------------------
--> this network has 8010 parameters (~0.0 million)
      of which: - learnable: 8010 (~0.0 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------
RAM BEFORE TRAINING: 0.3155326843261719
CPU BEFORE TRAINING: (137.08, 2.84)

Training...
PEAK TRAINING RAM: 0.342071533203125
RAM BEFORE EVALUATION: 0.32404327392578125
CPU BEFORE EVALUATION: (887.1, 69.37)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.9859
 - Task 2: 0.9821
 - Task 3: 0.9930
 - Task 4: 0.9926
 - Task 5: 0.9919
=> Average precision over all 5 tasks: 0.9891

=> Total training time = 153.5 seconds

RAM AT THE END: 0.32605743408203125
CPU AT THE END: (892.53, 71.15)

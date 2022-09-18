RAM AT BEGINNING: 0.22320938110351562
Latent replay turned on
CUDA is NOT(!!) used
RAM BEFORE LOADING DATA: 0.22769546508789062

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.2888946533203125
RAM BEFORE CLASSIFER: 0.2902107238769531
RAM AFTER CLASSIFER: 0.2911834716796875
RAM BEFORE PRE-TRAINING 0.2911834716796875
RAM AFTER PRE-TRAINING 0.31764984130859375
RAM BEFORE GENERATOR: 0.31764984130859375
RAM AFTER DECLARING GENERATOR: 0.31764984130859375
MACs of root classifier 524320
MACs of top classifier: 17664
RAM BEFORE REPORTING: 0.31764984130859375

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([128, 128, 128])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([128, 128, 128])--z100-c10)-s22068

----------------------------------------TOP----------------------------------------
CNNTopClassifier(
  (dropout2): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=128, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 17802 parameters (~0.0 million)
      of which: - learnable: 17802 (~0.0 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------ROOT----------------------------------------
CNNRootClassifier(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 10, kernel_size=(5, 5), stride=(1, 1))
  (dropout1): Dropout(p=0.25, inplace=False)
  (fc0): Linear(in_features=1440, out_features=128, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 187218 parameters (~0.2 million)
      of which: - learnable: 187218 (~0.2 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------GENERATOR----------------------------------------
AutoEncoderLatent(
  (fcE): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=128, out_features=128)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=128, out_features=128)
      (nl): ReLU()
    )
  )
  (toZ): fc_layer_split(
    (mean): fc_layer(
      (linear): LinearExcitability(in_features=128, out_features=100)
    )
    (logvar): fc_layer(
      (linear): LinearExcitability(in_features=128, out_features=100)
    )
  )
  (classifier): fc_layer(
    (linear): LinearExcitability(in_features=128, out_features=10)
  )
  (fromZ): fc_layer(
    (linear): LinearExcitability(in_features=100, out_features=128)
    (nl): ReLU()
  )
  (fcD): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=128, out_features=128)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=128, out_features=128)
      (nl): Sigmoid()
    )
  )
)
------------------------------------------------------------------------------------------
--> this network has 105966 parameters (~0.1 million)
      of which: - learnable: 105966 (~0.1 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------
RAM BEFORE TRAINING: 0.31764984130859375
CPU BEFORE TRAINING: (196.91, 3.78)

Training...
PEAK TRAINING RAM: 0.35449981689453125
RAM BEFORE EVALUATION: 0.33757781982421875
CPU BEFORE EVALUATION: (1473.13, 100.31)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.9990
 - Task 2: 0.9979
 - Task 3: 0.9985
 - Task 4: 0.9953
 - Task 5: 0.9888
=> Average precision over all 5 tasks: 0.9959

=> Total training time = 271.2 seconds

RAM AT THE END: 0.3314361572265625
CPU AT THE END: (1481.12, 101.62)

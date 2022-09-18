RAM AT BEGINNING: 0.2230224609375
Latent replay turned on
CUDA is NOT(!!) used
RAM BEFORE LOADING DATA: 0.22751235961914062

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.2884025573730469
RAM BEFORE CLASSIFER: 0.2897491455078125
RAM AFTER CLASSIFER: 0.290252685546875
RAM BEFORE PRE-TRAINING 0.290252685546875
RAM AFTER PRE-TRAINING 0.31789398193359375
RAM BEFORE GENERATOR: 0.31789398193359375
RAM AFTER DECLARING GENERATOR: 0.31789398193359375
MACs of root classifier 484000
MACs of top classifier: 14080
RAM BEFORE REPORTING: 0.31789398193359375

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([100, 100, 100])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([100, 100, 100])--z100-c10)-s6339

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
RAM BEFORE TRAINING: 0.31789398193359375
CPU BEFORE TRAINING: (232.02, 4.74)

Training...
PEAK TRAINING RAM: 0.3511238098144531
RAM BEFORE EVALUATION: 0.3387184143066406
CPU BEFORE EVALUATION: (1469.9, 89.25)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.9876
 - Task 2: 0.9923
 - Task 3: 0.9915
 - Task 4: 0.9986
 - Task 5: 0.9995
=> Average precision over all 5 tasks: 0.9939

=> Total training time = 243.6 seconds

RAM AT THE END: 0.357177734375
CPU AT THE END: (1475.52, 89.56)

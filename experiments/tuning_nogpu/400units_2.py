RAM AT BEGINNING: 0.22317123413085938
Latent replay turned on
CUDA is NOT(!!) used
RAM BEFORE LOADING DATA: 0.227874755859375

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.288970947265625
RAM BEFORE CLASSIFER: 0.2923316955566406
RAM AFTER CLASSIFER: 0.2945976257324219
RAM BEFORE PRE-TRAINING 0.2945976257324219
RAM AFTER PRE-TRAINING 0.3228263854980469
RAM BEFORE GENERATOR: 0.3228263854980469
RAM AFTER DECLARING GENERATOR: 0.3228263854980469
MACs of root classifier 916000
MACs of top classifier: 52480
RAM BEFORE REPORTING: 0.3235816955566406

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([400, 400, 400])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([400, 400, 400])--z100-c10)-s2841

----------------------------------------TOP----------------------------------------
CNNTopClassifier(
  (dropout2): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=400, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 52618 parameters (~0.1 million)
      of which: - learnable: 52618 (~0.1 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------ROOT----------------------------------------
CNNRootClassifier(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 10, kernel_size=(5, 5), stride=(1, 1))
  (dropout1): Dropout(p=0.25, inplace=False)
  (fc0): Linear(in_features=1440, out_features=400, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 579170 parameters (~0.6 million)
      of which: - learnable: 579170 (~0.6 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------GENERATOR----------------------------------------
AutoEncoderLatent(
  (fcE): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=400, out_features=400)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=400, out_features=400)
      (nl): ReLU()
    )
  )
  (toZ): fc_layer_split(
    (mean): fc_layer(
      (linear): LinearExcitability(in_features=400, out_features=100)
    )
    (logvar): fc_layer(
      (linear): LinearExcitability(in_features=400, out_features=100)
    )
  )
  (classifier): fc_layer(
    (linear): LinearExcitability(in_features=400, out_features=10)
  )
  (fromZ): fc_layer(
    (linear): LinearExcitability(in_features=100, out_features=400)
    (nl): ReLU()
  )
  (fcD): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=400, out_features=400)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=400, out_features=400)
      (nl): Sigmoid()
    )
  )
)
------------------------------------------------------------------------------------------
--> this network has 766110 parameters (~0.8 million)
      of which: - learnable: 766110 (~0.8 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------
RAM BEFORE TRAINING: 0.3235816955566406
CPU BEFORE TRAINING: (101.51, 11.51)

Training...
PEAK TRAINING RAM: 0.3898811340332031
RAM BEFORE EVALUATION: 0.37569427490234375
CPU BEFORE EVALUATION: (1902.07, 93.47)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.9974
 - Task 2: 0.9990
 - Task 3: 0.9960
 - Task 4: 0.9917
 - Task 5: 0.9990
=> Average precision over all 5 tasks: 0.9966

=> Total training time = 312.5 seconds

RAM AT THE END: 0.37286376953125
CPU AT THE END: (1907.64, 95.77)

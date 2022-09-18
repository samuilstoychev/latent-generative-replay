RAM AT BEGINNING: 0.22304153442382812
Latent replay turned on
CUDA is NOT(!!) used
RAM BEFORE LOADING DATA: 0.2276153564453125

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.2886466979980469
RAM BEFORE CLASSIFER: 0.2905082702636719
RAM AFTER CLASSIFER: 0.2917671203613281
RAM BEFORE PRE-TRAINING 0.2917671203613281
RAM AFTER PRE-TRAINING 0.3195228576660156
RAM BEFORE GENERATOR: 0.3195228576660156
RAM AFTER DECLARING GENERATOR: 0.3195228576660156
MACs of root classifier 628000
MACs of top classifier: 26880
RAM BEFORE REPORTING: 0.3195228576660156

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([200, 200, 200])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([200, 200, 200])--z100-c10)-s623

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
RAM BEFORE TRAINING: 0.3195228576660156
CPU BEFORE TRAINING: (141.69, 2.66)

Training...
PEAK TRAINING RAM: 0.37584686279296875
RAM BEFORE EVALUATION: 0.3556327819824219
CPU BEFORE EVALUATION: (1798.75, 87.88)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.9957
 - Task 2: 0.9917
 - Task 3: 0.9923
 - Task 4: 0.9892
 - Task 5: 0.9991
=> Average precision over all 5 tasks: 0.9936

=> Total training time = 298.0 seconds

RAM AT THE END: 0.3717498779296875
CPU AT THE END: (1804.69, 89.04)

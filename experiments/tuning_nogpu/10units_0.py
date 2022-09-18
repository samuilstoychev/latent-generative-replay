RAM AT BEGINNING: 0.22293853759765625
Latent replay turned on
CUDA is NOT(!!) used
RAM BEFORE LOADING DATA: 0.227508544921875

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.2885551452636719
RAM BEFORE CLASSIFER: 0.2885551452636719
RAM AFTER CLASSIFER: 0.2894134521484375
RAM BEFORE PRE-TRAINING 0.2894134521484375
RAM AFTER PRE-TRAINING 0.312713623046875
RAM BEFORE GENERATOR: 0.312713623046875
RAM AFTER DECLARING GENERATOR: 0.312713623046875
MACs of root classifier 354400
MACs of top classifier: 2560
RAM BEFORE REPORTING: 0.312713623046875

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([10, 10, 10])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([10, 10, 10])--z100-c10)-s26475

----------------------------------------TOP----------------------------------------
CNNTopClassifier(
  (dropout2): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=10, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 2698 parameters (~0.0 million)
      of which: - learnable: 2698 (~0.0 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------ROOT----------------------------------------
CNNRootClassifier(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 10, kernel_size=(5, 5), stride=(1, 1))
  (dropout1): Dropout(p=0.25, inplace=False)
  (fc0): Linear(in_features=1440, out_features=10, bias=True)
)
------------------------------------------------------------------------------------------
--> this network has 17180 parameters (~0.0 million)
      of which: - learnable: 17180 (~0.0 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------

----------------------------------------GENERATOR----------------------------------------
AutoEncoderLatent(
  (fcE): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=10, out_features=10)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=10, out_features=10)
      (nl): ReLU()
    )
  )
  (toZ): fc_layer_split(
    (mean): fc_layer(
      (linear): LinearExcitability(in_features=10, out_features=100)
    )
    (logvar): fc_layer(
      (linear): LinearExcitability(in_features=10, out_features=100)
    )
  )
  (classifier): fc_layer(
    (linear): LinearExcitability(in_features=10, out_features=10)
  )
  (fromZ): fc_layer(
    (linear): LinearExcitability(in_features=100, out_features=10)
    (nl): ReLU()
  )
  (fcD): MLP(
    (fcLayer1): fc_layer(
      (linear): LinearExcitability(in_features=10, out_features=10)
      (nl): ReLU()
    )
    (fcLayer2): fc_layer(
      (linear): LinearExcitability(in_features=10, out_features=10)
      (nl): Sigmoid()
    )
  )
)
------------------------------------------------------------------------------------------
--> this network has 3660 parameters (~0.0 million)
      of which: - learnable: 3660 (~0.0 million)
                - fixed: 0 (~0.0 million)
------------------------------------------------------------------------------------------
RAM BEFORE TRAINING: 0.312713623046875
CPU BEFORE TRAINING: (212.5, 3.53)

Training...
PEAK TRAINING RAM: 0.3369789123535156
RAM BEFORE EVALUATION: 0.3203239440917969
CPU BEFORE EVALUATION: (492.78, 56.54)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.5088
 - Task 2: 0.4909
 - Task 3: 0.8975
 - Task 4: 0.7951
 - Task 5: 1.0000
=> Average precision over all 5 tasks: 0.7385

=> Total training time = 78.1 seconds

RAM AT THE END: 0.3234825134277344
CPU AT THE END: (498.11, 58.21)

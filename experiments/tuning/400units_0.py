RAM AT BEGINNING: 0.22273635864257812
Latent replay turned on
CUDA is used
RAM BEFORE LOADING DATA: 0.22739791870117188

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.288299560546875
RAM BEFORE CLASSIFER: 2.2473487854003906
RAM AFTER CLASSIFER: 2.2473487854003906
RAM BEFORE PRE-TRAINING 2.2473487854003906
RAM AFTER PRE-TRAINING 2.2627487182617188
RAM BEFORE GENERATOR: 2.2627487182617188
RAM AFTER DECLARING GENERATOR: 2.2627487182617188
MACs of root classifier 916000
MACs of top classifier: 52480
RAM BEFORE REPORTING: 2.2627487182617188

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([400, 400, 400])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([400, 400, 400])--z100-c10)-s27722

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
RAM BEFORE TRAINING: 2.2627487182617188
CPU BEFORE TRAINING: (27.37, 2.93)
INITIALISING GPU TRACKER

Training...
PEAK TRAINING RAM: 2.2652359008789062
Peak mem and init mem: 999 955
GPU BEFORE EVALUATION: (13.0, 44)
RAM BEFORE EVALUATION: 2.2652359008789062
CPU BEFORE EVALUATION: (115.29, 5.02)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.9900
 - Task 2: 0.9991
 - Task 3: 0.9980
 - Task 4: 0.9954
 - Task 5: 0.9965
=> Average precision over all 5 tasks: 0.9958

=> Total training time = 59.9 seconds

RAM AT THE END: 2.2653541564941406
CPU AT THE END: (117.05, 5.04)

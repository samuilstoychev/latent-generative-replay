RAM AT BEGINNING: 0.22324752807617188
Latent replay turned on
CUDA is used
RAM BEFORE LOADING DATA: 0.22782516479492188

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.2888221740722656
RAM BEFORE CLASSIFER: 2.2377357482910156
RAM AFTER CLASSIFER: 2.2377357482910156
RAM BEFORE PRE-TRAINING 2.2377357482910156
RAM AFTER PRE-TRAINING 2.2539443969726562
RAM BEFORE GENERATOR: 2.2539443969726562
RAM AFTER DECLARING GENERATOR: 2.2539443969726562
MACs of root classifier 368800
MACs of top classifier: 3840
RAM BEFORE REPORTING: 2.2539443969726562

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([20, 20, 20])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([20, 20, 20])--z100-c10)-s13320

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
RAM BEFORE TRAINING: 2.2539443969726562
CPU BEFORE TRAINING: (28.25, 3.16)
INITIALISING GPU TRACKER

Training...
PEAK TRAINING RAM: 2.2575149536132812
Peak mem and init mem: 963 951
GPU BEFORE EVALUATION: (11.5, 12)
RAM BEFORE EVALUATION: 2.2575149536132812
CPU BEFORE EVALUATION: (116.41, 5.46)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.5032
 - Task 2: 0.9897
 - Task 3: 0.9742
 - Task 4: 0.9635
 - Task 5: 0.9950
=> Average precision over all 5 tasks: 0.8851

=> Total training time = 59.8 seconds

RAM AT THE END: 2.2576446533203125
CPU AT THE END: (118.17, 5.48)

RAM AT BEGINNING: 0.22303009033203125
Latent replay turned on
CUDA is NOT(!!) used
RAM BEFORE LOADING DATA: 0.22753143310546875

Preparing the data...
SPLIT RATIO: [50000, 10000]
 --> mnist: 'train'-dataset consisting of 60000 samples
 --> mnist: 'test'-dataset consisting of 10000 samples
RAM AFTER LOADING DATA: 0.2884941101074219
RAM BEFORE CLASSIFER: 0.29160308837890625
RAM AFTER CLASSIFER: 0.2938690185546875
RAM BEFORE PRE-TRAINING 0.2938690185546875
RAM AFTER PRE-TRAINING 0.32209014892578125
RAM BEFORE GENERATOR: 0.32209014892578125
RAM AFTER DECLARING GENERATOR: 0.32209014892578125
MACs of root classifier 916000
MACs of top classifier: 52480
RAM BEFORE REPORTING: 0.3233489990234375

Parameter-stamp...
 --> task:          splitMNIST5-task
 --> model:         CNN_CLASSIFIER_c10
 --> hyper-params:  i500-lr0.001-b128-adam
 --> replay:        generative-VAE(MLP([400, 400, 400])--z100-c10)
splitMNIST5-task--CNN_CLASSIFIER_c10--i500-lr0.001-b128-adam--generative-VAE(MLP([400, 400, 400])--z100-c10)-s3889

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
RAM BEFORE TRAINING: 0.3233489990234375
CPU BEFORE TRAINING: (101.34, 10.73)

Training...
PEAK TRAINING RAM: 0.3882293701171875
RAM BEFORE EVALUATION: 0.378173828125
CPU BEFORE EVALUATION: (1903.57, 77.66)


EVALUATION RESULTS:

 Precision on test-set:
 - Task 1: 0.9923
 - Task 2: 0.9936
 - Task 3: 0.9974
 - Task 4: 0.9949
 - Task 5: 0.9995
=> Average precision over all 5 tasks: 0.9955

=> Total training time = 310.6 seconds

RAM AT THE END: 0.3841972351074219
CPU AT THE END: (1908.92, 77.69)

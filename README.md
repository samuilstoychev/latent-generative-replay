# Latent Generative Replay for Resource-Efficient Continual Learning of Facial Expressions

This repository includes the code for the experiments presented in *"Latent Generative Replay for Resource-Efficient Continual Learning of Facial Expressions"* (Accepted at [FG 2023](https://fg2023.ieee-biometrics.org/))

![s](https://i.ibb.co/Kx12GGL/compared-approaches-alt-gen-2.jpg)

## Acknoledgments

The implementation has been adapted using the original Continual Learning repository by van de Ven et al. ([available on GitHub](https://github.com/GMvandeVen/continual-learning)). 

Model performance has been benchmarked adapting implementations from [the Avalanche project](https://avalanche.continualai.org/). 

## Files
The files in this repository are structured as follows: 

* `analysis` - includes Jupyter notebooks which summarise the experiment results and produce some of the tables and plots used in the paper. 
* `experiments` - includes the the scripts used to run the evaluation experiments as well as the logs with the results. 
* The facts in the root directory are structured similarly to the original Continual Learning repository. On top of that, we have added implementation for the latent generator (`autoencoder_latent.py`) and latent training (`train_latent.py`), added support for Naïve Rehearsal (`naive_rehearsal.py`) and VGG networks (`vgg_classifier.py`), and made ad-hoc changes to the other files where necessary. 

## Flags

We have also added a few flags that can be passed as options when running the `main.py` script. Those include: 
* `--latent-replay=(on|off)` - turns latent replay on or off. This flag is set to `on` for Latent Replay, Latent Generative Replay and Latent Generative Replay with Distillation. 
* `--network=(mlp|cnn)` - what type of classifier should be used? 
* `--latent-size` - the size of the latent replay layer (denoted as G<sub>OUT</sub> in the paper). 
* `--data-augmentation` - enable data augmentations. Augmentations include random horizontal flip and random rotation. 
* `--vgg-root` - use a pre-trained VGG-16 root/feature extractor. 
* `--buffer-size` - the size of the replay buffer for the rehearsal strategies. 
* `--early-stop` - enable early stopping. 
* `--validation` - validate performance after each task (i.e. obtain and log performance on the entire training dataset and the entire validation dataset). 

The other flags are documented in [the README file of van de Ven's continual learning repository](https://github.com/GMvandeVen/continual-learning). For example, to run Latent Generative Replay with CK+ on Task-IL (using the VGG-16 architecture), you can run the following command from the root directory of the repository: 
```
./main.py --time --scenario=task --experiment=splitCKPLUS --tasks=4 --vgg-root --network=cnn --iters=2000 --batch=32 --lr=0.0001 --latent-size=4096 --replay=generative --latent-replay=on --g-fc-uni=200 --distill
```

## Citation 

Please cite our paper in your publications if this code helps your research.

```
@inproceedings{stoychev2023latent,
  title={Latent Generative Replay for Resource-Efficient Continual Learning of Facial Expressions},
  author={Stoychev, Samuil and Churamani, Nikhil and Gunes, Hatice},
  booktitle={2023 IEEE 17th International Conference on Automatic Face and Gesture Recognition (FG)},
  pages={1--8},
  year={2023},
  organization={IEEE}
}
```

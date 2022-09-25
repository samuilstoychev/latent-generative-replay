# Latent Generative Replay for Resource-Efficient Continual Learning of Facial Expressions

This repository includes the code for the experiments presented in *"Latent Generative Replay for Resource-Efficient Continual Learning of Facial Expressions"* (Accepted at [FG 2023](https://fg2023.ieee-biometrics.org/))

![s](https://i.ibb.co/Kx12GGL/compared-approaches-alt-gen-2.jpg)

## Acknoledgments

The implementation has been adapted using the original Continual Learning repository by van de Ven et al. ([available on GitHub](https://github.com/GMvandeVen/continual-learning)). 

Model performance has been benchmarked adapting implementations from [the Avalanche project](https://avalanche.continualai.org/). 

## Files
The files in this repository are structured as follows: 

* `experiments` - includes the scripts used to run the evaluation experiments. Running them locally should generate the logs within the subdirectory. 
* The files in the root directory are structured similarly to the original Continual Learning repository. On top of that, we have added implementation for the latent generator (`autoencoder_latent.py`) and latent training (`train_latent.py`), added support for Naïve Rehearsal (`naive_rehearsal.py`) and VGG networks (`vgg_classifier.py`), and made ad-hoc changes to the other files where necessary. 

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

The other flags are documented in [the README file of van de Ven's continual learning repository](https://github.com/GMvandeVen/continual-learning). 

## Example Usage

You can run different CL methods as follows: 

### Latent Generative Replay (LGR) 

The command below will run Latent Generative Replay with CK+ on Task-IL using the VGG-16 architecture: 

```
./main.py --time --scenario=task --experiment=splitCKPLUS --tasks=4 --vgg-root --network=cnn --iters=2000 --batch=32 --lr=0.0001 --latent-size=4096 --replay=generative --latent-replay=on --g-fc-uni=200
```

### Latent Generative Replay with Distillation (LGR+d)

```
./main.py --time --scenario=task --experiment=splitCKPLUS --tasks=4 --vgg-root --network=cnn --iters=2000 --batch=32 --lr=0.0001 --latent-size=4096 --replay=generative --latent-replay=on --g-fc-uni=200 --distill
```

### Deep Generative Replay (DGR)

```
./main.py --time --scenario=task --experiment=splitCKPLUS --tasks=4 --vgg-root --network=cnn --iters=2000 --batch=32 --lr=0.0001 --latent-size=4096 --replay=generative --g-fc-uni=1600 
```

### Deep Generative Replay with Distillation (DGR+d)

```
./main.py --time --scenario=task --experiment=splitCKPLUS --tasks=4 --vgg-root --network=cnn --iters=2000 --batch=32 --lr=0.0001 --latent-size=4096 --replay=generative --g-fc-uni=1600 --distill
```

### Naïve Rehearsal (NR)

```
./main.py --time --scenario=task --experiment=splitCKPLUS --tasks=4 --vgg-root --network=cnn --iters=2000 --batch=32 --lr=0.0001 --latent-size=4096 --replay=naive-rehearsal --buffer-size=1000
```

### Latent Replay (NR) 

```
./main.py --time --scenario=task --experiment=splitCKPLUS --tasks=4 --vgg-root --network=cnn --iters=2000 --batch=32 --lr=0.0001 --latent-size=4096 --replay=naive-rehearsal --latent-replay=on --buffer-size=1000
```

## Citation 

Please cite our paper in your publications if it helps your research.

```
@inproceedings{Stoychev2023LGR, 
	author={Stoychev, Samuil and Churamani, Nikhil and Gunes, Hatice},
  	booktitle={2023 17th IEEE International Conference on Automatic Face and Gesture Recognition (FG))}, 
  	title={{Latent Generative Replay for Resource-Efficient Continual Learning of Facial Expressions}}, 
  	year={2023},
  	pages={To Appear},
}
```
#!/bin/bash 

cd ../..
echo "Switched to"
echo $(pwd)

echo "Running training with latent_size=g_fc_uni=10" 
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=10 --g-fc-uni=10 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/10units_0.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=10 --g-fc-uni=10 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/10units_1.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=10 --g-fc-uni=10 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/10units_2.py
echo "Running training with latent_size=g_fc_uni=20"
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=20 --g-fc-uni=20 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/20units_0.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=20 --g-fc-uni=20 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/20units_1.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=20 --g-fc-uni=20 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/20units_2.py
echo "Running training with latent_size=g_fc_uni=50"
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=50 --g-fc-uni=50 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/50units_0.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=50 --g-fc-uni=50 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/50units_1.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=50 --g-fc-uni=50 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/50units_2.py
echo "Running training with latent_size=g_fc_uni=100"
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=100 --g-fc-uni=100 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/100units_0.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=100 --g-fc-uni=100 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/100units_1.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=100 --g-fc-uni=100 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/100units_2.py
echo "Running training with latent_size=g_fc_uni=200"
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=200 --g-fc-uni=200 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/200units_0.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=200 --g-fc-uni=200 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/200units_1.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=200 --g-fc-uni=200 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/200units_2.py
echo "Running training with latent_size=g_fc_uni=300"
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=300 --g-fc-uni=300 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/300units_0.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=300 --g-fc-uni=300 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/300units_1.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=300 --g-fc-uni=300 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/300units_2.py
echo "Running training with latent_size=g_fc_uni=400"
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=400 --g-fc-uni=400 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/400units_0.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=400 --g-fc-uni=400 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/400units_1.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=400 --g-fc-uni=400 --no-gpus --seed="$RANDOM"> ./experiments/tuning_nogpu/400units_2.py










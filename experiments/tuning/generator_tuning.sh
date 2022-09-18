#!/bin/bash 

cd ../..
echo "Switched to"
echo $(pwd)

echo "Running training with latent_size=g_fc_uni=10" 
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=10 --g-fc-uni=10 --seed="$RANDOM"> ./experiments/tuning/10units_0.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=10 --g-fc-uni=10 --seed="$RANDOM"> ./experiments/tuning/10units_1.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=10 --g-fc-uni=10 --seed="$RANDOM"> ./experiments/tuning/10units_2.py
echo "Running training with latent_size=g_fc_uni=20"
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=20 --g-fc-uni=20 --seed="$RANDOM"> ./experiments/tuning/20units_0.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=20 --g-fc-uni=20 --seed="$RANDOM"> ./experiments/tuning/20units_1.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=20 --g-fc-uni=20 --seed="$RANDOM"> ./experiments/tuning/20units_2.py
echo "Running training with latent_size=g_fc_uni=50"
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=50 --g-fc-uni=50 --seed="$RANDOM"> ./experiments/tuning/50units_0.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=50 --g-fc-uni=50 --seed="$RANDOM"> ./experiments/tuning/50units_1.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=50 --g-fc-uni=50 --seed="$RANDOM"> ./experiments/tuning/50units_2.py
echo "Running training with latent_size=g_fc_uni=100"
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=100 --g-fc-uni=100 --seed="$RANDOM"> ./experiments/tuning/100units_0.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=100 --g-fc-uni=100 --seed="$RANDOM"> ./experiments/tuning/100units_1.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=100 --g-fc-uni=100 --seed="$RANDOM"> ./experiments/tuning/100units_2.py
echo "Running training with latent_size=g_fc_uni=200"
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=200 --g-fc-uni=200 --seed="$RANDOM"> ./experiments/tuning/200units_0.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=200 --g-fc-uni=200 --seed="$RANDOM"> ./experiments/tuning/200units_1.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=200 --g-fc-uni=200 --seed="$RANDOM"> ./experiments/tuning/200units_2.py
echo "Running training with latent_size=g_fc_uni=300"
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=300 --g-fc-uni=300 --seed="$RANDOM"> ./experiments/tuning/300units_0.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=300 --g-fc-uni=300 --seed="$RANDOM"> ./experiments/tuning/300units_1.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=300 --g-fc-uni=300 --seed="$RANDOM"> ./experiments/tuning/300units_2.py
echo "Running training with latent_size=g_fc_uni=400"
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=400 --g-fc-uni=400 --seed="$RANDOM"> ./experiments/tuning/400units_0.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=400 --g-fc-uni=400 --seed="$RANDOM"> ./experiments/tuning/400units_1.py
./main.py --replay=generative --time --scenario=task --network=cnn --iters=500 --out-channels=10 --latent-replay=on --latent-size=400 --g-fc-uni=400 --seed="$RANDOM"> ./experiments/tuning/400units_2.py

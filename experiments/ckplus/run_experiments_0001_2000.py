#!/usr/bin/env python3
import argparse
import subprocess
import time 
import random
import numpy as np
import os 

parser = argparse.ArgumentParser('./run_experiments.py', description='Run experiments.')


def get_command(replay_method, seed): 
        cmd = [
            "../../../main.py", 
            "--time", 
            "--scenario=task",
            "--experiment=splitCKPLUS",
            "--tasks=4", 
            "--network=cnn", 
            "--iters=2000", 
            "--batch=32",
            "--lr=0.0001",
            "--latent-size=4096", 
            "--vgg-root"
        ]
        if replay_method == "nr": 
            cmd.append("--replay=naive-rehearsal")
            cmd.append("--buffer-size=1500")
        elif replay_method == "lr": 
            cmd.append("--replay=naive-rehearsal")
            cmd.append("--latent-replay=on")
            cmd.append("--buffer-size=1000")
        elif replay_method == "gr":
            cmd.append("--replay=generative")
            cmd.append("--g-fc-uni=1600")
        elif replay_method == "lgr": 
            cmd.append("--replay=generative")
            cmd.append("--latent-replay=on")
            cmd.append("--g-fc-uni=200")
        elif replay_method == "grd":
            cmd.append("--replay=generative")
            cmd.append("--g-fc-uni=1600")
            cmd.append("--distill")
        elif replay_method == "lgrd": 
            cmd.append("--replay=generative")
            cmd.append("--latent-replay=on")
            cmd.append("--g-fc-uni=200")
            cmd.append("--distill")
        else: 
            raise Exception("Replay method not implemented: " + replay_method)

        cmd.append("--seed=" + str(seed))
        return cmd
        

def run_experiments():
    timestamp = time.strftime("%Y-%m-%d-%H-%M")
    print("Timestamp is", timestamp)
    run_command(["mkdir", timestamp])
    os.chdir(timestamp)

    commands_to_run = []
    random_seeds = [random.randint(0, 10000) for _ in range(3)]
    print("Random seeds are", random_seeds)

    for replay_method in ["nr", "lr", "gr", "lgr", "grd", "lgrd"]: 
        for seed in random_seeds: 
            filename = replay_method + "_" + str(seed)
            cmd = get_command(replay_method, seed)
            commands_to_run.append((cmd, filename))
    commands_to_run = np.random.permutation(commands_to_run) 

    n = len(commands_to_run)
    done = 0
    for cmd, filename in commands_to_run:
        print("Executing cmd =", " ".join(cmd))
        run_command(["echo", " ".join(cmd)], filename)
        run_command(cmd, filename)
        done += 1
        print("Executed %d/%d commands" % (done, n))
        time.sleep(1)

def run_command(cmd, outfile=None):
    if outfile: 
        outfile = open(outfile, "a")
    p = subprocess.Popen(cmd, stdout=outfile if outfile else None)
    ret_code = p.wait()
    if outfile: 
        outfile.flush()
    return ret_code

if __name__ == "__main__": 
    args = parser.parse_args()
    run_experiments() 

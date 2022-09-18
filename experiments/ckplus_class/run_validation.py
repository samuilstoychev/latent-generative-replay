#!/usr/bin/env python3
import argparse
import subprocess
import time 
import random
import numpy as np
import os 

def get_command(replay_method, seed): 
        cmd = [
            "../../main.py", 
            "--time", 
            "--scenario=class",
            "--experiment=splitCKPLUS",
            "--tasks=8", 
            "--network=cnn", 
            "--iters=3000", 
            "--batch=32",
            "--lr=0.0001",
            "--latent-size=4096", 
            "--vgg-root", 
            "--validation"
        ]
        if replay_method == "nr": 
            cmd.append("--replay=naive-rehearsal")
            cmd.append("--buffer-size=1000")
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
            cmd.append("--g-fc-uni=400")
        elif replay_method == "grd":
            cmd.append("--replay=generative")
            cmd.append("--g-fc-uni=1600")
            cmd.append("--distill")
        elif replay_method == "lgrd": 
            cmd.append("--replay=generative")
            cmd.append("--latent-replay=on")
            cmd.append("--g-fc-uni=400")
            cmd.append("--distill")
        else: 
            raise Exception("Replay method not implemented: " + replay_method)

        cmd.append("--seed=" + str(seed))
        cmd.append("--identifier=" + replay_method)
        return cmd
        

def run_experiments():
    commands_to_run = []
    # Using the seeds generated during the experiments from 2021-05-10-02-00
    random_seeds = [3312, 7711, 7888]

    for replay_method in ["nr", "lr", "gr", "lgr", "grd", "lgrd"]: 
        for seed in random_seeds: 
            cmd = get_command(replay_method, seed)
            commands_to_run.append(cmd)
    commands_to_run = np.random.permutation(commands_to_run) 

    n = len(commands_to_run)
    done = 0
    for cmd in commands_to_run:
        print("Executing cmd =", " ".join(cmd))
        run_command(cmd)
        done += 1
        print("Executed %d/%d commands" % (done, n))

def run_command(cmd):
    p = subprocess.Popen(cmd)
    ret_code = p.wait()
    return ret_code

if __name__ == "__main__": 
    run_experiments() 

#!/usr/bin/env python3
import argparse
import subprocess
import time 
import random
import numpy as np
import os 

parser = argparse.ArgumentParser('./run_experiments.py', description='Run experiments.')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--early-stopping', action='store_true')
parser.add_argument('--scenario', type=str, default='task', choices=['task', 'class'])

def get_command(architecture, replay_method, scenario, gpu, early_stopping, seed): 
        cmd = ["../../main.py", "--time"]
        if scenario=="task": 
            cmd.append("--scenario=task")
            cmd.append("--iters=500")
        else: 
            cmd.append("--scenario=class")
            cmd.append("--tasks=10")
            if architecture == "mlp": 
                cmd.append("--iters=500")
            else: 
                cmd.append("--iters=1000") 
        cmd.append("--network=" + architecture) 

        cmd.append("--latent-size=128")
        if replay_method == "gr":
            cmd.append("--replay=generative")
            cmd.append("--pretrain-baseline")
        elif replay_method == "grd": 
            cmd.append("--replay=generative")
            cmd.append("--pretrain-baseline")
            cmd.append("--distill")
        elif replay_method == "lgr": 
            cmd.append("--replay=generative")
            cmd.append("--latent-replay=on")
            cmd.append("--g-fc-uni=128")
        elif replay_method == "lgrd": 
            cmd.append("--replay=generative")
            cmd.append("--latent-replay=on")
            cmd.append("--g-fc-uni=128")
            cmd.append("--distill")
        elif replay_method == "nr": 
            cmd.append("--replay=naive-rehearsal")
            cmd.append("--pretrain-baseline")
        elif replay_method == "lr": 
            cmd.append("--replay=naive-rehearsal")
            cmd.append("--latent-replay=on")
        elif replay_method == "none": 
            cmd.append("--pretrain-baseline")

        if not gpu: 
            cmd.append("--no-gpus")
        if early_stopping: 
            cmd.append("--early-stop")

        cmd.append("--seed=" + str(seed))
        cmd.append("--validation")
        cmd.append("--identifier=" + replay_method)
        return cmd
        

def run_experiments(scenario, gpu, early_stopping):
    seed = random.randint(0, 10000)

    commands_to_run = []
    for architecture in ["cnn"]: 
        for replay_method in ["nr", "lr", "gr", "lgr", "grd", "lgrd", "none"]: 
            cmd = get_command(architecture, replay_method, scenario, gpu, early_stopping, seed)
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
    args = parser.parse_args()
    run_experiments(args.scenario, args.gpu, args.early_stopping) 

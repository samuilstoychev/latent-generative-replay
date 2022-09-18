#!/usr/bin/env python3
import argparse
import subprocess
import numpy as np

parser = argparse.ArgumentParser('./run_validation.py', description='Run validation.')


def get_command(replay_method, seed): 
        cmd = [
            "../../main.py", 
            "--time", 
            "--scenario=task",
            "--experiment=splitAffectNet",
            "--tasks=4", 
            "--network=cnn", 
            "--iters=2000", 
            "--batch=32",
            "--lr=0.0001",
            "--latent-size=4096", 
            "--vgg-root", 
            "--validation"
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
        elif replay_method == "none": 
            pass
        else: 
            raise Exception("Replay method not implemented: " + replay_method)

        cmd.append("--seed=" + str(seed))
        cmd.append("--identifier=" + replay_method)
        return cmd
        

def run_experiments():
    commands_to_run = []
    # Using the seeds from 2021-05-04-14-11
    random_seeds = [1842, 1856, 2306]

    for replay_method in ["nr", "lr", "gr", "lgr", "grd", "lgrd", "none"]: 
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

def run_command(cmd, outfile=None):
    p = subprocess.Popen(cmd)
    ret_code = p.wait()
    return ret_code

if __name__ == "__main__": 
    args = parser.parse_args()
    run_experiments() 

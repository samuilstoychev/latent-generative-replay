#!/usr/bin/env python3
import argparse
import subprocess
import time 
import random
import numpy as np
import os 

def get_command(n_iterations, g_fc_uni): 
    cmd = [
        "../../../main.py",
        "--replay=generative", 
        "--latent-replay=on",
        "--time",
        "--scenario=task",
        "--experiment=splitCKPLUS",
        "--tasks=4", 
        "--network=cnn", 
        "--iters=" + n_iterations, 
        "--lr=0.0001", 
        "--batch=32", 
        "--latent-size=4096", 
        "--g-fc-uni=" + g_fc_uni, 
        "--vgg-root"
    ]
    random_seed = random.randint(0, 10000)
    cmd.append("--seed=" + str(random_seed))
    return cmd
        

def run_experiments():
    timestamp = time.strftime("%Y-%m-%d-%H-%M")
    print("Timestamp is", timestamp)
    run_command(["mkdir", timestamp])
    os.chdir(timestamp)

    commands_to_run = []
    for n_iterations in ["500", "1000", "2000", "3000"]: 
        for g_fc_uni in ["200", "400", "800", "1600", "3200"]: 
            for i in range(3): 
                filename = n_iterations + "_" + g_fc_uni + "_" + str(i)
                cmd = get_command(n_iterations, g_fc_uni)
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
    run_experiments() 

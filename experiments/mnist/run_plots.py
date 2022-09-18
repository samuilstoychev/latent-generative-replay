#!/usr/bin/env python3
import argparse
import subprocess
import time
import random
import numpy as np
import os

parser = argparse.ArgumentParser('./run_experiments.py', description='Run experiments.')
parser.add_argument('--gpu', action='store_true', default=True)
parser.add_argument('--early-stopping', action='store_true')
parser.add_argument('--scenario', type=str, default='task', choices=['task'])


def form_folder_name(scenario, gpu, early_stopping):
	return scenario + ("_gpu" if gpu else "_nogpu") + ("_es" if early_stopping else "_fi")


def get_command(architecture, replay_method, scenario, gpu, early_stopping):
	cmd = ["../../../../main.py", "--time"]
	if scenario == "task":
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
	
	if not gpu:
		cmd.append("--no-gpus")
	if early_stopping:
		cmd.append("--early-stop")
	
	random_seed = random.randint(0, 10000)
	cmd.append("--seed=" + str(random_seed))
	return cmd


def run_experiments(scenario, gpu, early_stopping):
	timestamp = time.strftime("%Y-%m-%d-%H-%M")
	print("Timestamp is", timestamp)
	run_command(["mkdir", timestamp])
	run_command(["mkdir", timestamp + "/" + form_folder_name(scenario, gpu, early_stopping)])
	os.chdir(timestamp + "/" + form_folder_name(scenario, gpu, early_stopping))
	
	commands_to_run = []
	for architecture in ["cnn"]:
		# for replay_method in ["nr", "lr", "gr", "lgr", "grd", "lgrd"]:
		for replay_method in ["lgr"]:
			for i in range(3):
				filename = architecture + "_" + replay_method + "_" + str(i)
				cmd = get_command(architecture, replay_method, scenario, gpu, early_stopping)
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
		exit(0)


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
	run_experiments(args.scenario, args.gpu, args.early_stopping)

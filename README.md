ICCPS'24 Vulnerability Analysis for Safe Reinforcement Learning in Cyber-Physical System

This repo contains the source code of the ICCPS 2024 project.
The code is tested under Ubuntu 22.04, required packages and its version can be found at requirements.txt \

To simply reproduce the results in the paper, you can run the main.py file. \
The pretrained models can be find in models folder.
To customize this work to your problem, it can be done by follwing:
If you are trying to apply this work to a new system, you can build a new folder just like the benchmarks in CPS-benchmark folder.
   In this folder, it should have following scipts:
   a. attack_methods.py This file should have everything about the attack including the threat model and related parameters.
   b. train.py You can specify the training algorithms here, if you are trying to implement a new training algorithm, you can modify this file.
   c. run.py You can specify the output of the results such as settings of the plots etc.
   d. xxx_env.py This file should have the gym-like environment for training, you can also embed the system dynamics(ODE) into the this file.
   e. baseline.py This file contains the baselines we compared to our method, you can adjust parameters of them.
   
There are some uitlity files we have for this work mainly for the attacks which includes MAD attack, gradient attack and laa_attack, you may use it for convinience.

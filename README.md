ICCPS'24 Vulnerability Analysis for Safe Reinforcement Learning in Cyber-Physical System

This repo contains the source code of the ICCPS 2024 project.\
The code is tested under Ubuntu 22.04, required packages and its version can be found at requirements.txt

To simply reproduce the results in the paper, you can run the main.py file.\
The pretrained models can be find in models folder.\
To customize this work to your problem, it can be done by follwing:\
If you are trying to apply this work to a new system, you can build a new folder just like the benchmarks in CPS-benchmark folder.\
   In this folder, it should have following scipts:\
   a. attack_methods.py This file should have everything about the attack including the threat model and related parameters.\
   b. train.py You can specify the training algorithms here, if you are trying to implement a new training algorithm, you can modify this file.\
   c. run.py You can specify the output of the results such as settings of the plots etc.\
   d. xxx_env.py This file should have the gym-like environment for training, you can also embed the system dynamics(ODE) into the this file.\
   e. baseline.py This file contains the baselines we compared to our method, you can adjust parameters of them.
   
There are some uitlity files we have for this work mainly for the attacks which includes MAD attack, gradient attack and laa_attack, you may use it for convinience.
Below is an example of steps that if a user want to use this repo to train a new benchmark:
1. Build a xxx_env.py, then make sure it compatible with existing packages such as stable_baselines, there are usually three things need to be checked: the dimension of observation space and action space, the continuity of these spaces, the reward function you want to use to achieve safe RL.
2. Build the attack_methods.py, you should implement your attack here or you can use our attacks. If you system is linear, you can just use the one for DC motor. If you system is non-linear, you can just use the one for bicycle.
3. Build train.py, choose your trainning algorithms, we use stable_baselines for training, several common algorithms are supported such as SAC, PPO, TD3 etc. You can specify your own in this file, too.
4. Build run.py, after you have above programs, you can start the testing, you can customize the output format and monitor it. Additionally, you can also customize the visulization settings here to have a qualitative view of attack effectiveness.
5. Build the baselines.py, you can implement other baselines here that you want to compare with or use the implemented baselines we provided.

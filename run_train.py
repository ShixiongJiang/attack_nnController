# This a python script to train the model for each benchmark. We have provided the model so you may not need to run
# this.


print('start training')


from safetygym_benchmark.PointGoal import train
from safetygym_benchmark.CarCircle import train
from CPS_benchmark.DCmotor import train
from CPS_benchmark.bicycle import train
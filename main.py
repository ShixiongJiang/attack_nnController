# This is the main script to run the code to obtain the results table 1, table 2 and fig 2 in paper.
# The table 1 and 2 results will be shown in output.txt. And the fig will be shown in the generated .pdf files.
import sys
sys.stdout = open('output.txt','wt')
from safetygym_benchmark.PointGoal import run
from safetygym_benchmark.CarCircle import run
from CPS_benchmark.DCmotor import run
from CPS_benchmark.bicycle import run
"""Goal level 0."""
import safety_gymnasium
# Introduce the required objects
from safety_gymnasium.assets.geoms import Goal
# Need to inherit from BaseTask
from safety_gymnasium.bases.base_task import BaseTask




# env = GoalLevel0()
env_id = 'SafetyPointGoal1-v0'
safety_gymnasium_env = safety_gymnasium.make(env_id, render_mode=None)

env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium_env)
print(env.unwrapped.spec)
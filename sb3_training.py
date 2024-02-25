# -*- coding: utf-8 -*-
"""
- Train with PPO or A2C for 1_000_000 steps and save model every 10_000 step

python3 sb3_training.py --algo=PPO --timesteps=1_000_000 --save_every=10_000
python3 sb3_training.py --algo=A2C --timesteps=1_000_000 --save_every=10_000

- Load trained model:

python3 sb3_training.py --algo=PPO --model_name=40000.zip
python3 sb3_training.py --algo=A2C --model_name=20000.zip

- TensorrBoard: http://localhost:6006/

tensorboard --logdir=logs
"""

import argparse, sys

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from mayhem import *

# -------------------------------------------------------------------------------------------------

PLAY_WHILE_TRAINING = 1

DETERMINISTIC     = 1
MAX_FRAME_TO_PLAY = 10000
NB_TEST_RUN       = 5

# -------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('-width', '--width', help='', type=int, action="store", default=500)
parser.add_argument('-height', '--height', help='', type=int, action="store", default=500)

parser.add_argument('-algo', '--algo', help='', action="store", default="PPO", choices=("PPO", "A2C"))
parser.add_argument('-model_name', '--model_name', help='', action="store", default=None)

parser.add_argument('-timesteps', '--timesteps', help='', type=int, action="store", default=1_000_000)
parser.add_argument('-save_every', '--save_every', help='', type=int, action="store", default=20_000)

parser.add_argument('-log_dir', '--log_dir', help='', action="store", default="logs")

result = parser.parse_args()
args = dict(result._get_kwargs())

print("Args=", args)

# -------------------------------------------------------------------------------------------------

init_pygame()

game_window = GameWindow(args["width"], args["height"])

env = MayhemEnv(game_window, 
                level=1,
                max_fps=0, 
                debug_print=1,
                play_sound=False, 
                motion="gravity",
                sensor="ray", 
                record_play=False, 
                play_recorded=False)

#check_env(env)

# -------------------------------------------------------------------------------------------------

def play_model(model_path):

    #pygame.display.set_mode((400, 400))

    if args["algo"] == "A2C":
        model = A2C.load(model_path, env=env)
    elif args["algo"] == "PPO":
        model = PPO.load(model_path, env=env)

    for ep in range(NB_TEST_RUN):
        obs, info = env.reset()
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=DETERMINISTIC)
            obs, rewards, done, truncated, info = env.step(action, max_frame=MAX_FRAME_TO_PLAY)
            env.render(collision_check=False)

    #pygame.display.set_mode((1,1))

# -------------------------------------------------------------------------------------------------

class CustomSaveCallback(BaseCallback):

    def __init__(self, save_freq=10000, logdir="./logs/", model_prefix="PPO", verbose=1):
        super(CustomSaveCallback, self).__init__(verbose)
        
        self.save_freq = save_freq
        self.logdir = logdir
        self.save_path = os.path.join(self.logdir, "models", model_prefix)

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _on_step(self) -> bool:

        if self.n_calls % self.save_freq == 0:

            if self.verbose > 0:
                print(f"=> Saving {self.save_path}/{self.num_timesteps}.zip")

            self.model.save( f"{self.save_path}/{self.num_timesteps}" )

            if PLAY_WHILE_TRAINING:
                play_model(f"{self.save_path}/{self.num_timesteps}")

        return True

# -------------------------------------------------------------------------------------------------

if not args["model_name"]:
    #pygame.display.iconify()
    #pygame.display.set_mode((1,1)) 
    pass

obs, info = env.reset()

# -------------------------------------------------------------------------------------------------

# --- train
if not args["model_name"]:

    if args["algo"] == "A2C":
        model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=args["log_dir"])
    elif args["algo"] == "PPO":
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=args["log_dir"])

    save_cb = CustomSaveCallback(save_freq=args["save_every"], logdir=args["log_dir"], model_prefix=args["algo"], verbose=1)

    #model.learn(total_timesteps=args["timesteps"], reset_num_timesteps=False, tb_log_name=args["algo"])
    model.learn(total_timesteps=args["timesteps"], tb_log_name=args["algo"], callback=[save_cb, ])

# --- load
else:
    model_path = os.path.join(args["log_dir"], "models", args["algo"], args["model_name"])

    if args["algo"] == "A2C":
        model = A2C.load(model_path, env=env)
    elif args["algo"] == "PPO":
        model = PPO.load(model_path, env=env)

    for ep in range(NB_TEST_RUN):
        obs, info = env.reset()
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=DETERMINISTIC)
            obs, rewards, done, truncated, info = env.step(action, max_frame=MAX_FRAME_TO_PLAY)
            env.render(collision_check=False)

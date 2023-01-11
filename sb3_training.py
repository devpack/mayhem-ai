# -*- coding: utf-8 -*-
"""

"""

import argparse

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

from mayhem import *

MODE = "training"

# -------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('-width', '--width', help='', type=int, action="store", default=1200)
parser.add_argument('-height', '--height', help='', type=int, action="store", default=800)

result = parser.parse_args()
args = dict(result._get_kwargs())

print("Args", args)

# -------------------------------------------------------------------------------------------------

init_pygame()

game_window = GameWindow(args["width"], args["height"], MODE, debug_on_screen=0)

env = MayhemEnv(game_window, 
                vsync=False, 
                render_game=False, 
                nb_player=1, 
                mode=MODE, 
                motion="gravity",
                sensor="ray", 
                record_play=False, 
                play_recorded=False)
                
# -------------------------------------------------------------------------------------------------

#env = DummyVecEnv([lambda: env])
#env = make_vec_env(env, n_envs=10)
if 1:
    check_env(env)

#model_type = "A2C"
model_type = "PPO"

LOAD_MODEL_NAME = None  
LOAD_MODEL_NAME = "920000.zip"    
DETERMINISTIC = 0

EPISODES_PLAY = 5

TIMESTEPS = 10000
EPISODES_TRAIN = 100

if not LOAD_MODEL_NAME:
    pygame.display.iconify()

models_dir = "models/%s" % model_type
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

obs = env.reset()

# load
if LOAD_MODEL_NAME:
    model_path = f"{models_dir}/%s" % LOAD_MODEL_NAME

    if model_type == "A2C":
        model = A2C.load(model_path, env=env)
    elif model_type == "PPO":
        model = PPO.load(model_path, env=env)

    for ep in range(EPISODES_PLAY):
        obs = env.reset()
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=DETERMINISTIC)
            obs, rewards, done, info = env.step(action, max_frame=1000000000000000)
            env.render(collision_check=False)

# train
else:
    if model_type == "A2C":
        model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
    elif model_type == "PPO":
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

    #iters = 0
    for i in range(EPISODES_TRAIN):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=model_type)
        model.save(f"{models_dir}/{TIMESTEPS*i}")

    # train and play trained
    if 0:
        model.learn(total_timesteps=100000)

        episodes = 5

        for ep in range(episodes):
            obs = env.reset()
            done = False
            
            while not done:
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                env.render(collision_check=False)
                #print(rewards)

# test random moves
if 0:
    env.reset()

    print("sample action:", env.action_space.sample())
    print("observation space shape:", env.observation_space.shape)
    print("sample observation:", env.observation_space.sample())

    done = False

    while not done:
        new_state, reward, done, info = env.step(env.action_space.sample())
        env.render(collision_check=True)



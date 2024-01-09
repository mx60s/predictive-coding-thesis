import malmoenv
import argparse
from pathlib import Path
import time
import gym
import datetime
import random
from PIL import Image
import os
import numpy as np
import timeit
import math
import json
import logging
from typing import Tuple, List
from agents import *

logging.basicConfig(level=logging.DEBUG)

def convert_coords(x, z, min_x=-488, min_z=-690): # it's also possible that the +1 idx is a bit off
    return int(x - min_x), int(z - min_z)

def is_good_space(x, z, length, world_grid):
    # if this is super weird may need to switch dims
    if z * length + x >= len(world_grid) or x < 0 or z < 0:
        return False
    obs = world_grid[z * length + x]
    print(obs)
    return obs == 'grass' or obs == 'planks'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='malmoenv test')
    parser.add_argument('--missionname', type=str, default='facenorth', help='the mission xml')
    parser.add_argument('--port', type=int, default=9000, help='the mission server port')
    parser.add_argument('--steps', type=int, default=0, help='max number of steps per episode')
    parser.add_argument('--experimentUniqueId', type=str, default='test1', help="the experiment's unique id.")
    
    args = parser.parse_args()

    mission_path = '../envs/' + args.missionname + '.xml'
    
    xml = Path(mission_path).read_text()
    env = malmoenv.make()

    imgs_path = '../data/frames_random_' + args.missionname + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    coords_path = '../data/coords_random_' + args.missionname + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    bounds = ((-488, -423), (-690, -650))
    length = 65
    start = -444, -674
    yaw = 90
    start_relative = convert_coords(*start)
    frames, coords = [], []

    env.init(xml, args.port,
            server='127.0.0.05',
            exp_uid=args.experimentUniqueId,
            reshape=True)

    env.action_space.actions.append('strafe 1')
    env.action_space.actions.append('strafe -1')
    
    obs = env.reset()

    x, z = start_relative
    repeat = 0
    
    last_action = 0
    actions = list(range(4))

    # Walk forward until you get the world grid
    info = False
    while not info:
        obs, reward, done, info = env.step(0)
        x -= 1

    info_dict = json.loads(info)
    world_grid = info_dict['floor']

    try:
        for i in range(args.steps):
            img = np.asarray(env.render(mode='rgb_array'))
            frames.append(img)
            coords.append(np.array([x, z, yaw]))

            weights = [0.05, 0.05, 0.05, 0.05]
            weights[last_action] = 0.85
            a = random.choices(actions, weights)[0]
                
            obs, reward, done, info = env.step(a)
    
            last_action = a
            # I feel like my starting position is off and thats why the obstacle checking is a little wrong

            if a == 0 and is_good_space(x - 1, z, length, world_grid):
                x -= 1
            elif a == 1 and is_good_space(x + 1, z, length, world_grid):
                x += 1
            elif a == 2 and is_good_space(x, z - 1, length, world_grid):
                z -= 1
            elif a == 3 and is_good_space(x, z + 1, length, world_grid):
                z += 1
    
            time.sleep(.01)

    except Exception as e:
        print('Failed with exception', e)
        env.close()
    finally:
        np_frames, np_coords = None, None
        if frames and len(frames) > 7:
            print(len(frames))
            print("saving")
            np_frames = np.stack(frames, axis=0 )
            np.save(imgs_path, np_frames)
    
            np_coords = np.stack(coords, axis=0)
            np.save(coords_path, np_coords)
     

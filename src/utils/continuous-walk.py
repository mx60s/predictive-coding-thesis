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
logging.basicConfig(level=logging.DEBUG)

import malmoenv
import math
import gym
import heapq
import random
from typing import Tuple, List

random.seed(420)


class RatAgent():
    def __init__(self, mu, sigma, b, dt, border_region, bounds):
        self.mu = mu
        self.sigma = sigma
        self.b = b
        self.dt = dt
        self.border_region = border_region
        self.bounds = bounds

    def generate_trajectory(init_pos, init_heading, samples):
        v = np.random.rayleigh(self.b)
        position = np.zeros([samples+1, 2])
        head_dir = np.zeros(samples+1)
        turning = np.zeros(samples+1, dtype='bool')
    
        # Bounds of the Minecraft grid
        # remove later
        grid_min = np.array([-484, -694])
        grid_max = np.array([-427, -658])
    
        # Initial position and heading within Minecraft grid boundaries
        position[0] = init_pos #np.random.uniform(grid_min, grid_max)
        head_dir[0] = init_heading #np.random.uniform(0, 2*np.pi)
        ego_v = np.zeros(samples+1)
    
        random_turn = np.random.normal(self.mu, self.sigma, samples+1)
        random_vel = np.random.rayleigh(self.b, samples+1)
    
        for i in range(1, samples+1):
            # Check for proximity to border
            if np.any(position[i - 1] <= grid_min + border_region) or np.any(position[i-1] >= grid_max - border_region):
                turning[i-1] = True
                turn_angle = self.dt * random_turn[i]
                v = 0.25 * v
            else:
                v = random_vel[i]
                turn_angle = self.dt * random_turn[i]
    
            # Take a step
            ego_v[i-1] = v * self.dt
            position[i] = position[i-1] + ego_v[i-1] * np.array([np.cos(head_dir[i-1]), np.sin(head_dir[i-1])])
            head_dir[i] = (head_dir[i-1] + turn_angle) % (2 * np.pi)
    
        # Process the heading direction
        head_dir = (head_dir + np.pi) % (2 * np.pi) - np.pi
        return position[:-1,0], position[:-1, 1], head_dir[:-1], ego_v[:-1], turning[:-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='malmoenv test')
    parser.add_argument('--missionname', type=str, default='worldone', help='the mission xml')
    parser.add_argument('--port', type=int, default=9000, help='the mission server port')
    parser.add_argument('--steps', type=int, default=0, help='max number of steps per episode')
    parser.add_argument('--experimentUniqueId', type=str, default='test1', help="the experiment's unique id.")
    
    args = parser.parse_args()

    mission_path = '../envs/' + args.missionname + '.xml'
    
    xml = Path(mission_path).read_text()
    env = malmoenv.make()

    imgs_path = '../data/frames_' + args.missionname + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    coords_path = '../data/coords_' + args.missionname + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    env.init(xml, args.port,
            server='127.0.0.1',
             exp_uid=args.experimentUniqueId,
             reshape=True)

    mu = 0
    sigma = 5.76 * 2                        # stdev rotation velocity (rads/sec)
    b = 0.13 * 2 * np.pi                    # forward velocity rayleigh dist scale (m/sec)
    dt = 0.5                                # the time delta for each step
    border_region = 3                       # how close to the border should we start changing direction
    bounds = ((-484, -427), (-694, -658))
    turn_angle = 0
    
    agent = RatAgent

    original_pos = -451.5, -671.5
    frames, coords = [], [np.array(original_pos)]

    obs = env.reset()

    obs, reward, done, info = env.step(0)
    info_dict = json.loads(info)
    print(info_dict['XPos'], info_dict['ZPos'], info_dict['Yaw'])

    for i in range(args.steps):
        img = np.asarray(env.render(mode='rgb_array'))
        frames.append(img)

        #a = agent.next_step()
        #logging.info("Agent action: %s" % actions[a])

        #info = None
        #while not info:
        obs, reward, done, info = env.step(2)
        info_dict = json.loads(info)

        print(info_dict['XPos'], info_dict['ZPos'], info_dict['Yaw'])

        time.sleep(1)
        
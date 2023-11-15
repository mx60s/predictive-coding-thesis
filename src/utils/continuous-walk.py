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
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample_rotation(self):
        return random.gauss(mu=self.mu, sigma=self.sigma)


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

    original_pos = -451.5, -671.5
    bounds = ((-484, -427), (-694, -658))
    frames, coords = [], [np.array(original_pos)]

    agent = RatAgent(0.2, 0.2)
    obs = env.reset()

    obs, reward, done, info = env.step(0)
    info_dict = json.loads(info)
    print(info_dict['XPos'], info_dict['ZPos'], info_dict['Yaw'])

    for i in range(args.steps):
        #img = np.asarray(env.render(mode='rgb_array'))
        #frames.append(img)

        #a = agent.next_step()
        #logging.info("Agent action: %s" % actions[a])

        #info = None
        #while not info:
        obs, reward, done, info = env.step(2)
        info_dict = json.loads(info)

        print(info_dict['XPos'], info_dict['ZPos'], info_dict['Yaw'])

        time.sleep(1)
        
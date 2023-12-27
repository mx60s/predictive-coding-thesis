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

    bounds = ((-488, -423), (-690, -650))
    start = -446, -674
    start_relative = convert_coords(*start)
    frames, coords = [], []

    agent = AStarAgent((*start_relative, 90), 40, 65)
    
    env.init(xml, args.port,
            server='127.0.0.05',
            exp_uid=args.experimentUniqueId,
            reshape=True)

    #idx = 2
    #action_dict = {}
    #for i in range(-488, -423):
    #    for j in range(-690, -650):
    #        # lol
    #        env.action_space.actions.append(f'tp {i} 4 {j}')
    #        action_dict[(i, j)] = idx

    #env.action_space.actions.append('strafe 1')
    #env.action_space.actions.append('strafe -1')
    
    # Step backwards once to collect first observation
    obs = env.reset()
    info = None
    while not info:
        obs, reward, done, info = env.step(0)
        print('uhh')
    info_dict = json.loads(info)
    agent.x, agent.z = convert_coords(info_dict['XPos'], info_dict['ZPos'])
    agent.yaw = info_dict['Yaw']
        
    agent.world_state = info_dict['floor']

    #print(info_dict['XPos'], info_dict['ZPos'])
    #print(agent.x, agent.z)

    #print(agent.world_state)

    last_pos = agent.x, agent.z
    repeat = 0

    try:
        for i in range(args.steps):
            if not agent.target:
                agent.target = agent.generate_target()
            img = np.asarray(env.render(mode='rgb_array'))
            frames.append(img)
            coords.append(np.array([agent.x, agent.z, agent.yaw, agent.target[0], agent.target[1]]))
            print(agent.x, agent.z)
    
            #print(info_dict['XPos'], info_dict['ZPos'])
            #print(convert_back(agent.x, agent.z))
            
            #x, z = agent.next_step()
            #print("Agent action:", x, z)
    
            #coord = convert_back(x+5, z)
            #action_str = f'tp {coord[0]} 4 {coord[1]}'
            #print(coord)
            #print(action_str)
            #print(env.action_space.actions.index(action_str))
            #idx = env.action_space.actions.index(action_str)
    
            #obs, reward, done, info = env.step(idx)
    
            a = agent.next_step()
            info = None
    
            while not info:
                if a == 'backward':
                    print('backward')
                    obs, reward, done, info = env.step(2)
                    obs, reward, done, info = env.step(2)
                if a == 'right':
                    print('right')
                    obs, reward, done, info = env.step(2)
                if a == 'left':
                    print('left')
                    obs, reward, done, info = env.step(3)
                
                obs, reward, done, info = env.step(0)

            # this will be better to just change yaw, but leave for now.
    
            info_dict = json.loads(info)
            agent.x, agent.z = convert_coords(info_dict['XPos'], info_dict['ZPos'])
            agent.yaw = info_dict['Yaw']

            if agent.x < 0 or agent.x > 65 or agent.z < 0 or agent.z > 40:
                raise Exception("out of bounds")
    
            if last_pos == (agent.x, agent.z):
                print('repeat')
                repeat += 1
                if repeat > 4:
                    agent.target = agent.generate_target()
                    repeat = 0
            else:
                repeat = 0
    
            last_pos = agent.x, agent.z
    
            time.sleep(.01)

    except e:
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
     

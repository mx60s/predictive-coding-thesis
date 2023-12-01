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

    imgs_path = '../data/face-north/frames_' + args.missionname + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    coords_path = '../data/face-north/coords_' + args.missionname + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    original_pos = -453.5, -674.5, 0.0
    bounds = ((-484, -427), (-694, -658))
    frames, coords = [], []
    # water bounds are roughly -451.5, -440.5, -658.5,-672 and -451, -442 -677, -692
    
    # Shuffle the starting position within the environment
    x, z = -445.5, -660.5
    i = 0
    while (x > -451.5 and x < -440.5 and z < -658.5 and z > -672.5) or \
            (x > -451.5 and x < -442.5 and z < -677.5 and z > -692.5):
        x = random.randint(bounds[0][0], bounds[0][1]) + 0.5
        z = random.randint(bounds[1][0], bounds[1][1]) + 0.5
        i += 1
        if i > 100:
            raise Exception("I hate myself and you")

    print(f'Starting position {x, z}')
    xml.replace('x_position', str(x))
    xml.replace('z_position', str(z))

    agent = AStarAgent(original_pos, bounds)
    

    env.init(xml, args.port,
            server='127.0.0.05',
            exp_uid=args.experimentUniqueId,
            reshape=True)

    print(env.action_space.actions)
    env.action_space.actions.append('strafe 1')
    env.action_space.actions.append('strafe -1')
    print(env.action_space.actions)
    
    try:
        # Step backwards once to collect first observation
        obs = env.reset()
        info = None
        while not info:
            obs, reward, done, info = env.step(1)
        info_dict = json.loads(info)
        agent.world_state = info_dict['floor']
        
        last_place = agent.x, agent.z = info_dict['XPos'], info_dict['ZPos']
        agent.yaw = info_dict['Yaw']
        time.sleep(0.1)

        # commands are, in order, forward, backward, turn right, turn left
        repeat_count = 0
        special_repeats = 0
        special = False

        for i in range(args.steps):
            img = np.asarray(env.render(mode='rgb_array'))
            frames.append(img)
            coords.append(np.array([agent.x, agent.z, agent.yaw]))

            a = agent.next_step()
            logging.info("Agent action: %s" % a)

            if a == 'right':
                obs, reward, done, info = env.step(2)
            elif a == 'left':
                obs, reward, done, info = env.step(3)
            elif a == 'backwards':
                obs, reward, done, info = env.step(1)
            else:
                obs, reward, done, info = env.step(0)

            info_dict = json.loads(info)
            agent.x, agent.z, agent.yaw = info_dict['XPos'], info_dict['ZPos'], info_dict['Yaw']
            # this helps to unstick the agent, since my obstacle checking is not perfect right now
            if (agent.x, agent.z) == last_place:
                repeat_count += 1
                if repeat_count > 4:
                    env.step(1)
                    env.step(3)
                    agent._generate_target()
                if repeat_count > 10:
                    raise Exception("stuck")
            else:
                repeat_count = 0
                last_place = (agent.x, agent.z)

            # this is also dumb but works enough for now: fix
            if agent.x < bounds[0][0] or agent.x > bounds[0][1] or agent.z < bounds[1][0] or agent.z > bounds[1][1]:
                raise Exception("out of bounds")


            time.sleep(.01)
        
    except Exception as e:
        print("Failed to complete mission:", e)
        print(f"Completed {i} steps")

    finally:
        if frames and len(frames) > 7:
            print("saving")
            np_frames = np.stack(frames, axis=0 )
            np.save(imgs_path, np_frames)

            np_coords = np.stack(coords, axis=0)
            np.save(coords_path, np_coords)

        env.close()  

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
            server='127.0.0.05',
             exp_uid=args.experimentUniqueId,
             reshape=True)

    original_pos = -453.5, -674.5, 0.0
    bounds = ((-489, -424), (-695, -655))
    frames, coords = [], [np.array(original_pos)]

    agent = AStarAgent(original_pos, bounds)


    try:
        # Step backwards once to collect first observation
        obs = env.reset()
        info = None
        while not info:
            obs, reward, done, info = env.step(1)
        info_dict = json.loads(info)
        agent.world_state = info_dict['floor']
        last_place = agent.x, agent.z = info_dict['XPos'], info_dict['ZPos']

        time.sleep(0.1)

        # commands are, in order, forward, backward, turn right, turn left
        repeat_count = 0
        special_repeats = 0
        special = False

        for i in range(args.steps):
            img = np.asarray(env.render(mode='rgb_array'))
            frames.append(img)

            a = agent.next_step()
            logging.info("Agent action: %s" % a)

            if a == 'right':
                env.step(2)
                time.sleep(0.05)
            elif a == 'left':
                env.step(3)
                time.sleep(0.05)
            elif a == 'backwards':
                env.step(2)
                time.sleep(0.05)
                env.step(2)
                time.sleep(0.05)

            # whatever this is just to try and knock it out of any weird patterns
            if random.gauss(1, 0.5) > 1.5:
                env.step(2)
                time.sleep(0.05)
            
            obs, reward, done, info = env.step(0)

            info_dict = json.loads(info)
            agent.x, agent.z, agent.yaw = info_dict['XPos'], info_dict['ZPos'], info_dict['Yaw']
            coords.append(np.array([agent.x, agent.z, agent.yaw]))

            # this helps to unstick the agent, since my obstacle checking is not perfect right now
            if (agent.x, agent.z) == last_place:
                repeat_count += 1
                if (repeat_count > 4):
                    env.step(1)
                    env.step(2)
                    agent._generate_target()
                    repeat_count = 0
            else:
                repeat_count = 0
                last_place = (agent.x, agent.z)

            # this is also dumb but works enough for now: fix
            if agent.x < bounds[0][0] or agent.x > bounds[0][1] or agent.z < bounds[1][0] or agent.z > bounds[1][1]:
                raise Exception("out of bounds")


            time.sleep(.1)
        
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
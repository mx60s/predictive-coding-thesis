# modified from malmoenv video walk

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

from agents import AStarAgent

def extract_real_pos(info):
    info_dict = json.loads(info)
    return float(info_dict['XPos']), float(info_dict['ZPos'])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='malmoenv test')
    parser.add_argument('--missionname', type=str, default='default', help='the mission xml')
    parser.add_argument('--port', type=int, default=9000, help='the mission server port')
    parser.add_argument('--server', type=str, default='127.0.0.1', help='the mission server DNS or IP address')
    parser.add_argument('--port2', type=int, default=None, help="(Multi-agent) role N's mission port. Defaults to server port.")
    parser.add_argument('--server2', type=str, default=None, help="(Multi-agent) role N's server DNS or IP")
    parser.add_argument('--episodes', type=int, default=1, help='the number of resets to perform - default is 1')
    parser.add_argument('--episode', type=int, default=0, help='the start episode - default is 0')
    parser.add_argument('--role', type=int, default=0, help='the agent role - defaults to 0')
    parser.add_argument('--episodemaxsteps', type=int, default=0, help='max number of steps per episode')
    parser.add_argument('--resync', type=int, default=0, help='exit and re-sync every N resets'
                                                              ' - default is 0 meaning never.')
    parser.add_argument('--experimentUniqueId', type=str, default='test1', help="the experiment's unique id.")
    
    args = parser.parse_args()
    if args.server2 is None:
        args.server2 = args.server

    mission_path = '../envs/' + args.missionname + '.xml'
    
    xml = Path(mission_path).read_text()
    env = malmoenv.make()

    imgs_path = '../data/frames_' + args.missionname + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    coords_path = '../data/coords_' + args.missionname + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    env.init(xml, args.port,
             server=args.server,
             server2=args.server2, port2=args.port2,
             role=args.role,
             exp_uid=args.experimentUniqueId,
             episode=args.episode,
             resync=args.resync,
             reshape=True)

    original_pos = -451.5, 4, -671.5
    bounds = ((-484, -427), (-694, -658))
    actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
    frames, coords = [], [np.array(original_pos)]


    agent = AStarAgent(original_pos, bounds)

    try:
        tic=timeit.default_timer()
        
        for i in range(args.episodes):
            print("reset " + str(i))
            obs = env.reset()
            steps = 0
            done = False
            last_place = original_pos
            repeat_count = 0
            while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
                img = np.asarray(env.render(mode='rgb_array'))
                frames.append(img)

                a = agent.next_step()
                logging.info("Agent action: %s" % actions[a])
                
                # doublecheck why is this using a 1234 thing
                #print('prev position:', agent.x, agent.z)
                #print('a', a)
                obs, reward, done, info = env.step(a)
                #print('obs len', len(obs))
                #print('obs', obs[-1])
                print('action space', env.action_space)
                #print('info:', info[183:224])
                if info:
                    print(info)
                    agent.x, agent.z = extract_real_pos(info)
                    info_dict = json.loads(info)
                    agent.obstacles = info_dict['floor3x3']
                #print('new pos', agent.x, agent.z)
                # ok I can actually change the grid obs thing by adjusting y=-1 to something else
                # it's just looking at the floor lol

                # ok so now that I can see the whole grid
                # need to write a script to collect all of that into something I can use for my a star
                # it really just needs to grab it once
                # put everything on a .5 coord

                # I still haven't really figured out this stupid weird loop it falls into though
                # I think it's something to do with it not moving a full step and getting pulled back?
                # honestly tomorrow I might just switch over to sampling a velocity and a dir
                # because this is the worst.

                coords.append((agent.x, agent.y, agent.z))

                if (agent.x, agent.y, agent.z) == last_place:
                    repeat_count += 1
                    print('repeat')
                    if (repeat_count > 4):
                        print('generating new target bc repeats')
                        agent._generate_target()
                        repeat_count = 0
                else:
                    repeat_count = 0
                    last_place = (agent.x, agent.y, agent.z)

                steps += 1
    
                time.sleep(5)
    
        toc=timeit.default_timer()
        print(f'Elapsed time for {args.episodes} episodes and {args.episodemaxsteps} steps:{toc - tic}')

    #except Exception as e:
     #   print("Failed to complete mission:", e)
     #   print(f"Completed {i} episodes and {steps} steps")

    finally:
        if frames:
            np_frames = np.stack(frames, axis=0 )
            np.save(imgs_path, np_frames)

            np_coords = np.stack(coords, axis=0)
            np.save(coords_path, np_coords)

        env.close()

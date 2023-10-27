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

import logging
logging.basicConfig(level=logging.DEBUG)


# TODO change the env so it's 40 x 65, matches the described env, and also has A* traversal

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

    env.init(xml, args.port,
             server=args.server,
             server2=args.server2, port2=args.port2,
             role=args.role,
             exp_uid=args.experimentUniqueId,
             episode=args.episode,
             resync=args.resync,
             reshape=True)

    actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
    frames = []

    try:
        tic=timeit.default_timer()
        
        for i in range(args.episodes):
            print("reset " + str(i))
            obs = env.reset()
            steps = 0
            done = False
            while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
                img = np.asarray(env.render(mode='rgb_array'))
                frames.append(img)
                
                a = random.randint(0, len(actions) - 1)
                action = actions[a]
                #logging.info("Random action: %s" % actions[a])
                
                obs, reward, done, info = env.step(a)
                steps += 1
                #print("reward: " + str(reward))
                # print("done: " + str(done))
                #print("obs: " + str(obs))
                # print("info" + info)
    
                time.sleep(.05)
    
        toc=timeit.default_timer()
        print(f'Elapsed time for {args.episodes} episodes and {args.episodemaxsteps} steps:{toc - tic}')

    except Exception as e:
        print("Failed to complete mission:", e)
        print(f"Completed {i} episodes and {steps} steps")

    finally:
        if frames:
            np_frames = np.stack(frames, axis=0 )
            np.save(imgs_path, np_frames)

        env.close()
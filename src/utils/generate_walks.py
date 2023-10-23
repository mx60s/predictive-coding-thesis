# originally from: https://github.com/microsoft/malmo/blob/master/MalmoEnv/video_run.py

import malmoenv
import argparse
from pathlib import Path
import time
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import logging
logging.basicConfig(level=logging.DEBUG)


# do I need to 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='malmovnv test')
    parser.add_argument('--mission', type=str, default='envs/mission.xml', help='the mission xml')
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
    parser.add_argument('--video_path', type=str, default="video.mp4", help="Optional video path.")
    # TODO can I just save the video as a sequence of images?
    # should I be loading 
    args = parser.parse_args()
    if args.server2 is None:
        args.server2 = args.server

    xml = Path(args.mission).read_text()
    env = malmoenv.make()

    env.init(xml, args.port,
             server=args.server,
             server2=args.server2, port2=args.port2,
             role=args.role,
             exp_uid=args.experimentUniqueId,
             episode=args.episode,
             resync=args.resync,
             reshape=True)

    rec = VideoRecorder(env, args.video_path)
    # can I do this with malmoenv or does it need full malmo? agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)
    # or maybe it's literally just faster/easier to change mp4s into a string of images in the dataloader

    for i in range(args.episodes):
        print("reset " + str(i))
        obs = env.reset()
        rec.capture_frame()

        steps = 0
        done = False
        while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
            action = env.action_space.sample()
            rec.capture_frame()
            obs, reward, done, info = env.step(action)
            steps += 1
            print("reward: " + str(reward))
            # print("done: " + str(done))
            print("obs: " + str(obs))
            # print("info" + info)

            time.sleep(.05)

    rec.close()
    env.close()
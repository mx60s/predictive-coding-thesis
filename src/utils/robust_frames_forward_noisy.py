#from __future__ import print_function

#from future import standard_library
#standard_library.install_aliases()
from builtins import range
from builtins import object
import MalmoPython
import json
import logging
import math
import os
import random
import sys
import time
import malmoutils
import numpy as np
import datetime
if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk
    
malmoutils.fix_print()

agent_host = MalmoPython.AgentHost()
malmoutils.parse_command_line(agent_host)

save_images = True
if save_images:        
    from PIL import Image
    
class RandomAgent(object):

    def __init__(self, agent_host, action_set):
        self.rep = 0
        self.agent_host = agent_host
        self.action_set = action_set
        self.tolerance = 0.9
        self.world_grid = []
        self.grid_length = 69
        self.min_x = -490
        self.min_z = -692
        self.frames = []
        self.coords = []
        self.prev_turn = 0
        self.last_action = 0

    def waitForInitialState( self ):
        '''Before a command has been sent we wait for an observation of the world and a frame.'''
        # wait for a valid observation
        world_state = self.agent_host.peekWorldState()
        while world_state.is_mission_running and all(e.text=='{}' for e in world_state.observations):
            world_state = self.agent_host.peekWorldState()
        # wait for a frame to arrive after that
        num_frames_seen = world_state.number_of_video_frames_since_last_state
        while world_state.is_mission_running and world_state.number_of_video_frames_since_last_state == num_frames_seen:
            world_state = self.agent_host.peekWorldState()
        world_state = self.agent_host.getWorldState()

        if world_state.is_mission_running:
                
            assert len(world_state.video_frames) > 0, 'No video frames!?'
            
            obs = json.loads( world_state.observations[-1].text )
            self.world_grid =   obs[u'floor']
            self.prev_x   =     obs[u'XPos']
            self.prev_y   =     obs[u'YPos']
            self.prev_z   =     obs[u'ZPos']
            self.prev_yaw =     math.fmod(obs[u'Yaw'], 360)
            self.base_yaw = self.prev_yaw
            print('Initial position:',self.prev_x,',',self.prev_y,',',self.prev_z,'yaw',self.prev_yaw)
            
            if save_images:
                # save the frame, for debugging
                frame = world_state.video_frames[-1]
                image = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels) )
                self.iFrame = 0
                self.rep = self.rep + 1
                image.save( 'rep_' + str(self.rep).zfill(3) + '_saved_frame_' + str(self.iFrame).zfill(4) + '.png' )
            
        return world_state

    def convertCoords(self, x, z): # might need to be ceil? or? with neg?
        return int(x - self.min_x), int(z - self.min_z)

    def checkStep(self, x, z):
        x, z = self.convertCoords(x, z)

        if z * self.grid_length + x >= len(self.world_grid) or x < 0 or z < 0:
            return False
        obs = self.world_grid[z * self.grid_length + x]
        return obs == 'grass' or obs == 'planks'

    def waitForNextState( self ):
        '''After each command has been sent we wait for the observation to change as expected and a frame.'''
        # wait for the observation position to have changed
        #print('Waiting for observation...', end=' ')
        while True:
            world_state = self.agent_host.peekWorldState()
            if not world_state.is_mission_running:
                print('mission ended.')
                break
            if not all(e.text=='{}' for e in world_state.observations):
                obs = json.loads( world_state.observations[-1].text )
                self.curr_x   = obs[u'XPos']
                self.curr_y   = obs[u'YPos']
                self.curr_z   = obs[u'ZPos']
                self.curr_yaw = math.fmod(obs[u'Yaw'], 360)
                #print('curr?', self.curr_x, self.curr_z)
                if self.require_move:
                    if math.fabs( self.curr_x - self.prev_x ) > self.tolerance or\
                       math.fabs( self.curr_y - self.prev_y ) > self.tolerance or\
                       math.fabs( self.curr_z - self.prev_z ) > self.tolerance:
                        #print('received a move.')


                        break
                elif self.require_yaw_change:
                    if math.fabs( self.curr_yaw - self.prev_yaw ) > self.tolerance:
                        #print('received a turn.')
                        break
                else:
                    #print('received.')
                    break

        # wait for the render position to have changed
        #print('Waiting for render...', end=' ')
        while True:
            world_state = self.agent_host.peekWorldState()
            if not world_state.is_mission_running:
                #print('mission ended.')
                break
            if len(world_state.video_frames) > 0:
                frame = world_state.video_frames[-1]
                curr_x_from_render   = frame.xPos
                curr_y_from_render   = frame.yPos
                curr_z_from_render   = frame.zPos
                curr_yaw_from_render = math.fmod(frame.yaw, 360)
                if self.require_move:
                    if math.fabs( curr_x_from_render - self.prev_x ) > self.tolerance or\
                       math.fabs( curr_y_from_render - self.prev_y ) > self.tolerance or\
                       math.fabs( curr_z_from_render - self.prev_z ) > self.tolerance:
                        #print('received a move.')

                        break
                elif self.require_yaw_change:
                    if math.fabs( curr_yaw_from_render - self.prev_yaw ) > self.tolerance:
                        #print('received a turn.')
                        break
                else:
                    #print('received.')
                    break
            
        num_frames_before_get = len(world_state.video_frames)
        world_state = self.agent_host.getWorldState()

        if save_images:
            # save the frame, for debugging
            if world_state.is_mission_running:
                assert len(world_state.video_frames) > 0, 'No video frames!?'
                frame = world_state.video_frames[-1]
                image = np.array(Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels) ))
                self.iFrame = self.iFrame + 1
                self.frames.append(image)
                self.coords.append([self.curr_x, self.curr_z, self.curr_yaw])
            
        if world_state.is_mission_running:
            assert len(world_state.video_frames) > 0, 'No video frames!?'
            num_frames_after_get = len(world_state.video_frames)
            assert num_frames_after_get >= num_frames_before_get, 'Fewer frames after getWorldState!?'
            frame = world_state.video_frames[-1]
            obs = json.loads( world_state.observations[-1].text )
            self.curr_x   = obs[u'XPos']
            self.curr_y   = obs[u'YPos']
            self.curr_z   = obs[u'ZPos'] 
            self.curr_yaw = math.fmod(obs[u'Yaw'], 360)#math.fmod(180 + obs[u'Yaw'], 360)
            #print('1 New position from observation:',self.curr_x,',',self.curr_y,',',self.curr_z,'yaw',self.curr_yaw, end=' ')
            if math.fabs( self.curr_x   - self.expected_x   ) > self.tolerance or\
               math.fabs( self.curr_y   - self.expected_y   ) > self.tolerance or\
               math.fabs( self.curr_z   - self.expected_z   ) > self.tolerance or\
               math.fabs( self.curr_yaw - self.expected_yaw ) > self.tolerance:
                print(' - ERROR DETECTED! Expected:',self.expected_x,',',self.expected_y,',',self.expected_z,'yaw',self.expected_yaw)
                #print('RECEIVED:', self.curr_x,',',self.curr_y,',',self.curr_z,'yaw',self.curr_yaw,'turning', self.prev_turn)
                exit(1)
            else:
                pass
                #print('as expected.')
            curr_x_from_render   = frame.xPos
            curr_y_from_render   = frame.yPos
            curr_z_from_render   = frame.zPos
            #print('rendered yaw', frame.yaw)
            curr_yaw_from_render = math.fmod(frame.yaw, 360) #math.fmod(180 + frame.yaw ,360)
            #print('New position from render:',curr_x_from_render,',',curr_y_from_render,',',curr_z_from_render,'yaw',curr_yaw_from_render)
            if math.fabs( curr_x_from_render   - self.expected_x   ) > self.tolerance or\
               math.fabs( curr_y_from_render   - self.expected_y   ) > self.tolerance or \
               math.fabs( curr_z_from_render   - self.expected_z   ) > self.tolerance or \
               math.fabs( curr_yaw_from_render - self.expected_yaw ) > self.tolerance:
                print(' - ERROR DETECTED! Expected:',self.expected_x,',',self.expected_y,',',self.expected_z,'yaw',self.expected_yaw)
                exit(1)
            else:
                pass
                #print('as expected.')
            self.prev_x   = self.curr_x
            self.prev_y   = self.curr_y
            self.prev_z   = self.curr_z
            self.prev_yaw = self.curr_yaw
            
        return world_state
        
    def act( self ):
        '''Take an action from the action_set and set the expected outcome so we can wait for it.'''
        action_i = list(range(4))

        while True:
            weights = [0.85, 0.00, 0.05, 0.05]
            i_action = random.choices(action_i, weights)[0]

            if i_action == 0 or i_action == 1:
                i_yaw = indexOfClosest( [0,90,180,270], self.base_yaw )
                forward = [ (0,1), (-1,0), (0,-1), (1,0) ][ i_yaw ]
                if i_action == 0:
                    x = self.prev_x + forward[0]
                    z = self.prev_z + forward[1]
                else:
                    x = self.prev_x - forward[0]
                    z = self.prev_z - forward[1]

                if self.checkStep(x, z):
                    break
            
            else:
                break
        
        self.last_action = i_action

        yaw_noise =  random.randint(0, 15)

        if i_action == 0 or i_action == 1:
            self.agent_host.sendCommand( f'tp {x} {self.prev_y} {z}' )

            if yaw_noise != 0:
                self.expected_yaw = math.fmod(360 + self.base_yaw + yaw_noise, 360)
                self.agent_host.sendCommand( f'setYaw {self.expected_yaw}')
                self.require_yaw_change = True
            else:
                self.expected_yaw = self.prev_yaw
                self.require_yaw_change = False

            self.expected_x = x
            self.expected_z = z
            self.expected_y = self.prev_y
            self.require_move = True
        else:
            self.base_yaw = math.fmod(360 + self.base_yaw + [90, -90][i_action-2], 360 )
            self.expected_yaw = math.fmod(360 + self.base_yaw + yaw_noise, 360 )
            self.agent_host.sendCommand(f'setYaw {self.expected_yaw}')

            self.expected_x = self.prev_x
            self.expected_y = self.prev_y
            self.expected_z = self.prev_z
            self.require_move = False
            self.require_yaw_change = True

        #print('base yaw', self.base_yaw)
        #print('my yaw', self.expected_yaw)
        #print('Sending', action)
    
        
def indexOfClosest( arr, val ):
    '''Return the index in arr of the closest float value to val.'''
    i_closest = None
    for i,v in enumerate(arr):
        d = math.fabs( v - val )
        if i_closest == None or d < d_closest:
            i_closest = i
            d_closest = d
    return i_closest

# -- set up the mission --
xml = '''<?xml version="1.0"?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>Facing north (like the paper)</Summary>
        </About>

        <ModSettings>
            <MsPerTick>50</MsPerTick>
        </ModSettings>

        <ServerSection>
            <ServerInitialConditions>
                <Time>
                    <StartTime>6000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
            </ServerInitialConditions>
            <ServerHandlers>
		    <FileWorldGenerator src="/home/mag/malmo/mcworldfinished-2" forceReset="1"/> 
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>Agent0</Name>
            <AgentStart>
                  <Placement pitch="0" x="-467.5" y="4" yaw="90" z="-677.5"/>
            </AgentStart>
            <AgentHandlers>
                <ObservationFromGrid>
                    <Grid absoluteCoords="true" name="floor">
                    <min x="-490" y="3" z="-692"/>
                    <max x="-422" y="3" z="-649"/>
                    </Grid>
                </ObservationFromGrid>
                <ObservationFromFullStats/>
                <VideoProducer want_depth="false">
                   <Width>128</Width>
                   <Height>128</Height>
                </VideoProducer>
            </AgentHandlers>            
        </AgentSection>
</Mission>'''


my_client_pool = MalmoPython.ClientPool()
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10001))
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10002))

my_mission = MalmoPython.MissionSpec(xml,True)

missionname = 'faceforward'
imgs_path = 'data/frames_random_' + missionname + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
coords_path = 'data/coords_random_' + missionname + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# -- test each action set in turn --
max_retries = 3
action_sets = ['teleport']#,'discrete_relative', 'teleport']
for action_set in action_sets:

    if action_set == 'discrete_absolute':
        my_mission.allowAllDiscreteMovementCommands()
    elif action_set == 'discrete_relative':
        my_mission.allowAllDiscreteMovementCommands()
    elif action_set == 'teleport':
        my_mission.allowAllAbsoluteMovementCommands()
    else:
        print('ERROR: Unsupported action set:',action_set)
        exit(1)

    my_mission_recording = MalmoPython.MissionRecordSpec()
    #my_mission_recording = malmoutils.get_default_recording_object(agent_host, action_set)
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_client_pool, my_mission_recording,0, "craftTestExperiment" )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2.5)

    print("Waiting for the mission to start", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    print()

    # the main loop:
    steps = 12000#0 # 83k / 5
    s = 0
    agent = RandomAgent( agent_host, action_set )
    world_state = agent.waitForInitialState()
    
    while world_state.is_mission_running and s < steps:
        agent.act()
        world_state = agent.waitForNextState()
        s += 1

    print('Saving frames')

    if agent.frames:
        print(len(agent.frames), len(agent.coords))
        np_frames = np.stack(agent.frames, axis=0 )
        np.save(imgs_path, agent.frames)

        np_coords = np.stack(agent.coords, axis=0)
        np.save(coords_path, np_coords)

    time.sleep(5) # allow room to release clients

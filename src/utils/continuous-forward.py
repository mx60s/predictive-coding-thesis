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
        self.tolerance = 0.5
        self.world_grid = []
        self.grid_length = 69
        self.min_x = -490
        self.min_z = -692
        self.frames = []
        self.coords = []
        self.turn_momentum = 0
        self.momentum_decay = 0.8
        self.turn_change_chance = 0.2

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
            self.prev_yaw =     obs[u'Yaw']
            print('Initial position:',self.prev_x,',',self.prev_y,',',self.prev_z,'yaw',self.prev_yaw)
            
            if save_images:
                # save the frame, for debugging
                frame = world_state.video_frames[-1]
                image = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels) )
                self.iFrame = 0
                self.rep = self.rep + 1
                image.save( 'rep_' + str(self.rep).zfill(3) + '_saved_frame_' + str(self.iFrame).zfill(4) + '.png' )
            
        return world_state

    def convert_coords(self, x, z):
        grid_points = []
        for dx in [-0.3, 0, 0.3]:
            for dz in [-0.3, 0, 0.3]:
                grid_points.append((int(x + dx - self.min_x), int(z + dz - self.min_z)))
        return grid_points

    def check_step(self, x, z):
        # avoid tricky spots
        if (x, z) in [(-487.5, -678.5), (-447.5, -692.5)]:
            return False

        grid_points = self.convert_coords(x, z)
        #print(len(grid_points))

        for gx, gz in grid_points:
            if gz * self.grid_length + gx >= len(self.world_grid) or gx < 0 or gz < 0:
                return False

            obs = self.world_grid[gz * self.grid_length + gx]
            if not (obs == "grass" or obs == "planks"):
                return False

        return True

    def check_proximity(self, x, z):
        proximity_checks = []
        proximity_score = 0

        # Define a range or set of directions to check around the agent
        for dx in [-0.4, 0, 0.4]:
            for dz in [-0.4, 0, 0.4]:
                if dx == 0 and dz == 0:
                    continue  # Skip checking the current position

                check_x = x + dx
                check_z = z + dz
                proximity_checks.append(self.check_step(check_x, check_z))

        # Count the number of free (True) vs. obstructed (False) spaces
        free_spaces = proximity_checks.count(True)
        total_checks = len(proximity_checks)

        # Calculate proximity score based on the ratio of free spaces to total checks
        # Higher score means more free space around, lower score means closer to obstacles
        if total_checks > 0:
            proximity_score = free_spaces / total_checks

        return proximity_score

    def waitForNextState( self ):
        '''After each command has been sent we wait for the observation to change as expected and a frame.'''
        # wait for the observation position to have changed
        #print('Waiting for observation...', end=' ')
        while True:
            world_state = self.agent_host.peekWorldState()
            if not world_state.is_mission_running:
                #print('mission ended.')
                break
            if not all(e.text=='{}' for e in world_state.observations):
                #print('got observation')
                obs = json.loads( world_state.observations[-1].text )
                self.curr_x   = obs[u'XPos']
                self.curr_y   = obs[u'YPos']
                self.curr_z   = obs[u'ZPos']
                self.curr_yaw = math.fmod(obs[u'Yaw'], 360)
                #print('curr?', self.curr_x, self.curr_z)
                if self.require_move:
                    #print('require move')
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
                    ##print('received.')
                    break

        # wait for the render position to have changed
        #print('Waiting for render...', end=' ')
        while True:
            world_state = self.agent_host.peekWorldState()
            if not world_state.is_mission_running:
                #print('mission ended.')
                break
            if len(world_state.video_frames) > 0:
                #print('render changed')
                frame = world_state.video_frames[-1]
                curr_x_from_render   = frame.xPos
                curr_y_from_render   = frame.yPos
                curr_z_from_render   = frame.zPos
                curr_yaw_from_render = math.fmod(frame.yaw, 360)
                if self.require_move:
                    #print('render move required')
                    if math.fabs( curr_x_from_render - self.prev_x ) > self.tolerance or\
                       math.fabs( curr_y_from_render - self.prev_y ) > self.tolerance or\
                       math.fabs( curr_z_from_render - self.prev_z ) > self.tolerance:
                        #print('render received a move.')

                        break
                elif self.require_yaw_change:
                    if math.fabs( curr_yaw_from_render - self.prev_yaw ) > self.tolerance:
                        #print('render received a turn.')
                        break
                else:
                    #print('render received.')
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
                #print(' - ERROR DETECTED! Expected:',self.expected_x,',',self.expected_y,',',self.expected_z,'yaw',self.expected_yaw)
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
                #print(' - ERROR DETECTED! Expected:',self.expected_x,',',self.expected_y,',',self.expected_z,'yaw',self.expected_yaw)
                exit(1)
            else:
                pass
                #print('as expected.')
            self.prev_x   = self.curr_x
            self.prev_y   = self.curr_y
            self.prev_z   = self.curr_z
            self.prev_yaw = self.curr_yaw
            
        return world_state


    def act(self):
        checks = 0
        
        while True:
            checks += 1

            if checks > 30:
                # if we get really stuck then help it out
                max_turn = 360
            else:
                proximity = self.check_proximity(self.prev_x, self.prev_z)
                max_turn = 15 + int(proximity * 35)

            if random.random() < self.turn_change_chance:
                self.turn_momentum *= -1

            if self.turn_momentum > 0:
                yaw_disp = random.randint(0, max_turn)
            elif self.turn_momentum < 0:
                yaw_disp = random.randint(-max_turn, 0)
            else:
                yaw_disp = random.randint(-max_turn, max_turn)

            new_yaw = math.fmod(360 + self.prev_yaw + yaw_disp, 360)

            x = self.prev_x - math.sin(math.radians(new_yaw))
            z = self.prev_z + math.cos(math.radians(new_yaw))

            if self.check_step(x, z):
                break
        
        self.agent_host.sendCommand(f'tp {x} {self.prev_y} {z}')

        self.expected_x = x
        self.expected_y = self.prev_y
        self.expected_z = z
        self.require_move = True
        self.turn_momentum = yaw_disp + self.turn_momentum * self.momentum_decay

        if yaw_disp != 0:
            self.agent_host.sendCommand(f'setYaw {new_yaw}')
            self.expected_yaw = new_yaw
            self.require_yaw_change = True
        else:
            self.expected_yaw = self.prev_yaw
            self.require_yaw_change = False


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
		    <FileWorldGenerator src="/home/maggie/malmo-real/mcworldfinished-2" forceReset="1"/> 
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>Agent0</Name>
            <AgentStart>
                  <Placement pitch="0" x="-475.5" y="4" yaw="90" z="-673.5"/>
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

my_mission = MalmoPython.MissionSpec(xml,True)

missionname = 'faceforward'
imgs_path = 'data/frames_random_' + missionname + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
coords_path = 'data/coords_random_' + missionname + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# -- test each action set in turn --
max_retries = 1
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
    steps = 14000
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

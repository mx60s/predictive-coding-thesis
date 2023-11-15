import malmoenv
import math
import gym
import heapq
import random
from typing import Tuple, List

class AStarAgent:
    """
    An agent which comes up with random paths throughout the environment, and then generates the 
    actions via A* to traverse them.
    """
    def __init__(self, start_pos: Tuple, bounds: Tuple):
        self.x, self.z, self.yaw = start_pos
        self.x_bound = bounds[0]
        self.z_bound = bounds[1]
        self.target = None
        self.world_state = []

    def get_obs_from_grid(self, x, z):
        x = math.floor(x)
        z = math.floor(z)

        w = self.x_bound[1] - self.x_bound[0] + 1
        h = self.z_bound[1] - self.z_bound[0] + 1
        
        # convert coordinates to 0-indexing
        x_idx = x - self.x_bound[0]
        z_idx = z - self.z_bound[0]
        # calculate 1d index
        idx = z_idx * w + x_idx
        
        if 0 <= idx < len(self.world_state):
            return self.world_state[idx]
        else:
            return 'oob'

    def _is_obstacle(self, x, z):
        obs = self.get_obs_from_grid(x, z)
        return not (obs == 'grass' or obs == 'plank')

    def _generate_target(self):
        i = 0
        while True:
            x = random.randint(self.x_bound[0], self.x_bound[1])
            z = random.randint(self.z_bound[0], self.z_bound[1])
            # something is 100% wrong with the bounds but it works ok enough for now
            if not self._is_obstacle(x, z):
                self.target = (x + 0.5, z + 0.5)
                break
            i += 1
    
    def _heuristic(self, a, b):
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) 
    
    def _get_neighbors(self, point):
        x, z = point
        neighbors = [(x+1, z), (x-1, z), (x, z+1), (x, z-1)]
        return [n for n in neighbors if not self._is_obstacle(*n)]
   
    def _a_star_search(self):
        """Find the next step towards the target using the A* algorithm."""
        start = (self.x, self.z)
        target = self.target
        closed_set = set()
        open_set = [(0, start)]
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, target)}
        came_from = {}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == target:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path[0]  # Return the next step after the start

            closed_set.add(current)

            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                # The distance from start to a neighbor
                tentative_g_score = g_score[current] + 1  # Each step costs 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, target)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # If there is no path to the target

    def _get_relative_dir(self, d_step):
        # directions are in yaw degrees
        # keeping this as strings here for readability
        direction_map = {
            180 : {'left': (-1, 0), 'right': (1,0), 'forward': (0,-1), 'backward': (0,1)},
            0 : {'left': (1, 0), 'right': (-1,0), 'forward': (0,1), 'backward': (0,-1)},
            270 : {'left': (0, -1), 'right': (0,1), 'forward': (1, 0), 'backward': (-1, 0)},
            90 : {'left': (0, 1), 'right': (0,-1), 'forward': (-1, 0), 'backward': (1,0)},
        }

        for relative_dir, move in direction_map[self.yaw].items():
            if move == d_step:
                return relative_dir

        return 'backward'

    def next_step(self):
        if not self.target or self._heuristic((self.x, self.z), self.target) < 4:
            self._generate_target()
    
        next_point = self._a_star_search()
        if not next_point: # if you can't find a path, make new target and try again
            self.target = None
            return 'special'

        # Translate point to action
        dx = next_point[0] - self.x
        dz = next_point[1] - self.z

        return self._get_relative_dir((dx, dz))
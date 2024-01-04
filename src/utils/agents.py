import math
import heapq
import random
from typing import Tuple, List


class AStarAgent:
    """
    An agent which comes up with random paths throughout the environment, and then generates the
    actions via A* to traverse them.
    """
    def __init__(self, start_pos: Tuple, width: int, length: int):
        self.x, self.z, self.yaw = start_pos
        self.width = width
        self.length = length
        self.target = None
        self.world_state = []

    def get_obs_from_grid(self, x, z):
        # if this is super weird may need to switch dims
        if z * self.length + x >= len(self.world_state) or x < 0 or z < 0:
            return 'bad'
        return self.world_state[z * self.length + x]

    def generate_target(self):
        while True:
            #print('gennnn target')
            x, z = random.randrange(0, self.length), random.randrange(0, self.width)
            #print(x, z)
            if not self._is_obstacle(x, z):
                return x, z

    def _is_obstacle(self, x, z):
        obs = self.get_obs_from_grid(x, z)
        #print('obs', obs)
        return not (obs == 'grass' or obs == 'plank')
    
    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _get_neighbors(self, point):
        x, z = point
        neighbors = [(x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)]
        return [n for n in neighbors if not self._is_obstacle(*n)]

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from.keys():
            current = came_from[current]
            path.insert(0, current)
            
        return path
   
    def _a_star_search(self, start, target):
        print('search start', start)
        print('search target', target)
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, target)} # default value of inf?
        
        while open_set:
            #print('open_set')
            _, current = heapq.heappop(open_set)
            
            if current == target:
                path = self._reconstruct_path(came_from, current)
                #print(path)
                return path

            #print('neighbors', self._get_neighbors(current))
            for neighbor in self._get_neighbors(current):
                #print('neighbor')
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, target)
                    
                    if neighbor not in open_set:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        
        return None

    def _get_relative_dir(self, d_step):
        #print('relative dir')

        direction_map = {
            180: {
                "left": (-1, 0),
                "right": (1, 0),
                "forward": (0, -1),
                "backward": (0, 1),
            },
            0: {
                "left": (1, 0),
                "right": (-1, 0),
                "forward": (0, 1),
                "backward": (0, -1),
            },
            270: {
                "left": (0, -1),
                "right": (0, 1),
                "forward": (1, 0),
                "backward": (-1, 0),
            },
            90: {
                "left": (0, 1),
                "right": (0, -1),
                "forward": (-1, 0),
                "backward": (1, 0),
            }
        }

        return next(
            (dir for dir, move in direction_map[self.yaw].items() if move == d_step),
            "backward",
        )

    def next_step(self):
        #print('target', self.target)
        if not self.target or self._heuristic((self.x, self.z), self.target) < 2:
            #print('gen target')
            self.target = self.generate_target()
            print('new target', self.target)
    
        trajectory = self._a_star_search((self.x, self.z), self.target)
        if not trajectory: # if you can't find a path, make new target and try again
            print('no path')
            self.target = None
            return "special"

        #print(trajectory)
        
        dx = trajectory[1][0] - self.x
        dz = trajectory[1][1] - self.z

        # Translate point to action
        #print(trajectory[1])
        #return trajectory[1]
<<<<<<< HEAD
        return self._get_relative_dir((dx, dz))
=======
        return self._get_relative_dir((dx, dz))
>>>>>>> 98602d9a1c7fdf920487e66dd587b16ac017451e

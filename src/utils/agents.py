import malmoenv
import gym
import heapq
import random
import math
from typing import Tuple, List

class AStarAgent:
    """
    An agent which comes up with random paths throughout the environment, and then generates the 
    actions via A* to traverse them.
    """
    def __init__(self, start_pos: Tuple, bounds: Tuple, obstacles: List):
        self.x, self.y, self.z = start_pos
        self.obstacles = obstacles
        self.x_bound = bounds[0]
        self.z_bound = bounds[1]
        self.target = None
        self.steps = 0

    # this isn't perfect, also need to check overhanging things
    def _is_obstacle(self, x, y, z):
        x_f = math.floor(self.x)
        z_f = math.floor(self.z)
        for o in self.obstacles:
            if x_f == o[0] and z_f == o[1]:
                #print('obstacle at', o)
                return True
        return False
    
    def _generate_target(self):
        while True:
            x = random.randint(self.x_bound[0], self.x_bound[1]) + 0.5
            z = random.randint(self.z_bound[0], self.z_bound[1]) + 0.5
            dist = abs(x - self.x) + abs(z - self.z)
            #print('dist', dist)
            if dist >= 5 and not self._is_obstacle(x, self.y, z):
                self.target = (x, self.y, z)
                break
        print('target', self.target)
    
    def _heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[2] - b[2])
    
    def _get_neighbors(self, point):
        x, y, z = point
        neighbors = [(x+1, y, z), (x-1, y, z), (x, y, z+1), (x, y, z-1)]
        return [n for n in neighbors if not self._is_obstacle(*n)]
    
    def _a_star_search(self):
        start = (self.x, self.y, self.z)
        goal = self.target
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            _, current = heapq.heappop(frontier)
            
            if current == goal:
                break
            
            for neighbor in self._get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._heuristic(goal, neighbor)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        current = goal
        next_step = None
        while current in came_from and came_from[current] is not None:
            next_step = current
            current = came_from[current]

        return next_step
    
    def next_step(self):
        if not self.target or self.steps > 10:# or self.target == (self.x, self.y, self.z):
            print('stuck, recalculating')
            self._generate_target()
            self.steps = 0
        
        next_point = self._a_star_search()
        
        if not next_point:
            self._generate_target()
            next_point = self._a_star_search()
        
        # Translate point to action
        dx = next_point[0] - self.x
        dz = next_point[2] - self.z
        print('dx dz', dx, dz)

        self.steps += 1
        
        if dx == 1:
            return 3
        elif dx == -1:
            return 2
        elif dz == 1:
            return 0
        elif dz == -1:
            return 1
        else:
            self.steps -= 1
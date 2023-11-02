import malmoenv
import gym
import heapq
import random
from typing import Tuple

class AStarAgent:
    """
    An agent which comes up with random paths throughout the environment, and then generates the 
    actions via A* to traverse them.
    """
    def __init__(self, start_pos: Tuple, bounds: Tuple):
        self.x, self.y, self.z = start_pos
        self.x_bound = start_pos[0]
        self.z_bound = start_pos[1]
        self.target = None
        self.path = []
    
    def _is_obstacle(self, x, y, z):
        
        return False
    
    def _generate_target(self):
        while True:
            x = random.randint(self.x_bound[0], self.x_bound[1])
            z = random.randint(self.z_bound[0], self.z_bound[1])
            if not self._is_obstacle(x, self.y, z):
                self.target = (x, self.y, z)
                break
    
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
        
        # Reconstruct path
        self.path = []
        current = goal
        while current != start:
            self.path.append(current)
            current = came_from[current]
        self.path.reverse()
    
    def next_step(self):
        if not self.path and not self.target:
            self._generate_target()
            self._a_star_search()
        
        if not self.path and self.target:
            self._generate_target()
            self._a_star_search()
        
        next_point = self.path.pop(0)
        self.x, self.y, self.z = next_point
        
        # Translate point to action
        dx = next_point[0] - self.x
        dz = next_point[2] - self.z
        if dx == 1:
            return 3
        elif dx == -1:
            return 2
        elif dz == 1:
            return 0
        elif dz == -1:
            return 1

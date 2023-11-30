import math
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

    def _get_index_from_coordinates(self, x: int, z: int) -> int:
        """
        Converts 2D grid coordinates to a 1D index.
        """
        x_idx = x - self.x_bound[0]
        z_idx = z - self.z_bound[0]
        return z_idx * (self.x_bound[1] - self.x_bound[0] + 1) + x_idx

    def _get_obs_from_grid(self, x: int, z: int) -> str:
        """
        Retrieves the observation at the specified grid location.
        """
        x_idx, z_idx = map(math.floor, [x, z])

        idx = self._get_index_from_coordinates(x_idx, z_idx)

        if 0 <= idx < len(self.world_state):
            return self.world_state[idx]
        else:
            return "oob"

    def _is_obstacle(self, x, z):
        """
        Determines if the given coordinates correspond to an obstacle.
        """
        obs = self._get_obs_from_grid(x, z)
        return not (obs == "grass" or obs == "plank")

    def _generate_target(self):
        while True:
            x, z = random.randint(*self.x_bound), random.randint(*self.z_bound)
            if not self._is_obstacle(x, z):
                self.target = (x + 0.5, z + 0.5)
                break

    def _heuristic(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, point: Tuple[float, float]) -> List[Tuple[float, float]]:
        x, z = point
        neighbors = [(x + 1, z), (x - 1, z), (x, z + 1), (x, z - 1)]
        return [n for n in neighbors if not self._is_obstacle(*n)]

    def _a_star_search(self) -> Tuple[float, float]:
        """
        Implements the A* algorithm to find the next step towards the target.
        """
        start = (self.x, self.z)
        closed_set = set()
        open_set = [(0, start)]
        g_score, f_score = {start: 0}, {start: self._heuristic(start, self.target)}
        came_from = {}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == self.target:
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
                    f_score[neighbor] = tentative_g_score + self._heuristic(
                        neighbor, self.target
                    )
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def _get_relative_dir(self, d_step):
        """
        Translates a direction step into a relative direction based on the agent's current yaw.
        """
        # directions are in yaw degrees
        # keeping this as strings here for readability
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
            },
        }

        return next(
            (dir for dir, move in direction_map[self.yaw].items() if move == d_step),
            "backward",
        )

    def next_step(self) -> str:
        """
        Determines the next action for the agent. Generates a new target if needed.
        """
        if not self.target or self._heuristic((self.x, self.z), self.target) < 4:
            self._generate_target()

        next_point = self._a_star_search()
        if not next_point:  # if you can't find a path, make new target and try again
            self.target = None
            return "special"

        # Translate point to action
        dx = next_point[0] - self.x
        dz = next_point[1] - self.z

        return self._get_relative_dir((dx, dz))

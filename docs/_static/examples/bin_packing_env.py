import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BinPacking2DEnv(gym.Env):
    def __init__(
        self,
        config: dict = {
            "bin_width": 10,
            "bin_length": 10,
            "bin_max_height": 10,
            "min_package_size": 1,
            "max_package_size": 3,
            "num_packages": 2,
            "episode_max": 100,
        },
    ):
        super().__init__()
        self.config = config
        # TODO: currently only 1 bin
        self.grid_size = (
            self.config["bin_width"],
            self.config["bin_length"],
        )  # 10x10 grid
        self.max_height = self.config["bin_max_height"]  # Maximum height in any column
        self.min_package_size = self.config["min_package_size"]
        self.max_package_size = self.config["max_package_size"]
        self.grid = np.zeros(self.grid_size, dtype=int)  # 2D grid of current heights
        self.NUM_PACKAGES = self.config["num_packages"]

        # Can place any package in any position in the box rotated in any way around z-axis
        # TODO: Change to any orientation. 6 possible permutations
        self.action_space = spaces.Discrete(
            2 * self.NUM_PACKAGES * self.grid_size[0] * self.grid_size[1]
        )

        # State consists of Height at every point in the grid, Size of each package to be placed, and size of bin
        # Size of box os agent can understand geometry of bin.
        # self.observation_space = spaces.Dict({
        #     'state': spaces.Box(
        #         np.array([0]*self.grid_size[0]*self.grid_size[1] + [self.min_package_size]*self.NUM_PACKAGES*3 + [0]*3),
        #         np.array([self.max_height]*self.grid_size[0]*self.grid_size[1] + [self.max_package_size]*self.NUM_PACKAGES*3 + [self.grid_size[0], self.grid_size[1], self.max_height]),
        #         dtype=int
        #     ),
        #     'action_mask': spaces.Box(
        #         np.array([0]*2*self.NUM_PACKAGES*self.grid_size[0]*self.grid_size[1]),
        #         np.array([1]*2*self.NUM_PACKAGES*self.grid_size[0]*self.grid_size[1]),
        #         dtype=int
        #     )  # Action mask (valid/invalid actions)
        # })

        self.observation_space = spaces.Box(
            np.array(
                [0] * self.grid_size[0] * self.grid_size[1]
                + [self.min_package_size] * self.NUM_PACKAGES * 3
                + [0] * 3
            ),
            np.array(
                [self.max_height] * self.grid_size[0] * self.grid_size[1]
                + [self.max_package_size] * self.NUM_PACKAGES * 3
                + [self.grid_size[0], self.grid_size[1], self.max_height]
            ),
            dtype=int,
        )

        self.step_count = 0

    def generate_packages(self):
        # Random packages with (width, height) between 1x1 and 2x2
        # Width is 'vertical' and height is 'horizontal'
        return (
            np.random.randint(1, self.max_package_size + 1, size=(2,)),
            np.random.randint(1, self.max_package_size + 1),
        )

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.packages = [self.generate_packages() for _ in range(self.NUM_PACKAGES)]
        self.step_count = 0
        self.portion_filled = 0
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self):
        # Return the current state of the grid and next two packages
        flattened_grid = self.grid.flatten()
        next_packages = np.array(
            [
                [self.packages[i][0][0], self.packages[i][0][1], self.packages[i][1]]
                for i in range(len(self.packages))
            ]
        ).flatten()

        package_configs = [
            (package, ((package[0][1], package[0][0]), package[1]))
            for package in self.packages
        ]
        package_configs = [
            subpackage for package in package_configs for subpackage in package
        ]

        # valid_actions = np.zeros((self.NUM_PACKAGES * 2, self.grid_size[0], self.grid_size[1]))
        # for i, package in enumerate(package_configs):
        #     for y in range(self.grid_size[0]):
        #         for x in range(self.grid_size[1]):
        #             if self.is_valid_placement(package, (y,x)):
        #                 valid_actions[i, y, x] = np.int8(1)
        return np.array(
            list(flattened_grid)
            + list(next_packages)
            + [self.grid_size[0], self.grid_size[1], self.max_height]
        )

    def is_valid_placement(self, package, position):
        # Check if the package can be placed at the given (x, y) position
        (width, height), package_height = package
        y, x = position

        # Ensure package fits within grid bounds
        if y + width > self.grid_size[0] or x + height > self.grid_size[1]:
            return False

        # Ensure package doesn't exceed max height in any column
        if np.any(
            self.grid[y : y + width, x : x + height] + package_height > self.max_height
        ):
            return False

        # Check if at least 3 corners are supported by the current grid heights
        corners_supported = 0
        corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
        corner_heights = [self.grid[y + cy, x + cx] for cy, cx in corners]
        corner_heights = corner_heights - max(corner_heights)
        corners_supported = sum(1 for i in corner_heights if i == 0)

        return corners_supported >= 3

    def place_package(self, package, position):
        (width, height), package_height = package
        y, x = position

        # Increase the height in the grid where the package is placed
        self.grid[y : y + width, x : x + height] += package_height

    def step(self, action):
        package_index = action // (2 * (self.grid_size[0] * self.grid_size[1]))
        orientation = (
            action - (2 * package_index * self.grid_size[0] * self.grid_size[1])
        ) // (self.grid_size[0] * self.grid_size[1])
        x = (
            action
            - (
                (2 * package_index + orientation)
                * self.grid_size[0]
                * self.grid_size[1]
            )
        ) % self.grid_size[1]
        y = (
            action
            - (
                (2 * package_index + orientation)
                * self.grid_size[0]
                * self.grid_size[1]
            )
        ) // self.grid_size[1]
        position = (y, x)

        # Get the chosen package
        if orientation == 0:
            package = self.packages[package_index]
        elif orientation == 1:
            package = (
                np.array(
                    [
                        self.packages[package_index][0][1],
                        self.packages[package_index][0][0],
                    ]
                ),
                self.packages[package_index][1],
            )
        else:
            print(package_index, orientation)
            raise ValueError

        # Check if placement is valid
        if self.is_valid_placement(package, position):
            self.place_package(package, position)
            self.step_count += 1
            done = self.step_count >= self.config["episode_max"]

            self.packages.pop(package_index)
            self.packages += [(self.generate_packages())]
            obs = self._get_obs()
            # TODO: Add reward based on how full/empty at end?
            # if sum(obs['action_mask']) == 0:
            # done = True

            new_portion_filled = sum(sum(self.grid)) / (
                self.grid_size[0] * self.grid_size[1] * self.max_height
            )

            reward = self._get_reward(new_portion_filled, obs, package, x, y)

            self.portion_filled = new_portion_filled
        else:
            reward = -10
            obs = self._get_obs()
            done = True

        info = self._get_info()

        return obs, reward, done, False, info

    def _get_reward(self, new_portion_filled, obs, package, x, y):
        filled_this_step = new_portion_filled - self.portion_filled
        # package_areas = [(self.packages[i][0][0]*self.packages[i][0][1]) for i in range(len(self.packages))]

        reward = ((self.step_count**1.3) * filled_this_step) ** 2
        # reward += (sum(obs['action_mask']) - (2 * np.sum(self.grid == 0)))*(sum(package_areas)/100)

        # # Add reward for having lots of grid spaces at similar height
        # same_height = 0
        # for i in range(1, self.config['bin_max_height']):
        #     single_same_height = np.sum(self.grid == i)
        #     single_same_height = (single_same_height ** 2)/np.sqrt(self.grid_size[0]*self.grid_size[1])
        #     same_height == single_same_height
        # reward += same_height

        # Reward for placing next to something bigger
        # same_above = [(self.grid[y-1, length] <= self.grid[y, length]) if (y-1 >= 0) else 1 for length in range(x, x+package[0][1])]
        # same_below = [(self.grid[y+package[0][0]-1, length] <= self.grid[y+package[0][0], length]) if (y+package[0][0]+1 <= self.config['bin_length']) else 1 for length in range(x, x+package[0][1])]
        # same_left = [(self.grid[height, x] >= self.grid[height, x-1]) if (x-1 >= 0) else 1 for height in range(y, y+package[0][0])]
        # same_right = [(self.grid[height, x+package[0][1]-1] <= self.grid[height, x+package[0][1]]) if (x+package[0][1]+1 <= self.config['bin_width']) else 1 for height in range(y, y+package[0][0])]

        # Reward for placing next to same size depending on size
        # same_above = [((self.max_height - np.abs(self.grid[y-1, length] - self.grid[y, length]))/self.max_height) if (y-1 >= 0) else 0.7 for length in range(x, x+package[0][1])]
        # same_below = [((self.max_height - np.abs(self.grid[y+package[0][0]-1, length] - self.grid[y+package[0][0], length]))/self.max_height) if (y+package[0][0]+1 <= self.config['bin_length']) else 0.7 for length in range(x, x+package[0][1])]
        # same_left = [((self.max_height - np.abs(self.grid[height, x] - self.grid[height, x-1]))/self.max_height) if (x-1 >= 0) else 0.7 for height in range(y, y+package[0][0])]
        # same_right = [((self.max_height - np.abs(self.grid[height, x+package[0][1]-1] == self.grid[height, x+package[0][1]]))/self.max_height) if (x+package[0][1]+1 <= self.config['bin_width']) else 0.7 for height in range(y, y+package[0][0])]

        # Reward for placing next to something bigger
        same_above = [
            (self.grid[y - 1, length] == self.grid[y, length]) if (y - 1 >= 0) else 1
            for length in range(x, x + package[0][1])
        ]
        same_below = [
            (
                (
                    self.grid[y + package[0][0] - 1, length]
                    == self.grid[y + package[0][0], length]
                )
                if (y + package[0][0] + 1 <= self.config["bin_length"])
                else 1
            )
            for length in range(x, x + package[0][1])
        ]
        same_left = [
            (self.grid[height, x] == self.grid[height, x - 1]) if (x - 1 >= 0) else 1
            for height in range(y, y + package[0][0])
        ]
        same_right = [
            (
                (
                    self.grid[height, x + package[0][1] - 1]
                    == self.grid[height, x + package[0][1]]
                )
                if (x + package[0][1] + 1 <= self.config["bin_width"])
                else 1
            )
            for height in range(y, y + package[0][0])
        ]
        same_reward = (
            sum(same_above) + sum(same_below) + sum(same_left) + sum(same_right)
        )
        if same_reward == 0:
            reward -= 10
        else:
            reward += same_reward

        if self.step_count == 100:
            reward += 50

        if self.portion_filled >= 0.55:
            reward += 3 * self.portion_filled

        return reward

    def _get_info(self):
        info_dict = {"portion_filled": self.portion_filled}

        return info_dict

    # def render(self):
    #     # Visualize the current grid state
    #     # Using ANSI escape sequences to color-code grid based on height levels
    #     max_value = np.max(self.grid) if np.max(self.grid) > 0 else 1
    #     grid_color = np.clip(self.grid / max_value, 0, 1)  # Normalize for colors

    #     # Map of colors (grayscale representation for height)
    #     def colorize(val, max_value):
    #         grayscale = int(232 + (val / max_value) * 23)
    #         return f"\033[48;5;{grayscale}m  \033[0m"  # Background color

    #     print("Current grid state:")
    #     for row in self.grid:
    #         for val in row:
    #             sys.stdout.write(colorize(val, max_value))
    #         print()  # Newline after each row
    #     print("\n")

    #     # Show package information
    #     print(f"Remaining packages to place: {self.packages}")


class BinPacking2DEnvCNN(BinPacking2DEnv, gym.Env):
    def __init__(
        self,
        config: dict = {
            "bin_width": 10,
            "bin_length": 10,
            "bin_max_height": 10,
            "min_package_size": 1,
            "max_package_size": 3,
            "num_packages": 2,
            "episode_max": 100,
        },
    ):
        super().__init__(config)

        self.observation_space = spaces.Dict(
            {
                "grid_state": spaces.Box(
                    0, self.max_height, shape=self.grid_size, dtype=int
                ),
                "package_state": spaces.Box(
                    np.array([self.min_package_size] * 3 * self.NUM_PACKAGES),
                    np.array([self.max_package_size] * 3 * self.NUM_PACKAGES),
                    dtype=int,
                ),
                "bin_size_state": spaces.Box(
                    np.array([0] * 3),
                    np.array([self.grid_size[0], self.grid_size[1], self.max_height]),
                    dtype=int,
                ),
                "action_mask": spaces.Box(
                    np.array(
                        [0]
                        * 2
                        * self.NUM_PACKAGES
                        * self.grid_size[0]
                        * self.grid_size[1]
                    ),
                    np.array(
                        [1]
                        * 2
                        * self.NUM_PACKAGES
                        * self.grid_size[0]
                        * self.grid_size[1]
                    ),
                    dtype=int,
                ),
            }
        )

    def _get_obs(self):
        # Return the current state of the grid and next two packages
        next_packages = np.array(
            [
                [self.packages[i][0][0], self.packages[i][0][1], self.packages[i][1]]
                for i in range(len(self.packages))
            ]
        ).flatten()

        package_configs = [
            (package, ((package[0][1], package[0][0]), package[1]))
            for package in self.packages
        ]
        package_configs = [
            subpackage for package in package_configs for subpackage in package
        ]

        valid_actions = np.zeros(
            (self.NUM_PACKAGES * 2, self.grid_size[0], self.grid_size[1])
        )
        for i, package in enumerate(package_configs):
            for y in range(self.grid_size[0]):
                for x in range(self.grid_size[1]):
                    if self.is_valid_placement(package, (y, x)):
                        valid_actions[i, y, x] = np.int8(1)

        return {
            "grid_state": self.grid,
            "package_state": next_packages,
            "bin_size_state": np.array(
                [self.grid_size[0], self.grid_size[1], self.max_height]
            ),
            "action_mask": valid_actions.flatten().flatten(),
        }

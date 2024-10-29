import math
import json

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame

from gymnasium import spaces
from scipy.stats import norm

from itomori.envs.utils import euclidean_distance
from itomori.envs.configs import default_uav_args


class Navigation2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self, 
        render_mode='human', 
        map_filepath="itomori/navigation2d_config/maps/10x10/10x10_NoObstacle.json",
        uav_args=default_uav_args,
        reward_weights={
            'distance': 0.9,
            'steps': 0.1,
            'risk': 0,
        }
    ):
        """Path Planning Environment
        
        Args:
            render_mode (str): Rendering mode, can be either "human" or "rgb_array"
            map_filepath (str): Filepath to the map json file
            uav_filepath (str): Filepath to the UAV json file
       """
        super(Navigation2DEnv, self).__init__()

        assert render_mode in self.metadata["render_modes"], f"Invalid render_mode: {render_mode}"
        self.render_mode = render_mode

        assert sum(reward_weights.values()) == 1, "Rewards weights should sum to 1"
        self.reward_weights = reward_weights

        # Game params
        self.destination_tolerance = 1.5  # Tolerance radius for reaching the target (m)
        self.window_size = 768  # Size of the window in pixels
        self.render_potential_field_resolution = 0.05  # Resolution of the potential field when calculating it for rendering

        # Read map and uav
        self.load_uav_args(uav_args)
        self.read_map(map_filepath)


        self.starting_agent_location = np.array(
            [
                self.origin[0], 
                self.origin[1], 
                self.velocity_initial, 
                self.angle_initial
            ]
        ).astype(float)

        """ Observation Space
        The observation space is a Dict with the following structure:
        observation = {
            "origin": [x, y],
            "destination": [x, y],
            "obstacles": ,
            "agent_location": [x, y, v, a, risk],
              x = float: x-coordinate of the agent
              y = float: y-coordinate of the agent
              v = float: velocity of the agent
              a = float: angle of the agent
              risk = float: risk of the agent
            "agent_location_history": [agent_location_1, agent_location_2, ..., agent_location_n]
        }
        """
        self.observation_space = spaces.Dict({
            "origin": spaces.Box(
                low = np.array([0, 0]),
                high = np.array([50, 50]),
                shape=(2,),
                dtype = float
            ),  # x, y
            "destination": spaces.Box(
                low = np.array([0, 0]),
                high = np.array([self.size, self.size]),
                shape=(2,),
                dtype = float
            ),  # x, y
            "agent_location": spaces.Box(
                low = np.array([0, 0, self.velocity_min, -np.inf, 0, 0]),
                high = np.array([self.size, self.size, self.velocity_max, np.inf, np.inf, np.inf]),
                shape=(6,),
                dtype = float
            ),  # x, y, v, a, risk, gps_uncertainty
        })
        
        """ Action Space
        Action Space is a Box(2,) with the following constraints:
        action = [vc, ac]
            vc = float: velocity change (paper: a, acceleration)
            ac = float: angle change (paper: e, angle variation)
        """
        self.action_space = spaces.Box(
            low = np.array([self.angle_change_min, self.velocity_change_min]),
            high = np.array([self.angle_change_max, self.velocity_change_max]),
            shape=(2,),
            dtype = float
        )

        """ Rendering
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        # Save potential field in memory to avoid recalculating it every frame
        self.surface_potential_field = None

    def load_uav_args(self, uav_args):
        '''Get UAV parameters from dictionary
        
        Args:
            uav_args (dict): UAV parameters

        Dict structure:
            velocity_min (float): Minimum velocity of the agent (m/s)
            velocity_max (float): Maximum velocity of the agent (m/s)
            angle_change_min (float): Minimum angle change variation of the agent (degrees/s)
            angle_change_max (float): Maximum angle change variation of the agent (degrees/s)
            velocity_change_min (float): Minimum velocity change, acceleration of the agent (m/s**2)
            velocity_change_max (float): Maximum velocity change, acceleration of the agent (m/s**2)
            delta_T (float): Time it takes from one waypoint to the next (s)
            gps_uncertainty (float): GPS imprecision (m)
        '''

        self.velocity_initial = uav_args['velocity_initial']
        self.velocity_min = uav_args['velocity_min']
        self.velocity_max = uav_args['velocity_max']
        self.velocity_change_min = uav_args['velocity_change_min']
        self.velocity_change_max = uav_args['velocity_change_max']
        self.angle_initial = uav_args['angle_initial']
        self.angle_change_min = uav_args['angle_change_min']
        self.angle_change_max = uav_args['angle_change_max']
        self.delta_T = uav_args['delta_T']
        self.gps_uncertainty_initial = uav_args['gps_uncertainty_initial']
        self.gps_uncertainty = self.gps_uncertainty_initial
        self.gps_uncertainty_increment = uav_args['gps_uncertainty_increment']
      
    def read_map(self, map_filepath):
        '''Read map from json file
        
        Args:
            map_filepath (str): Filepath to the map json file

        Json file structure:
            size (int): Size of the map (m)
            origin (list): Origin of the map [x, y]
            destination (list): Destination of the map [x, y]
            obstacles (list): List of obstacles in the map [[x1, y1], [x2, y2], ...]
        '''

        with open(map_filepath) as f:
            map_data = json.load(f)

        self.size = map_data['size']
        self.origin = np.array(map_data['origin']).astype(float)
        self.destination = np.array(map_data['destination']).astype(float)
        self.obstacles = np.array(map_data['obstacles']).astype(float)
        self.n_obstacles = self.obstacles.shape[0]
           
    def _get_obs(self):
        return {
            "origin": self.origin,
            "destination": self.destination,
            "agent_location": self._agent_location,
        }

    def _get_info(self):
        return {
            "risk": self._agent_location[4],
            "failed": self.is_failed(),
            "destination_tolerance": self.destination_tolerance,
            "agent_location_history": self.agent_location_history,
            "obstacles": self.obstacles,
        }

    def is_failed(self):
        '''Check if the agent entered a failure state'''
        if self.is_outofbounds(self._agent_location[:2]):
            return True
        if self.is_hit(self._agent_location[:2], self.previous_agent_location):
            return True
        return False
    
    def is_done(self):
        '''Check if the agent has reached the target within the destination tolerance.'''

        dist = euclidean_distance(self._agent_location[:2], self.destination)
        if dist < self.destination_tolerance:
            return True

        elif self.is_failed():
            return True

        return False        

    def is_outofbounds(self, location):
        return any(location <= 0) or any(location >= self.size)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.timestep = 0

        # Set agent location to origin
        self.previous_agent_location = None
        self.risk = self.chance_constraint(self.starting_agent_location[:2])
        self._agent_location = np.append(self.starting_agent_location, [self.risk, self.gps_uncertainty])
        # self._agent_location = self.starting_agent_location.append(self.risk)
        self.agent_location_history = [self._agent_location]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info

    def step(self, action):

        self.timestep += 1

        # TODO: decresce com base no timestep
        self.gps_uncertainty += self.gps_uncertainty_increment


        # Calculate new agent location
        self.previous_agent_location = self._agent_location
        _agent_location = self._move(action)
        self.risk = self.chance_constraint(_agent_location[:2])
        self._agent_location = np.append(_agent_location, [self.risk, self.gps_uncertainty])
        self.agent_location_history.append(self._agent_location)

        observation = self._get_obs()
        reward = self.reward()
        terminated = self.is_done()
        truncated = False
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    
    def _move_old(self, action):
        ''''
        vc := velocity change
        ac := angle change
        
        x, y := current x, y agent's position
        v, a := current velocity and angle of the agent

        delta_T := time it takes from one waypoint to the next (s)

        xn := x next
        yn := y next
        vn := v next
        an := a next

        TODO: add references
        Equações descritas em (Arantes 2016) - adaptações por Claudio (jan/20)

        '''
        ac, vc = action
        x, y, v, a = self._agent_location[:4]
        delta_T = self.delta_T

        # Math: xn = x + (v * cos(a) * \Delta_T) + (vc * cos(a) * (\frac{\Delta_T^2}{2}))
        # Update position
        xn = x + (v * np.cos(a) * delta_T) + (vc * np.cos(a) * ((delta_T ** 2) / 2))
        xn = np.clip(xn, 0, self.size)

        yn = y + (v * np.sin(a) * delta_T) + (vc * np.sin(a) * ((delta_T ** 2) / 2))
        yn = np.clip(yn, 0, self.size)

        # Update velocity
        vn = v + (vc * delta_T)  # - ((F / m) * delta_T)
        vn = np.clip(vn, self.velocity_min, self.velocity_max)

        # Update angle
        an = a + (ac * delta_T)
        an = np.clip(an, self.angle_change_min, self.angle_change_max)

        new_location = np.array([xn, yn, vn, an])

        return new_location
    
    def _move(self, action):
        ''''
        vc := velocity change
        ac := angle change
        
        x, y := current x, y agent's position
        v, a := current velocity and angle of the agent

        delta_T := time it takes from one waypoint to the next (s)

        xn := x next
        yn := y next
        vn := v next
        an := a next

        TODO: add references
        Equações descritas em (Arantes 2016) - adaptações por Claudio (jan/20)

        '''
        ac, vc = action
        x, y, v, a = self._agent_location[:4]
        delta_T = self.delta_T

        # Update velocity
        vn = v + (vc * delta_T)  # - ((F / m) * delta_T)
        vn = np.clip(vn, self.velocity_min, self.velocity_max)

        # Update angle
        an = a + (ac * delta_T)
        an = np.clip(an, self.angle_change_min, self.angle_change_max)

        # Math: xn = x + (v * cos(a) * \Delta_T) + (vc * cos(a) * (\frac{\Delta_T^2}{2}))
        # Update position
        xn = x + (vn * np.cos(an) * delta_T) + (vc * np.cos(an) * ((delta_T ** 2) / 2))
        xn = np.clip(xn, 0, self.size)
        
        yn = y + (vn * np.sin(an) * delta_T) + (vc * np.sin(an) * ((delta_T ** 2) / 2))
        yn = np.clip(yn, 0, self.size)

        new_location = np.array([xn, yn, vn, an])

        return new_location
      
    def reward(self):
        '''Calculate the reward based on the distance to the target.'''
        if self.is_failed():
            return math.inf
        
        # Distance from agent to destination
        distance = euclidean_distance(self._agent_location[:2], self.destination)

        # Route length
        steps = len(self.agent_location_history)

        # Risk
        risk = self._agent_location[4]

        reward = (
            + self.reward_weights['distance'] * (distance) 
            + self.reward_weights['steps'] * (steps)
            + self.reward_weights['risk'] * (risk)
        )

        return reward

    def render(self, mode=None, **kwargs):
        if self.render_mode == "rgb_array":
            return self._render_frame(mode=mode, **kwargs)
        if mode == 'human':
            self._render_frame(mode=mode, **kwargs)

    def _render_frame(self, mode=None, **kwargs):
        ''''
        Render the current state of the environment.

        Args:
            mode (str): Rendering mode, can be either "human" or "rgb_array"
            kwargs (dict): Additional arguments to be passed to the rendering function
                Keys:
                'plot_potential_field' (bool): If True, plot the potential field of the environment
        '''
        window_sizes = (self.window_size, self.window_size)
        if ((self.window is None and self.render_mode == "human")
            or (self.window is None and mode == "human")):
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(window_sizes)
        if ((self.clock is None and self.render_mode == "human")
            or (self.clock is None and mode == "human")):
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(window_sizes)
        canvas.fill((255, 255, 255)) # White

        pix_square_size = (self.window_size / self.size)  # The size of a single grid square in pixels

        # --

        # Plot the potential field
        # only if kwargs contains the key 'plot_potential_field' and its value is True
        if 'plot_potential_field' in kwargs and kwargs['plot_potential_field']:
            if self.surface_potential_field is None:
                # Calculate the potential field
                _, potential_field = self._calculate_potential_field()
                # Normalize the potential field
                potential_field = 1 - potential_field
                # Scale the potential field to the range [0, 255]
                potential_field = potential_field * 255
                # Convert the potential field to a 3-channel image
                potential_field = np.stack((potential_field,)*3, axis=-1)
                # Set the first channel to 255 (white) to highlight the obstacles
                potential_field[:, :, 0] = 255
                # Swap the axes to match the coordinate system of the environment
                potential_field = potential_field.swapaxes(0, 1)
                # Create a Pygame surface from the potential field
                image_surface = pygame.surfarray.make_surface(potential_field)
                # Resize the image to fit the window
                surface_potential_field = pygame.transform.scale(image_surface, window_sizes)
                self.surface_potential_field = surface_potential_field
                    
            canvas.blit(self.surface_potential_field, (0, 0))

        # --

        # Add gridlines
        for x in range(self.size + 1):
            
            color_grey = (240, 240, 240) # Light grey
            if x % 10 == 0:
                color_grey = (210, 210, 210) # Grey

            pygame.draw.line(
                canvas,
                color_grey,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                color_grey,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )


        # Draw the destination point
        pygame.draw.circle(
            canvas,
            pygame.Color(209,187,215),  # Light pink
            self.destination * pix_square_size,
            self.destination_tolerance * pix_square_size,
        )
        pygame.draw.circle(
            canvas,
            (136,46,114),  # Red
            self.destination * pix_square_size,
            pix_square_size / 8,
        )

        # Draw the obstacles
        for obstacle in self.obstacles:
            pygame.draw.polygon(
                canvas,
                (100, 100, 100),  # Dark grey
                obstacle * pix_square_size,
            )

        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self._agent_location[:2] * pix_square_size,
            pix_square_size / 10,
        )

        # Draw the agent's path
        pygame.font.init()
        font = pygame.font.SysFont('timesnewroman', 11)

        def _draw_route(agent_location_history, color, add_text=False):
            for i in range(1, len(agent_location_history)):
                pygame.draw.line(
                    canvas,
                    color,
                    agent_location_history[i - 1][:2] * pix_square_size,
                    agent_location_history[i][:2] * pix_square_size,
                    width=2,
                )
                # add an dot at each point
                pygame.draw.circle(
                    canvas,
                    color,
                    agent_location_history[i][:2] * pix_square_size,
                    pix_square_size / 10,
                )

                if 'plot_gps_uncertainty' in kwargs and kwargs['plot_gps_uncertainty']:
                    gps_uncertainty = agent_location_history[i][5]

                    # add a circle around the dot to represent the gps uncertainty
                    pygame.draw.circle(
                        canvas,
                        color,
                        agent_location_history[i][:2] * pix_square_size,
                        gps_uncertainty * pix_square_size,
                        width=1,
                    )

                    # add a text with the gps_uncertainty to each point
                    if add_text:
                        text = font.render(f'{round(gps_uncertainty, 5)}', True, (0, 0, 0))
                        offset = gps_uncertainty/2
                        pos = (agent_location_history[i][:2] + offset) * pix_square_size
                        canvas.blit(text, pos)
            
                # add a text with the order, risk and reward to each point
                if add_text:
                    risk = agent_location_history[i][4]
                    text = font.render(f'{i}|{round(risk,2)}', True, (0, 0, 0))
                    pos = agent_location_history[i][:2] * pix_square_size
                    canvas.blit(text, pos)
            
        
        color = (25,101,176) # Blue - Discrete rainbow no 10
        agent_location_history = self.agent_location_history
        _draw_route(agent_location_history, color, add_text=True)
        
        if 'extra_routes' in kwargs:
            for extra_route in kwargs['extra_routes']:
                agent_location_history = extra_route['agent_location_history']
                color = extra_route['color'] if 'color' in extra_route else (174, 118, 163)
                _draw_route(agent_location_history, color, add_text=extra_route.get('add_text', False))

        # Draw agent's initial position
        pygame.draw.circle(
            canvas,
            (255, 50, 50),
            self.starting_agent_location[:2] * pix_square_size,
            pix_square_size / 7,
        )

        if self.render_mode == "human" or mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    
    # ----------------------------------------------------------

    # --- Chance Constraint

    def _calculate_potential_field(self):
        '''Calculate the potential field for the agent.'''
        
        # Mathematical function we need to plot
        @np.vectorize
        def z_func(x, y):
            chance = self.chance_constraint(np.array([x, y]))
            return chance

        # Setting up input values
        x = np.arange(0, self.size, self.render_potential_field_resolution )
        y = np.arange(0, self.size, self.render_potential_field_resolution )
        X, Y = np.meshgrid(x, y)
        
        # Calculating the output and storing it in the array Z
        Z = z_func(X, Y)

        # Plotting the potential field
        # im = plt.imshow(Z, cmap=plt.cm.Purples, extent=(0, self.size, 0, self.size), interpolation='bilinear', origin='lower')
        # plt.colorbar(im)
        # # Plot the obstacles limits
        # for obs in self.obstacles:
        #     obs = np.append(obs, [obs[0]], axis=0)
        #     plt.plot(obs[:, 0], obs[:, 1], 'k-')
        # # plt.title('Chance Constraint over a No-Fly Zone')
        # # plt.show()

        return None, Z


    def chance_constraint(self, P, obstacles=None):

        def distance_point_line(P, A, B, return_normal=False):
            '''Calculates the distance between the point P and the line that crosses the points A and B'''
            
            # Director vector of line
            # D = Vector((A.x - B.x), (A.y - B.y))
            D = A - B

            # Normal vector of line
            # N = Vector(D.y, -D.x)
            N = np.array([D[1], -D[0]])
            
            # Normalized normal vector of line AB
            # aux = (math.sqrt(D.y**2 + D.x**2))
            # N = Vector( (D.y / aux), (-D.x / aux) )
            N = N / np.linalg.norm(D)

            # b = N.x * A.x + N.y * A.y
            b = np.dot(N, A)

            # distance = P.x * N.x + P.y * N.y - b
            distance = np.dot(P, N) - b

            if return_normal:
                return distance, N
            
            return distance

        def prob_collision(distance, gps_uncertainty):
            # Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html

            # mi    : média         : location (loc)
            # sigma : desvio padrão : scale

            return norm.sf(distance, loc=0, scale=gps_uncertainty)
        
        obstacles = self.obstacles if obstacles is None else obstacles

        chance = 0
        for obs in obstacles:
            distances = []
            n = len(obs)
            for i in range(n):
                A = obs[i]
                B = obs[(i + 1) % n]
                distance = distance_point_line(P, A, B)
                distances.append(distance)

            distances = np.array(distances)
            
            # Calculate the maximum distance
            max_distance = np.max(distances)
        
            chance += prob_collision(max_distance, self.gps_uncertainty)

        return chance


    def is_hit(self, P, O=None, obstacles=None):
        '''Check if the point P has hit an obstacle.
        Check if the line defined by the point P and it's previous point O is 
        intersecting any of the obstacles.
        '''
        
        def is_inside_polygon(P, obs):
            # Calculates if the point P is inside the polygon defined by the obstacles
            n = len(obs)
            inside = False
            for i in range(n):
                A = obs[i]
                B = obs[(i + 1) % n]
                if A[1] <= P[1] <= B[1] or B[1] <= P[1] <= A[1]:
                    if P[0] <= (B[0] - A[0]) * (P[1] - A[1]) / (B[1] - A[1]) + A[0]:
                        inside = not inside
            return inside
        
        def is_intersecting_line(P, O, obs):
            '''Calculates if the line defined by P and it's previous point O is 
            intersecting the obstacles.
            '''

            n = len(obs)
            for i in range(n):
                A = obs[i]
                B = obs[(i + 1) % n]
                C = O[:2]
                D = P

                # Check if the lines intersect
                # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
                def ccw(A, B, C):
                    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

                def intersect(A, B, C, D):
                    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

                if intersect(A, B, C, D):
                    return True

        obstacles = self.obstacles if obstacles is None else obstacles
        for obs in obstacles:
            if is_inside_polygon(P, obs):
                return True
            if O is not None and is_intersecting_line(P, O, obs):
                return True
        return False


class DiscreteNavigation2DEnv(Navigation2DEnv):
    def __init__(self, diagonal=False, *args, **kwargs):
        """Discrete Navigation 2D Environment
        
        Args:
            diagonal (bool): If True, allow diagonal movements, else 
                only allow horizontal and vertical movements
        """

        super(DiscreteNavigation2DEnv, self).__init__(*args, **kwargs)

        self.diagonal = diagonal

        if self.diagonal:
            self.action_space = spaces.Discrete(8)
            self._action_to_direction = {
                0: np.array([1, 0]),  # right
                1: np.array([1, 1]),  # right-down
                2: np.array([0, 1]),  # down
                3: np.array([-1, 1]), # left-down
                4: np.array([-1, 0]), # left
                5: np.array([-1, -1]),# left-up
                6: np.array([0, -1]), # up
                7: np.array([1, -1]), # right-up
            }
        else:
            self.action_space = spaces.Discrete(4)
            self._action_to_direction = {
                0: np.array([1, 0]),  # right
                1: np.array([0, 1]),  # down
                2: np.array([-1, 0]), # left
                3: np.array([0, -1]), # up
            }
  

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = self._agent_location[:2] + direction
        self.agent_location_history.append(self._agent_location)
        
        observation = self._get_obs()
        reward = self.reward()
        terminated = self.is_done()
        truncated = False
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info   


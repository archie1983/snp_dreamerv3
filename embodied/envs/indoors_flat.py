import logging, threading, elements, random, embodied, traceback #, time
import numpy as np
from ai2_thor_model_training.training_data_extraction import RobotNavigationControl
from ai2_thor_model_training.ae_utils import (NavigationUtils, action_mapping, euclidean_dist,
                                              action_to_index, index_to_action, inverted_action_mapping,
                                              AI2THORUtils, get_path_length, get_centre_of_the_room,
                                              room_this_point_belongs_to, get_rooms_ground_truth,
                                              get_all_objects_of_type, is_point_inside_room_ground_truth,
                                              create_full_grid_from_room_layout, add_buffer_to_unreachable,
                                              RoomType, BoundaryCalculations)

import thortils as tt
from thortils import launch_controller
from thortils.agent import thor_reachable_positions
from thortils.utils import roundany, getch
from thortils.utils.math import sep_spatial_sample

from shapely.geometry import Point
import pickle, json

np.float = float
np.int = int
np.bool = bool

class Roomcentre(embodied.Wrapper):

    def __init__(self, *args, **kwargs):
        self.logdir = kwargs["logdir"]
        actions = action_mapping
        reward_close_enough = kwargs["reward_close_enough"]

        # Actions
        actions = actions.copy()
        #if "STOP" in actions:
        #    actions.pop("STOP")  # remove STOP action because that will be treated differently

        self.rewards = [
            DistanceReductionReward(),
            TargetAchievedRewardRoomCentre(epsilon=reward_close_enough),
        ]
        length = kwargs.pop('length', 36000)
        env = RoomCentreFinder(actions, *args, **kwargs)
        self.unwrapped_env = env
        env = embodied.wrappers.TimeLimit(env, length)
        print("AE: TimeWrapped: ", env)
        super().__init__(env)

    def step(self, action):
        env_res = self.env.step(action, add_extra = True)
        #print("AE: env_res: ", env_res)
        obs, extra_obs = env_res
        #obs, extra_obs = self.env.step(action, add_extra = True)

        # We will use choose_habitats_randomly_or_sequentially flag for determining whether we are in training mode (True)
        # and need to calculate reward, or we're in testing mode (False) and reward should be set to 0.
        if self.unwrapped_env.choose_habitats_randomly_or_sequentially:
            reward = sum([fn(obs, extra_obs, action) for fn in self.rewards])
        else:
            reward = 0.0

        obs['reward'] = np.float32(reward)

        if obs['is_last'] and not self.unwrapped_env.env_retired:# and self.unwrapped_env.hab_set != "train":
            episode_stats = {
                "final_reward": str(obs['reward']),
            }
            with open(self.logdir + "/episode_data.jsonl", "a") as f:
                f.write(json.dumps(episode_stats) + "\n")

        # we may not want to train on distance_left parameter, but if we pop it, then wrappers complain,
        # so perhaps it can stay for now.
        #obs.pop("distance_left")
        return obs

class Door(embodied.Wrapper):

    def __init__(self, *args, **kwargs):
        self.logdir = kwargs["logdir"]
        #print("DI1")
        actions = action_mapping
        #print("*args: ", args, " **kwargs: ", kwargs)
        reward_close_enough = kwargs["reward_close_enough"]

        # Actions
        actions = actions.copy()
        #if "STOP" in actions:
        #    actions.pop("STOP")  # remove STOP action because that will be treated differently

        self.rewards = [
            DistanceReductionReward(scale=1.0),
            StepCountPenalizer(scale=1.0),
            TargetAchievedRewardForDoor(epsilon=reward_close_enough)
        ]
        length = kwargs.pop('length', 36000)
        #print("AE: len", length)
        env = DoorFinder(actions, *args, **kwargs)
        self.unwrapped_env = env
        env = embodied.wrappers.TimeLimit(env, length)
        super().__init__(env)
        #print("DI2")

    def step(self, action):
        #print("A1")
        #t1 = time.time()
        obs, extra_obs = self.env.step(action, add_extra=True)
        #t2 = time.time()

        # We will use choose_habitats_randomly_or_sequentially flag for determining whether we are in training mode (True)
        # and need to calculate reward, or we're in testing mode (False) and reward should be set to 0.
        if self.unwrapped_env.choose_habitats_randomly_or_sequentially:
            reward = sum([fn(obs, extra_obs, action) for fn in self.rewards])
        else:
            reward = 0.0

        obs['reward'] = np.float32(reward)
        #print("A2")

        if obs['is_last'] and not self.unwrapped_env.env_retired:# and self.unwrapped_env.hab_set != "train":
            episode_stats = {
                "final_reward": str(obs['reward']),
            }
            with open(self.logdir + "/episode_data.jsonl", "a") as f:
                f.write(json.dumps(episode_stats) + "\n")

        # we may not want to train on distance_left parameter, but if we pop it, then wrappers complain,
        # so perhaps it can stay for now.
        #obs.pop("distance_left")
        #t3 = time.time()
        #print("AE: Tt = ", round(t3-t1, 4), " Td = ", round(t2-t1, 4))
        return obs

        # Introduce a marker on the image that points towards the door that we want to go to. That would allow input and
        # training guidance to navigate to a specific door, not just a random door. Introduce room field in observation so that we can
        # classify target achieved when we change rooms.

class Perimeter(embodied.Wrapper):

    def __init__(self, *args, **kwargs):
        self.logdir = kwargs["logdir"]
        actions = action_mapping
        reward_close_enough = kwargs["reward_close_enough"]

        # Actions
        actions = actions.copy()
        #if "STOP" in actions:
        #    actions.pop("STOP")  # remove STOP action because that will be treated differently

        self.rewards = [
            DistanceReductionRewardForPerimeter(),
            TargetAchievedRewardForPerimeter(),
            StepCountPenalizerForPerimeter()
        ]
        length = kwargs.pop('length', 36000)
        env = PerimeterFinder(actions, *args, **kwargs)
        self.unwrapped_env = env
        env = embodied.wrappers.TimeLimit(env, length)
        print("AE: TimeWrapped: ", env)
        super().__init__(env)

    def step(self, action):
        env_res = self.env.step(action, add_extra = True)
        #print("AE: env_res: ", env_res)
        obs, extra_obs = env_res
        #obs, extra_obs = self.env.step(action, add_extra = True)

        # We will use choose_habitats_randomly_or_sequentially flag for determining whether we are in training mode (True)
        # and need to calculate reward, or we're in testing mode (False) and reward should be set to 0.
        if self.unwrapped_env.choose_habitats_randomly_or_sequentially:
            reward = sum([fn(obs, extra_obs, action) for fn in self.rewards])
        else:
            reward = 0.0

        obs['reward'] = np.float32(reward)

        if obs['is_last'] and not self.unwrapped_env.env_retired:# and self.unwrapped_env.hab_set != "train":
            episode_stats = {
                "final_reward": str(obs['reward']),
            }
            with open(self.logdir + "/episode_data.jsonl", "a") as f:
                f.write(json.dumps(episode_stats) + "\n")

        # we may not want to train on distance_left parameter, but if we pop it, then wrappers complain,
        # so perhaps it can stay for now.
        #obs.pop("distance_left")
        return obs

##
# Using this class, we can stack objectives of the agent behaviour. E.g., to first achieve the
# middle of the room and only then look for the doors. Or even find all doors in order.
##
class DistanceReductionReward:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.prev_distance = None
        self.best_distance_so_far = None

    def __call__(self, obs, extra_obs, action):
        #print("D1")
        reward = 0.0
        #distance_left = obs['distance_left']
        distance_left = extra_obs['distanceleft']

        if obs['is_first']:
            self.best_distance_so_far = distance_left
        else:
            if self.best_distance_so_far > distance_left:
                '''
                if we improved best distance, then reward is the improvement factor
                '''
                #reward = self.scale * (self.best_distance_so_far - distance_left)
                reward = 1
                self.best_distance_so_far = distance_left
                #print("BIG REW: ", reward)
                #print("r", reward, end="", sep="")
                print("r", end="", sep="")
            # elif self.best_distance_so_far == distance_left:
            #     '''
            #     if no improvement, then bigger penalty. No movement needs to be discouraged
            #     '''
            #     reward = -0.25
            elif self.best_distance_so_far < distance_left and self.prev_distance < distance_left:
                '''
                if we have moved away from the target, then penalty by the reduction
                '''
                #reward = self.scale * (self.prev_distance - distance_left)
                reward = -0.5
            elif self.best_distance_so_far <= distance_left and self.prev_distance > distance_left:
                '''
                if we have improved our position from last time, but not yet the best path, then small reward
                '''
                reward = 0.25
            elif self.best_distance_so_far <= distance_left and self.prev_distance == distance_left:
                '''
                if no improvement since last time, then penalty to discourage not moving
                '''
                reward = -0.25
            else:
                '''
                shouldn't happen. If it does, then the above code has error.
                '''
                print("CHECK DistanceReductionReward CODE!!!")
                exit()

        self.prev_distance = distance_left
        #print("D2")

        return np.float32(reward)

class DistanceReductionRewardForPerimeter(DistanceReductionReward):
    def __call__(self, obs, extra_obs, action):
        #print("extra_obs: ", extra_obs)
        if not extra_obs['first_point_reached']:
            return super().__call__(obs, extra_obs, action)
        else:
            return np.float32(0.0)

##
# Penalize each step with -1 after the initial nearest length of the steps for a door.
# The fewer the steps, the better final reward.
##
class StepCountPenalizer:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.steps_done = 0

    def __call__(self, obs, extra_obs, action):
        reward = 0.0
        if obs['is_first']:
            self.steps_done = 0
        else:
            self.steps_done += 1

        #if extra_obs['all_target_dists_initial'] is not None and len(extra_obs['all_target_dists_initial']) > 0 and self.steps_done > 10 * np.max(extra_obs['all_target_dists_initial']):
        #    reward = -0.25
        #if self.steps_done > extra_obs['initial_distance']:
        #    reward = -0.1
        if self.steps_done > 2 * extra_obs['best_path_length']:
            reward = -0.25

        return np.float32(reward * self.scale)

##
# Penalize each step after the initial nearest length of the path for first boundary point.
# The fewer the steps, the better final reward.
##
class StepCountPenalizerForPerimeter:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.steps_done = 0
        self.first_point_visited = False
        self.initial_number_of_boundary_points = 0

    def __call__(self, obs, extra_obs, action):
        reward = 0.0
        if obs['is_first']:
            self.steps_done = 0
            self.first_point_visited = False
            self.initial_number_of_boundary_points = extra_obs['boundary_points_left']
        else:
            self.steps_done += 1

        if extra_obs['first_point_reached']:
            # if we have already seen the first point, then allow only 3 penalty free steps (in case of large rotation) until next point is reached
            # reset the steps_done count every time a boundary point is seen.
            if extra_obs['next_point_reached'] or not self.first_point_visited:
                self.first_point_visited = True
                self.steps_done = 0

            # calculate penalty if steps_done counter is more than 3
            if self.steps_done >= 3:
                #reward = -0.25
                # make penalty dependent on the number of points left to give a clue to the optimizer of how many points are left.
                #reward = -(extra_obs['boundary_points_left'] / self.initial_number_of_boundary_points)
                reward = -0.25

        else:
            if self.steps_done > 2 * extra_obs['best_path_length']:
                # If we're still looking for the first boundary point and have exceeded best path 2 times, then penalize
                reward = -0.25

        return np.float32(reward * self.scale)

##
# Issue a reward for achieving the target - once per scene
##
class TargetAchievedRewardForDoor:
    def __init__(self, epsilon = 0.0, min_steps_in_new_room = 3, max_steps_in_new_room = 10):
        '''
        :param epsilon: How close is close enough to issue the reward
        '''
        self.reward_issued = False
        self.min_steps_in_new_room = min_steps_in_new_room
        self.max_steps_in_new_room = max_steps_in_new_room
        self.epsilon = epsilon
        self.steps_done = 0

    def __call__(self, obs, extra_obs, action):
        #print("T1")
        reward = 0
        if obs['is_first']:
            self.reward_issued = False
            self.steps_done = 0
        elif not self.reward_issued and index_to_action(int(action['action'])) == "STOP":
            '''
            We only want to issue this reward once the STOP action has been issued by the model. And at that point we will calculate
            how much we award based on what has been achieved.
            '''
            # high reward for achieving epsilon requirement for any door
            for dist in extra_obs['all_target_dists']:
                if dist <= self.epsilon:
                    reward += 100
                    break

            # any crossing into the new room is great
            if (extra_obs['stepsafterroomchange'] > 0):
                reward += 100

            # high reward for correct amount of steps in the new room
            if (extra_obs['stepsafterroomchange'] <= self.max_steps_in_new_room and extra_obs['stepsafterroomchange'] >= self.min_steps_in_new_room):
                reward += 100

            # Participation prize if any of the distances have become smaller
            # Participation prize equals to the best reduction of the distances
            if extra_obs['all_target_dists_initial'] is not None and len(extra_obs['all_target_dists']) == len(extra_obs['all_target_dists_initial']) and len(extra_obs['all_target_dists']) > 0:
                reward += 2 * max([d1 - d2 for d1, d2 in zip(extra_obs['all_target_dists_initial'], extra_obs['all_target_dists'])])

            # If none of the above rewards have been earned, then check if it needs a penalty for
            # early STOP (not walking enough to get even through the nearest door)
            if reward <= 0:
                # if extra_obs['all_target_dists_initial'] is not None and len(extra_obs['all_target_dists_initial']) > 0:
                #     min_distance_walk = np.min(extra_obs['all_target_dists_initial'])
                #     mean_distance_walk = np.mean(extra_obs['all_target_dists_initial'])
                # else:
                #     min_distance_walk = extra_obs['initial_distance']
                #     mean_distance_walk = min_distance_walk
                # if self.steps_done < 4 * mean_distance_walk:
                #     reward += min(-1 * (4 * min_distance_walk - self.steps_done) , 0)

                if self.steps_done < extra_obs['best_path_length']:
                    reward += -1 * (extra_obs['best_path_length'] - self.steps_done)

                reward = max(reward, -100)

            self.reward_issued = True
            #print("final reward: ", reward, " = ", extra_obs['initial_distance'], " - ", extra_obs['distanceleft'])
        else:
            self.steps_done += 1
        return np.float32(reward)

##
# Issue a reward for achieving the target - once per scene
##
class TargetAchievedRewardRoomCentre:
    def __init__(self, epsilon = 0.0, steps_in_new_room = 3):
        '''
        :param epsilon: How close is close enough to issue the reward
        '''
        self.reward_issued = False
        self.steps_in_new_room = steps_in_new_room
        self.epsilon = epsilon

    def __call__(self, obs, extra_obs, action, inventory=None):
        #print("T1")
        reward = 0
        if obs['is_first']:
            self.reward_issued = False
        elif not self.reward_issued and index_to_action(int(action['action'])) == "STOP":
            '''
            We only want to issue this reward once the STOP action has been issued by the model. And at that point we will calculate
            how much we award based on distance left
            '''
            # double the penalty for distance left to discourage early STOP
            if extra_obs['distanceleft'] >= extra_obs['initial_distance']:
                reward = (extra_obs['initial_distance'] - 2 * extra_obs['distanceleft'])
            else:
                reward = 3 * (extra_obs['initial_distance'] - extra_obs['distanceleft'])
            # extra reward for achieving epsilon requirement
            if extra_obs['distanceleft'] <= self.epsilon:
                reward += 20
            self.reward_issued = True
            #print("final reward: ", reward, " = ", extra_obs['initial_distance'], " - ", extra_obs['distanceleft'])
        return np.float32(reward)

##
# Issue a reward for achieving the target - once per scene
##
class TargetAchievedRewardForPerimeter:
    def __init__(self):
        '''
        :param epsilon: How close is close enough to issue the reward
        '''
        self.first_point_reward_issued = False
        self.stop_reward_issued = False

    def __call__(self, obs, extra_obs, action):
        #print("T1")
        reward = 0
        if obs['is_first']:
            self.first_point_reward_issued = False
            self.stop_reward_issued = False

        if not self.stop_reward_issued and index_to_action(int(action['action'])) == "STOP":
            # TODO: No reward for getting closer to the first point. Make it similar to TargetAchievedRewardForRoomcentre
            '''
            We only want to issue this reward once the STOP action has been issued by the model. And at that point we will calculate
            how much we award based on distance left and remaining points left unvisited
            '''
            if not extra_obs['first_point_reached']:
                # if we haven't reached the first point, then do the same that we do for room center
                if extra_obs['distanceleft'] >= extra_obs['initial_distance']:
                    reward = (extra_obs['initial_distance'] - 2 * extra_obs['distanceleft'])
                else:
                    reward = 3 * (extra_obs['initial_distance'] - extra_obs['distanceleft'])
                # extra reward for achieving epsilon requirement
                if extra_obs['distanceleft'] <= self.epsilon:
                    reward += 20
                # print("final reward: ", reward, " = ", extra_obs['initial_distance'], " - ", extra_obs['distanceleft'])
            else:
                # if we did achieve first point, then reward that with 100
                reward += 100

            if extra_obs['boundary_points_left'] > 0:
                reward -= 0.5 * extra_obs['boundary_points_left']  # 0.5 points taken away for each un-visited point
            else:
                reward += 100  # 100 for all points visited
            self.stop_reward_issued = True
            # print("final reward: ", reward, " = ", extra_obs['initial_distance'], " - ", extra_obs['distanceleft'])
        else:
            '''
            If this is not a STOP condition, then usual reward for point reaching
            '''
            if not self.first_point_reward_issued and extra_obs['first_point_reached']:
                reward += 10
                self.first_point_reward_issued = True

            if extra_obs['next_point_reached']:
                reward += 2.5

        return np.float32(reward)

class AI2ThorBase(embodied.Env):

    LOCK = threading.Lock()

    hab_exploration_stats_collection = []

    def __init__(self,
                 actions,
                 logdir="not_set",
                 repeat=1,
                 size=(64, 64),
                 logs=False,
                 hab_space=(100, 600),
                 hab_set="train",
                 places_per_hab=20,
                 grid_size=0.125,
                 reward_close_enough=0.125,
                 plan_close_enough=0.25,
                 env_index=-1
                 ):
        '''

        :param actions:
        :param repeat:
        :param size:
        :param logs:
        :param hab_space:
        :param hab_set:
        :param places_per_hab:
        :param grid_size:
        :param reward_close_enough:
        :param plan_close_enough:
        :param env_index: If this is anything other than -1, then we are evaluating with 3 envs and we want to split
            the hab_space into three and only use one portion per env. This could be improved by also specifying the
            number of envs, not just the index, but for now we will work with the assumption that the number of envs is 3.
        '''
        #print("C1")
        if logs:
            logging.basicConfig(level=logging.DEBUG)

        # if we have an env_index, then we assume that there are 3 envs and we will split hab_space between those
        # 3 envs and assign a portion to the current env according to its index.
        if env_index > -1:
            (hab_min, hab_max) = hab_space
            hab_diff = hab_max - hab_min
            hab_step = int(hab_diff / 3)
            hab_starts = range(hab_min, hab_max, hab_step)
            new_hab_min = hab_starts[env_index]
            new_hab_max = (new_hab_min + hab_step - 1) if env_index < 2 else (new_hab_min + hab_step)
            hab_space = (new_hab_min, new_hab_max)

        # AE: AI2-Thor simulation stuff
        self.rnc = RobotNavigationControl()
        self.controller = None
        self.atu = AI2THORUtils()
        self.rooms_in_habitat = None
        self.current_path_length = 1000
        self.reachable_positions = None
        self.grid_size = grid_size # how fine do we want the 2D grid to be.
        self.reward_close_enough = reward_close_enough # how close to the target is close enough for the purposes of reward. If we're this close or closer in simulation to the target, then consider it done
        self.plan_close_enough = plan_close_enough # how close to the target is close enough for the purposes of path planning. We may end up planning path to a point anywhere near the actual target by this much
        self.nu = NavigationUtils(step=self.grid_size)
        # If we get into a bad spot from which for whatever reason we can't plan a path out, then we'll set this to
        # True and based on it will teleport to a new place when we see this set.
        self._bad_spot = False
        self._bad_spot_cnt = 0
        self._total_reward_for_this_run = 0
        self.step_count_in_current_episode = 0
        self.step_count_since_start = 0
        self.distance_left = np.float32(1000.0)
        self.cur_pos_xy = None
        self.all_target_dists_initial = None
        self.room_type = -1 # current room type
        self.starting_room = None # which room we end up in when we spawn
        self.target_room = None # which room we want to end up in
        self.current_room = None # which room are we in now
        self.steps_in_new_room = 0 # how many steps have we made inside the new room since we first stepped into the target room (resets if we leave target room)
        self.env_retired = False # in some cases we want to be able to signal to driver.py that this env does not need driving anymore. This will help with that.
        self.prev_obs = None
        self.prev_extra_obs = None

        # When we store the statistics of each test run, we will want to capture these variables
        self.astar_path = []
        self.path_start = None
        self.path_dest = None
        self.travelled_path = []
        self.chosen_actions = []
        self.logdir = logdir

        print("AE hab_space:", hab_space, " logdir: ", logdir)
        #traceback.print_stack()
        # AE: based on whether we're training or evaluating, we will want to use different subsets of the habitat set
        (self.hab_min, self.hab_max) = hab_space
        self.hab_set = hab_set
        self.places_per_hab = places_per_hab

        self.choose_habitats_randomly_or_sequentially = True
        if (hab_set == "train"):
            print(f"AE Training on : {type(self).__name__}")
            self.choose_habitats_randomly_or_sequentially = True
        elif (hab_set == "test"):
            print(f"AE Testing on : {type(self).__name__}")
            self.choose_habitats_randomly_or_sequentially = False
        else:
            print(f"AE ?Validation? on : {type(self).__name__}")
            self.choose_habitats_randomly_or_sequentially = False

        self.evaluation_mode = (not self.choose_habitats_randomly_or_sequentially)

        # when we select a random position and plan path to the room centre, we will assign a value to this parameter
        # with the A* path length from that random position to the desired point. This will help calculate reward from all
        # further points.
        self.initial_path_length = 0
        self.initial_distance = 1000.0

        # If we use reward that tracks the best length of the path left, then we will need this variable
        self.best_path_length = 0

        # We will need to keep track of the target point that we want to reach because we will be re-planning path to it
        # from all sorts of different points.
        self.current_target_point = None

        # upon beginning we don't have any habitat loaded yet, but we will check this variable to determine if we have
        self.habitat_id = None
        self.explored_placements_in_current_habitat = []

        # Dreamer stuff
        self._size = size
        self._repeat = repeat
        self.isFirst = False

        # Here we create the first AI2Thor environment (controller)
        #with self.LOCK:
        self.load_next_start_point()

        self._step = 0
        self._obs_space = self.obs_space

        self._action_names = tuple(actions.keys())
        self._action_values = tuple(actions.values())
        message = f'Indoor Navigation action space ({len(self._action_values)}):'
        print(message, ', '.join(self._action_names))
        #print("C2")

    @property
    def obs_space(self):
        return {
            'image': elements.Space(np.uint8, self._size + (3,)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
            'doorvis': elements.Space(np.float32),
            'newroom': elements.Space(np.float32),
            #'distanceleft': elements.Space(np.float32),
            #'stepsafterroomchange': elements.Space(np.float32),
            #'roomtype': elements.Space(np.float32),
        }

    @property
    def act_space(self):
        return {
            'action': elements.Space(np.int32, (), 0, len(self._action_values)),
            'reset': elements.Space(bool),
        }

    def step(self, action, add_extra = False):
        #t1 = time.time()
        # If this env has been retired (in evaluation mode we have evaluated everything already), then
        # don't actually do any stepping, but just return the previous obs
        if self.env_retired:
            if add_extra:
                return self.prev_obs, self.prev_extra_obs
            else:
                return self.prev_obs

        #print("S1")
        action = action.copy()
        #index = action.pop('action')
        #print("action: ", action, " self._action_values: ", self._action_values, " inder:", index)
        #action.update(self._action_values[index])
        #print("action: ", action)

        if action['reset']:
            #tr1 = time.time()
            print('R', end='', sep='')
            # STORE EPISODE STATS:
            # A* path length, A* path, travelled path length, travelled path, habitat id, actions taken.
            #if self.hab_set == "train":
            # If evaluating, then store episode stats
            if not self.choose_habitats_randomly_or_sequentially:
                episode_stats = {
                    "local_step": self.step_count_since_start,
                    "steps_used": self.step_count_in_current_episode,
                    "habitat_id": str(self.habitat_id),
                    "bad_spot": self._bad_spot,
                    "have_arrived": str(self.have_we_arrived(epsilon=self.reward_close_enough, eval=True)),
                    "path_start": self.path_start,
                    "path_dest": self.path_dest,
                    "astar_path": self.astar_path,
                    "travelled_path": self.travelled_path,
                    "chosen_actions": self.chosen_actions,
                }
                # specially for door finder- if we're running it, we want all door targets
                if hasattr(self, 'all_door_targets'):
                    episode_stats['all_door_targets'] = self.all_door_targets

                if hasattr(self, 'all_astar_paths'):
                    episode_stats['all_astar_paths'] = self.all_astar_paths
                #print(episode_stats)

                with open(self.logdir + "/episode_data.jsonl", "a") as f:
                    f.write(json.dumps(episode_stats) + "\n")

            #tr2 = time.time()
            obs, extra_obs = self._reset()
            #tr3 = time.time()
            #print("AE Tr = ", round(tr2-tr1,4), " Trr = ", round(tr3-tr2, 4))
        elif index_to_action(int(action['action'])) == "STOP":
            #t1s = time.time()
            self.chosen_actions.append(int(action['action']))
            self._done = True

            try:
                self.distance_left, self.room_type, self.cur_pos_xy = self.get_current_path_and_pose_state()
                self.travelled_path.append(self.cur_pos_xy)
                self.all_target_dists = self.euclidean_dist_to_all_targets()
            except ValueError as e:
                self.distance_left = np.float32(1000.0)
                self._bad_spot = True
                print('O', end='', sep='')

            print('S', end='', sep='')
            obs, extra_obs = self.current_ai2thor_observation()
            #t2s = time.time()
            #print("AE Ts = ", round(t2s-t1s,4))
        else:
            #t1a = time.time()
            raw_action = index_to_action(int(action['action']))
            #t1r = time.time()
            self.rnc.execute_action(raw_action, moveMagnitude=self.grid_size, grid_size=self.grid_size, adhere_to_grid=True)
            #t2r = time.time()
            #print("AE Trnc = ", round(t2r-t1r,4))
            self.chosen_actions.append(int(action['action']))
            # This is slightly ugly, but we need to calculate distance_left variable right after rnc.execute_action
            # to allow observation to be up to date. In time this should be moved to some function instead of relying
            # on global variables.
            try:
                self.distance_left, self.room_type, self.cur_pos_xy = self.get_current_path_and_pose_state()
                self.travelled_path.append(self.cur_pos_xy)
                if self.all_target_dists_initial is None:
                    self.all_target_dists_initial = self.euclidean_dist_to_all_targets()
                #self._done = self.have_we_arrived(self.reward_close_enough)
            except ValueError as e:
                self.distance_left = np.float32(1000.0)
                self._bad_spot = True

            if self._bad_spot:
                #print("FORCED SCENE CHANGE!!!", self.step_count_in_current_episode)
                #STORE EPISODE STATS
                print('O', end='', sep='')
                ##
                # This must be self._done = True instead of direct reset. We will reset in the next loop
                ##
                #obs = self._reset()
                self._done = True
            #else:
            obs, extra_obs = self.current_ai2thor_observation()
            #t2a = time.time()
            #print("AE Ta = ", round(t2a-t1a,4))

        # Now we turn the obs that was returned by the environment into obs that we use for training,
        # and to not confuse the two, make sure that 'pov' field is not there, because it should be 'image'.
        obs, extra_obs = self._obs(obs, extra_obs)
        self._step += 1
        self.step_count_in_current_episode += 1
        self.step_count_since_start += 1
        assert 'pov' not in obs, list(obs.keys())
        #print("S2")
        self.prev_obs = obs
        self.prev_extra_obs = extra_obs
        #t2 = time.time()
        #print("AE Tt = ", round(t2-t1,4))
        if add_extra:
            return obs, extra_obs
        else:
            return obs

    ##
    # Returns current observation of the state (image mostly)
    ##
    def current_ai2thor_observation(self):
        #print("O1")
        event = self.controller.last_event
        self._current_image = event.cv2img

        #print("self.current_room == self.target_room", self.current_room, self.target_room)
        # If we've changed a room, then count how many steps we're making in it.
        # If we change a room again, start the count from 0
        if self.starting_room is not None and self.current_room != self.starting_room:
            self.steps_in_new_room += 1
            if self.steps_in_new_room > 15:
                #self.steps_in_new_room = 0
                self.starting_room = self.current_room

        all_visible_doors = self.nu.get_all_visible_doors(self.controller)

        obs = dict(
            reward = 0.0,
            pov = self._current_image,
            is_first = np.bool(self.isFirst),
            is_last = np.bool(self._done),
            is_terminal = np.bool(self._done),
            doorvis = np.float32(len(all_visible_doors) > 0),
            newroom = np.float32(self.steps_in_new_room > 0)
        )

        extra_obs = dict(
            distanceleft=np.float32(self.distance_left),
            stepsafterroomchange=np.float32(self.steps_in_new_room),
            roomtype=np.float32(self.room_type),
            initial_distance=np.float32(self.initial_distance),
            all_target_dists=self.all_target_dists,
            all_target_dists_initial=self.all_target_dists_initial,
            best_path_length = self.best_path_length
        )

        if self._done:
            print('D', sep='', end='')

        self.isFirst = False # this will have to be set to True when we reset the env
        #print("O2")
        return obs, extra_obs

    def _reset(self):
        #t1r = time.time()
        #print("R1")
        self.astar_path = []
        self.path_start = None
        self.path_dest = None
        self.travelled_path = []
        self.chosen_actions = []
        # Load new point or even a habitat, set reward to 0 and is_first = True and is_last = False and self._done = False
        #with self.LOCK:
        self.load_next_start_point()
            #obs = self._env.step({'reset': True})
        #t2r = time.time()

        self.step_count_in_current_episode = 0
        self._step = 0
        self._done = False
        self._bad_spot = False

        self.distance_left = 1000
        self.all_target_dists = []
        self.steps_in_new_room = 0
        self.room_type = -1
        self.starting_room = None
        self.all_target_dists_initial = None

        obs, extra_obs = self.current_ai2thor_observation()
        #t3r = time.time()
        #print("AE: Trt = ", round(t3r - t1r, 4), " Tl = ", round(t2r - t1r, 4), " Tca = ", round(t3r - t2r, 4))
        #print("R2")
        return obs, extra_obs

    def _obs(self, obs, extra_obs):
        #print("_O1")
        obs = {
            'image': obs['pov'],
            'reward': np.float32(0.0), # reward will be calculated later
            'is_first': obs['is_first'],
            'is_last': obs['is_last'],
            'is_terminal': obs['is_terminal'],
            'doorvis': obs['doorvis'],
            'newroom': obs['newroom']
        }

        extra_obs = {
            'distanceleft': extra_obs['distanceleft'],
            'stepsafterroomchange': extra_obs['stepsafterroomchange'],
            'roomtype': extra_obs['roomtype'],
            'initial_distance': extra_obs['initial_distance'],
            'all_target_dists': extra_obs['all_target_dists'],
            'all_target_dists_initial': extra_obs['all_target_dists_initial'],
            'best_path_length': extra_obs['best_path_length']
        }

        #print("obs: ", obs)
        for key, value in obs.items():
            space = self._obs_space[key]
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            #print("val: ", value, " space: ", space, " key: ", key, " (key, value, @dtype@, value.shape, space): ", (key, value, value.shape, space))
            assert value in space, (key, value, value.dtype, value.shape, space)
        #print("obs: ", obs)
        #print("_O2")
        return obs, extra_obs

    def load_random_habitat(self):
        #print("LRH1")
        # choose a random habitat from a space of given habitats by self.hab_max and self.hab_min
        loaded = False

        # we are going to choose a completely new habitat now. Before we do that, we want to register somewhere
        # what habitat was being explored up until now and what placements were looked at in there.
        if len(self.explored_placements_in_current_habitat) > 0:
            hab_exploration_stats = {
                "local_step": self.step_count_since_start,
                "habitat_id": self.habitat_id,
                "explored_placements_in_current_habitat": self.explored_placements_in_current_habitat
            }
            #print(hab_exploration_stats)
            AI2ThorBase.hab_exploration_stats_collection.append(hab_exploration_stats)
            with open("stat_store", "wb") as stat_store:
                pickle.dump(AI2ThorBase.hab_exploration_stats_collection, stat_store)
        # now that we've saved previous habitat exploration stats, we can carry on with a new habitat

        while not loaded:
            try:
                if (self.choose_habitats_randomly_or_sequentially): # if we want a random habitat (e.g. we're training)
                    sp = elements.Space(np.int32, (), self.hab_min, self.hab_max)
                    self.habitat_id = sp.sample()
                else:
                    # if we want a sequential habitat (e.g. we're evaluating or testing)
                    if (self.habitat_id == None):
                        self.habitat_id = self.hab_min
                    elif (self.habitat_id < self.hab_max):
                        self.habitat_id += 1
                    else:
                        # we're done, we need to terminate the evaluation process now
                        #exit()
                        # but instead of just exiting the whole program, let's set up a flag that will tell driver.py
                        # that this env does not need driving anymore.
                        self.env_retired = True
                        break

                # load_habitat will also call self.choose_random_placement_in_habitat(), which will in turn calculate
                # current distance cost to the target
                self.load_habitat(self.habitat_id)
                # enfore at least 2 rooms in a habitat
                if len(self.rooms_in_habitat) >= 2:
                    loaded = True
            except ValueError as e:
                continue
        #print("LRH2")
    ##
    # This kind of combines 2 functions: load_random_habitat and choose_random_placement_in_habitat.
    # The idea is that usually we only want to load a different starting point within the same habitat,
    # but sometimes we will want to load a new habitat entirely. Also, if we have exhausted all usable random
    # places in the given habitat, then we want to load a new habitat. This is all best handled in one place-
    # this function.
    ##
    def load_next_start_point(self):
        #t1 = time.time()
        #print("L1")
        # if nothing has been loaded, then we just load a brand new habitat - Simple
        if self.habitat_id is None:
            self.load_random_habitat()
            #t2_1 = time.time()
            #print("AE: Tl1 = ", round(t1 - t2_1, 4))
        else:
            # otherwise, we want to look at what have we explored and what is available
            # if we have already explored 20 random locations in this habitat, then it's time to move on
            if len(self.explored_placements_in_current_habitat) > self.places_per_hab:
                self.load_random_habitat()
                #t2_2 = time.time()
                #print("AE: Tl2 = ", round(t1 - t2_2, 4))
            else:
                # otherwise try to load the next random placement (it will attempt a few times, currently 10).
                # If that fails, then we load new habitat.
                try:
                    self.choose_random_placement_in_habitat()
                    #t2_3 = time.time()
                    #print("AE: Tl3 = ", round(t1 - t2_3, 4))
                except ValueError as e:
                    self.load_random_habitat()
                    #t2_4 = time.time()
                    #print("AE: Tl4 = ", round(t1 - t2_4, 4))

        self.isFirst = True # we just loaded a new scene or habitat. The next observation will be first
        #print("L2")

    ##
    # Load the given habitat- load it, and put agent in a random place
    ##
    def load_habitat(self, habitat_id):
        #print("LH1")
        # load required habitat
        #print("AE: haba: ", habitat_id)
        self.habitat = self.atu.load_proctor_habitat(int(habitat_id), self.hab_set)
        self.explored_placements_in_current_habitat = []

        # Launch a controller for the loaded habitat. If we already have a controller,
        # then reset it instead of loading a new one.
        if (self.controller == None):
            self.controller = launch_controller({"scene": self.habitat,
                                                 "VISIBILITY_DISTANCE": 3.0,
                                                 "headless": True,
                                                 "IMAGE_WIDTH": 64,
                                                 "IMAGE_HEIGHT": 64,
                                                 "GRID_SIZE": self.grid_size,
                                                 "GPU_DEVICE": 1,
                                                 "RENDER_DEPTH": False,
                                                 "RENDER_INSTANCE_SEGMENTATION": False,
                                                 "RENDER_IMAGE": True
                                                 # "IMAGE_WIDTH": 64,
                                                 # "IMAGE_HEIGHT": 64
                                                 })
            self.rnc.set_controller(self.controller)  # This allows our control scripts to interact with AI2-THOR environment
            self.atu.set_controller(self.controller)
        else:
            self.controller.reset(self.habitat)
            # self.reset_state()
            self.rnc.reset_state()
            # self.rnc.set_controller(self.controller)

        # Take a snapshot of all available positions- these won't change while we're in this habitat,
        # so no need to re-do them everytime we plan a path.
        #self.grid_size = self.controller.initialization_parameters["gridSize"]
        self.reachable_positions, self.unreachable_postions, self.full_grid, self.rooms_in_habitat = self.update_navigation_artifacts(self.habitat)

        # Now place the robot in a random position and figure out the target from there.
        self.choose_random_placement_in_habitat()
        #self.choose_specific_placement_in_habitat()
        #print("LH2")

    def choose_specific_placement_in_habitat(self):
        #params["position"] = dict(x=7.0, y=0.9009997844696045, z=5.625)
        #params["rotation"] = dict(x=0.0, y=270, z=0.0)
        # self.controller.step(action="Teleport", **pos_navigate_to)
        place_with_rtn = (5.62, 3.5, 270)
        self.rnc.teleport_to(place_with_rtn)
        try:
            self.current_target_point, _ = self.choose_door_target(place_with_rtn)
        except ValueError as err:
            print("O", err)

        #print("AE: Path Length: ", path_length)
        (cur_path, reachable_positions, start, dest) = self.nu.get_last_path_and_params()
        print("AE: Path: ", cur_path)
        self.atu.visualise_path2(cur_path, self.reachable_positions, self.unreachable_postions, self.rooms_in_habitat, start, dest,
                            show_unreachable_pos = False,
                            show_reachable_pos = True)
        k = getch()

        cur_pos = self.rnc.get_agent_pos_and_rotation()
        target_point = Point(5.12, 4.0)#((3.12, 0.88, 4.75), (0.0, 180, 0.0))
        door_path_length = self.nu.get_path_cost_to_target_point(cur_pos,
                                                                         target_point,
                                                                         self.reachable_positions,
                                                                         close_enough=self.plan_close_enough,
                                                                         step=self.grid_size,
                                                                         debug=True)
        print("AE: door_path_length: ", door_path_length)
        k = getch()

    def create_rnd_object(self):
        seed = 1983
        if not hasattr(self, "rnd"):
            self.rnd = random.Random(seed)
        return self.rnd

    ##
    # Here we will select a number of random placements and then choose one to navigate from it
    # to some goal.
    ##
    def choose_random_placement_in_habitat(self):
        #t1 = time.time()
        #print("CH1")
        ## All we need is a set of random positions and we get them like this:
        # params for the random teleportation part
        seed = 1983
        num_stops = 20
        num_rotates = 4
        sep = 1.0
        v_angles = [30]
        h_angles = [0, 45, 90, 135, 180, 225, 270, 315]

        """
        num_stops: Number of places the agent will be placed
        num_rotates: Number of random rotations at each place
        sep: the minimum separation the sampled agent locations should have
    
        kwargs: See thortils.vision.projection.open3d_pcd_from_rgbd;
        """
        ## If we are training (i.e., loading habitats randomly), then don't use a seed
        if (self.choose_habitats_randomly_or_sequentially):
            rnd = random.Random()
        else: # if, on the other hand, we are evaluating or testing (loading habitats sequentially), then test everything the same way- use a seed
            rnd = self.create_rnd_object()

        #t2 = time.time()
        #initial_agent_pose = tt.thor_agent_pose(self.controller)
        #initial_horizon = tt.thor_camera_horizon(self.controller.last_event)
        #t3 = time.time()

        # reachable_positions = tt.thor_reachable_positions(self.controller)
        # self.reachable_positions
        placements = sep_spatial_sample(self.reachable_positions, sep, num_stops, rnd=rnd)
        #t4 = time.time()

        # print(placements)

        # Choose one placement in the set of placements and then plan path from that placement to
        # the middle of the room. If planning path is not possible, then choose another one.
        path_planned = False
        placement_attempts = 0
        while not path_planned:
            placement_attempts += 1
            #els = elements.Space(np.int32, (), 0, len(placements))
            #p = list(placements)[int(els.sample())]
            el_ndx = rnd.randrange(0, len(placements))
            p = list(placements)[el_ndx]
            # p = placements.pop()

            # append a rotation to the place.
            yaw = rnd.sample(h_angles, 1)[0]
            place_with_rtn = p + (yaw,)
            #print("Placement: ", place_with_rtn, " el_ndx: ", el_ndx)
            self.explored_placements_in_current_habitat.append(place_with_rtn)
            ## Teleport, then start new exploration. Achieve goal. Then repeat.
            #tt1 = time.time()
            self.rnc.teleport_to(place_with_rtn)
            #tt2 = time.time()

            # We've just been put in a random place in a habitat. We want to move now to where we want to go,
            # e.g., middle of the room, a door, etc.. For that we need to plan a path to there.
            try:
                point_for_room_search = (p[0], "", p[1])
                #tp1 = time.time()
                self.current_target_point = self.choose_target_point(place_with_rtn, point_for_room_search) # self.target_room will be set in this function
                #tp2 = time.time()
                #print("AE: Tctp = ", round(tp2-tp1,4))

                cur_pos = self.rnc.get_agent_pos_and_rotation()
                #tp3 = time.time()
                #print("Placement: ", place_with_rtn, " cur_pos: ", cur_pos, " el_ndx: ", el_ndx)
                self.initial_path_length = self.nu.get_path_cost_to_target_point(cur_pos,
                                                                                 self.current_target_point,
                                                                                 self.reachable_positions,
                                                                                 close_enough=self.plan_close_enough,
                                                                                 step=self.grid_size)
                #tp4 = time.time()
                # Now let's remember the A* path- we will want it for results.
                (self.astar_path, _, self.path_start, self.path_dest) = self.nu.get_last_path_and_params()
                #print("AE: Path: ", self.astar_path)

                #tp5 = time.time()
                # If we have several permissible destinations, then calculate A* path for all of them
                if hasattr(self, 'all_door_targets') and self.evaluation_mode:
                    self.all_astar_paths = []
                    for dt in self.all_door_targets:
                        target_point = Point(dt["pos"]["x"], dt["pos"]["z"])
                        path_length = self.nu.get_path_cost_to_target_point(cur_pos,
                                                                             target_point,
                                                                             self.reachable_positions,
                                                                             close_enough=self.plan_close_enough,
                                                                             step=self.grid_size)
                        (astar_path, _, _, _) = self.nu.get_last_path_and_params()
                        self.all_astar_paths.append(astar_path)
                #tp6 = time.time()
                #print("AE: Tapc = ", round(tp6 - tp5, 4))
                if isinstance(self, DoorFinder):
                    # what is the room we start in
                    self.starting_room = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)
                    self.steps_in_new_room = 0

                    if (self.starting_room == None): raise ValueError("Starting room not identifiable")

                    # We must ensure that we navigate from one room to another
                    if self.target_room == self.starting_room: raise ValueError("start and end points in same room")
                #tp7 = time.time()
                #print("AE Ttel = ", round(tt2 - tt1,4), " Tpl1 = ", round(tp2-tp1,4), " Tpl2=", round(tp3-tp2,4), " Tpl3=", round(tp4-tp3,4), " Tpl4=", round(tp5-tp4,4), " Tpl5=", round(tp6-tp5,4), " Tpl6=", round(tp7-tp6,4))
            except ValueError as e:
                # If the path could not be planned, then drop it and carry on with the next one
                #print(f"ERROR: {e}")
                if placement_attempts <= 10:
                    print('.', sep='', end='')
                    continue
                else:
                    # If we have tried for 10 times already, then give up with this habitat
                    print("next_hab", sep="", end="")
                    raise e

            # print("PATH & PLAN: ", path_and_plan)
            #            path = path_and_plan[0]
            #            plan = path_and_plan[1]
            # self.prev_pose = thor_agent_pose(self.controller)  # This is where we are before the plan started
            # place_with_rtn
            # thor_pose_as_tuple(self.prev_pose)
            #print("AE poses: place_with_rtn: ", place_with_rtn, " p: ", p, " self.rnc.get_agent_pos_and_rotation(): ",
            #      self.rnc.get_agent_pos_and_rotation())
            #            cur_pos = self.rnc.get_agent_pos_and_rotation()
            #            self.initial_path_length = get_path_length(path, cur_pos)

            # at this point current path length is the initial path length. We will re-calculate current path length
            # many times and reward will be calculated using it.
            self.current_path_length = self.initial_path_length
            self.best_path_length = self.initial_path_length

            try:
                self.initial_distance, _, _ = self.get_current_path_and_pose_state()
            except ValueError as e:
                self.initial_distance = np.float32(1000.0)
                self._bad_spot = True
                print('o', end='', sep='')

            #print("CH2")
            path_planned = True
        #t5 = time.time()
        #print("AE, Tb1 = ", round(t2 - t1, 4), " Tb2 = ", round(t3-t2,4), " Tb3 = ", round(t4-t3,4), " Tb4 = ", round(t5-t4, 4), " placement_attempts: ", placement_attempts)
        #print("AE, Tb4 = ", round(t5 - t4, 4), " placement_attempts: ", placement_attempts)

    # Get all reachable positions and store them in a variable.
    def update_navigation_artifacts(self, house):
        #print("U1")
        reachable_positions = [
            tuple(map(lambda x: round(roundany(x, self.grid_size), 2), pos))
            for pos in thor_reachable_positions(self.controller)]
        # print(reachable_positions, self.grid_size)

        # In this habitat we have these rooms
        rooms_in_habitat = get_rooms_ground_truth(house)
        # print(house["rooms"])
        # print("reachable_positions: ", reachable_positions)
        # AE: Path length infra set up
        # pos_ba = thor_reachable_positions(controller, by_axes = True)
        # print("AE, by axes: ", pos_ba)
        full_grid = create_full_grid_from_room_layout(rooms_in_habitat, step=self.grid_size)
        full_grid = [tuple(map(lambda x: round(x, 2), pos)) for pos in full_grid]
        unreachable_postions = set(full_grid) - set(reachable_positions)
        #(safe_pos, buf_unreachable_pos) = add_buffer_to_unreachable(set(reachable_positions), set(full_grid), step=self.grid_size)

        #print("U2")
        return reachable_positions, unreachable_postions, full_grid, rooms_in_habitat

    # This function will calculate path length to the desired point from the current position.
    # Also- what room type we're in
    def get_current_path_and_pose_state(self):
        #print("G1")
        try:
            cur_pos = self.rnc.get_agent_pos_and_rotation()
            # we only need to re-calculate path length at every step, when we train. If we test, then we don't need to.
            if self.choose_habitats_randomly_or_sequentially:
                self.current_path_length = self.nu.get_path_cost_to_target_point(cur_pos,
                                                                                 self.current_target_point,
                                                                                 self.reachable_positions,
                                                                                 close_enough=self.plan_close_enough,
                                                                                 step=self.grid_size)
            else:
                self.current_path_length = 0.0

            self.current_path_length = np.float32(self.current_path_length)
        except ValueError as e:
            #print(f"ERROR: {e}")
            #print("Using previous current_path_length: ", self.current_path_length)
            print('!', end='', sep='')
            self._bad_spot = True
            self._bad_spot_cnt += 1
            raise e # pass it on because reward calculation also needs to know

        # if we've been successful so far, then we can now look up room type
        cur_pos_xy = (cur_pos[0][0], "", cur_pos[0][2])
        room_type = self.find_room_type_of_this_point(cur_pos_xy)
        if room_type == None:
            print('i', end='', sep='')
            raise ValueError("Bad room picked, type can't be determined.")
        # and the actual room
        self.current_room = room_this_point_belongs_to(self.rooms_in_habitat, cur_pos_xy)
        if (self.current_room == None):
            print('y', end='', sep='')
            raise ValueError("Current room not identifiable")

        #print("G2")
        return self.current_path_length, room_type, cur_pos_xy

    # Determines if we have little enough left to call it an achieved goal
    def have_we_arrived(self, epsilon = 0.0, eval=False):
        pass

    def close(self):
        if (self.controller != None):
            self.controller.stop()

    ##
    # Chooses the target point that we want to navigate to, given current position.
    # This will have to be implemented in derived classes.
    ##
    def choose_target_point(self, place_with_rtn = None, place_with_no_rtn = None):
        pass
        # if (self.doors_or_centre):
        #     self.current_target_point = self.choose_door_target(place_with_rtn)
        #     # print("self.current_target_point: ", self.current_target_point)
        # else:
        #     point_for_room_search = (p[0], "", p[1])
        #     # print("point_for_room_search: ", point_for_room_search)
        #     self.current_target_point = self.find_room_centre_target(point_for_room_search)

    def find_room_type_of_this_point(self, point_for_room_search):
        '''
        We want to find out an ID for the room type that this point belongs to
        :param point_for_room_search:
        :return:
        '''
        #print("FR1")
        room_of_placement = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)
        if room_of_placement == None:
            return None
        room_type = room_of_placement[0]
        room_type = np.uint8(RoomType.interpret_label(room_type).value)
        #print("FR2")
        return room_type

    ##
    # Calculate all Euclidean distances to all door targets
    ##
    def euclidean_dist_to_all_targets(self):
        dists = []
        if hasattr(self, 'all_door_targets') and self.all_door_targets != None:
            for dt in self.all_door_targets:
                p1 = (dt['pos']['x'], dt['pos']['z'])
                p2 = (self.cur_pos_xy[0], self.cur_pos_xy[2])
                dists.append(euclidean_dist(p1, p2))
        return dists

##
# Room centre finding task
##
class RoomCentreFinder(AI2ThorBase):
    def __init__(self, actions, *args, **kwargs):
        super().__init__(actions, *args, **kwargs)

    def choose_target_point(self, place_with_rtn = None, place_with_no_rtn = None):
        return self.find_room_centre_target(place_with_no_rtn)

    ##
    # Finds the centre of the current room given the current position and the rooms in habitat.
    ##
    def find_room_centre_target(self, point_for_room_search):
        #print("FRC1")
        # We've just been put in a random place in a habitat. We want to move now to where we want to go,
        # e.g., middle of the room, a door, etc.. For that we need to plan a path to there.
        room_of_placement = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)

        if (room_of_placement == None): raise ValueError("Room of placement not identifiable")

        room_centre = room_of_placement[2]
        #print("FRC2")
        return room_centre

    # Determines if we have little enough left to call it an achieved goal
    def have_we_arrived(self, epsilon = 0.0, eval = False):
        return (self.current_path_length <= epsilon)

##
# Door Finding Task
##
class DoorFinder(AI2ThorBase):
    def __init__(self, actions, *args, **kwargs):
        super().__init__(actions, *args, **kwargs)
        self.all_door_targets = None

    def choose_target_point(self, place_with_rtn = None, place_with_no_rtn = None):
        target_point, self.all_door_targets = self.choose_door_target(place_with_rtn)
        return target_point

    ##
    # Go through all the doors and find the most appropriate as a target, then add a little bit extra so that
    # we end up going through the door.
    # place_with_rtn: Place with rotation, e.g.: (6.62, 6.25, 180)
    ##
    def choose_door_target(self, place_with_rtn):
        #print("CD1")
        current_target_point = None
        try:
            current_target_point, all_door_targets = self.nu.find_door_target(place_with_rtn,
                                                            self.rooms_in_habitat,
                                                            self.reachable_positions,
                                                            self.habitat,
                                                            self.controller, close_enough=self.plan_close_enough,
                                                            step=self.grid_size, extend_path=True)

            # t1 = time.time()
            # pose = ((place_with_rtn[0], 0.0, place_with_rtn[1]),
            #         (0.0, place_with_rtn[2], 0.0))  # place_with_rtn in AI2-Thor format
            # path_length = self.nu.get_path_cost_to_target_point(pose,
            #                                                     current_target_point,
            #                                                     self.reachable_positions,
            #                                                     close_enough=self.plan_close_enough,
            #                                                     step=self.grid_size)

            # if we've been successful so far, then we can now look up room type
            trg_pos_xy = (current_target_point.x, "", current_target_point.y)
            self.target_room = room_this_point_belongs_to(self.rooms_in_habitat, trg_pos_xy)
            # print("AE: path plan time: ", (time.time() - t1))
        except ValueError as e:
            #path_length = 0
            # print("AE: No Path Found", e)
            print('£', end='', sep='')

        if (self.target_room == None):
            raise ValueError("Target room not identifiable")

        if current_target_point == None:
            raise ValueError("No suitable doors were found")

        # print("AE: Path Length: ", path_length)
        #(cur_path, reachable_positions, start, dest) = self.nu.get_last_path_and_params()
        # print("AE: Path: ", cur_path)
        # atu.visualise_path2(cur_path, reachable_positions, unreachable_postions, rooms_in_habitat, start, dest,
        #                    show_unreachable_pos=False,
        #                    show_reachable_pos=False)
        # atu.visualise_path2(cur_path, reachable_positions, buf_unreachable_pos, rooms_in_habitat, start, dest, show_unreachable_pos=True)
        #print("CD2")
        return current_target_point, all_door_targets

    # Determines if we have little enough left to call it an achieved goal
    def have_we_arrived(self, epsilon = 0.0, eval = False):
        # If we're free to choose any door as destination (i.e., we are evaluating),
        # then mark it as arrived if we're within distance of any door
        #if eval: # This was relaxed for training - to allow any door as target
        if self.all_door_targets != None:
            for dt in self.all_door_targets:
                p1 = (dt['pos']['x'], dt['pos']['z'])
                p2 = (self.cur_pos_xy[0], self.cur_pos_xy[2])
                if euclidean_dist(p1, p2) <= epsilon:
                    return True
        return False
        # else: # otherwise we have a calculated path length to the specific door that we want and we need to compare that to allowed error
        #     return (self.current_path_length <= epsilon or self.steps_in_new_room >= 3)

##
# Room perimeter finding task
##
class PerimeterFinder(AI2ThorBase):
    def __init__(self, actions, *args, **kwargs):
        #super().__init__(actions, *args, **kwargs)
        self.bc = BoundaryCalculations()
        self.boundary_points = set()
        self.first_point_reached = False
        self.looking_at = None
        self.steps_to_point_acceptable = 2
        super().__init__(actions, *args, **kwargs)

    def choose_target_point(self, place_with_rtn = None, place_with_no_rtn = None):
        self.first_point_reached = False
        return self.find_perimeter_point_looking_at(place_with_rtn, place_with_no_rtn)

    ##
    # Finds the centre of the current room given the current position and the rooms in habitat.
    ##
    def find_perimeter_point_looking_at(self, current_location, point_for_room_search):
        room_of_placement = room_this_point_belongs_to(self.rooms_in_habitat, point_for_room_search)
        if (room_of_placement == None): raise ValueError("Room of placement not identifiable")

        self.boundary_points = self.bc.find_room_perimeter_path(self.reachable_positions, self.unreachable_postions,
                                                           room_of_placement, self.habitat)

        # current_location:  (2.38, 4.62, 225)
        self.looking_at = self.bc.get_target_boundary_point(self.boundary_points, current_location)
        #print("cur_pos: ", current_location, " looking_at: ", looking_at)
        if self.looking_at is None:
            raise ValueError("Looking at point can't be determined")
        return Point(self.looking_at)

    def current_ai2thor_observation(self):
        #print("O1.child")
        '''
        Override the current observation method to add additional extra_obs fields
        :return:
        '''
        orig_obs, extra_obs = super().current_ai2thor_observation()
        #print("O1.child2")
        cur_pos = self.rnc.get_agent_pos_and_rotation()
        extra_obs['cur_pos'] = cur_pos
        extra_obs['next_point_reached'] = False

        cur_pos_2d = (cur_pos[0][0], cur_pos[0][2])
        # if we have reached the first point of perimeter within close proximity, then proceed to the next stage of perimeter navigation
        if (not self.first_point_reached and euclidean_dist(cur_pos_2d, self.looking_at) <= self.steps_to_point_acceptable * self.grid_size):
            self.first_point_reached = True
            #if self.looking_at not in self.boundary_points:
            #    breakpoint()
            self.boundary_points.remove(self.looking_at)

        # if first point has already been reached, then check which one is the next nearest point. if we are near any other points
        if self.first_point_reached and len(self.boundary_points) > 0:
            distance_to_np, next_nearest_point = min([(euclidean_dist(cur_pos_2d, p), p) for p in self.boundary_points], key=lambda x: x[0])
            if distance_to_np <= self.steps_to_point_acceptable * self.grid_size:
                # reached next point
                self.boundary_points.remove(next_nearest_point)
                extra_obs['next_point_reached'] = True

        extra_obs['boundary_points_left'] = len(self.boundary_points)
        extra_obs['first_point_reached'] = self.first_point_reached

        #traceback.print_stack()
        #print("extra_obs O2: ", extra_obs)

        return orig_obs, extra_obs

    def _obs(self, obs, extra_obs):
        '''
        Also override _obs function so that it respects the relevant fields
        :param obs:
        :param extra_obs:
        :return:
        '''
        #print("_O1")
        obs = {
            'image': obs['pov'],
            'reward': np.float32(0.0), # reward will be calculated later
            'is_first': obs['is_first'],
            'is_last': obs['is_last'],
            'is_terminal': obs['is_terminal'],
            'doorvis': obs['doorvis'],
            'newroom': obs['newroom']
        }

        extra_obs = {
            'distanceleft': extra_obs['distanceleft'],
            'initial_distance': extra_obs['initial_distance'],
            'best_path_length': extra_obs['best_path_length'],
            'first_point_reached': extra_obs['first_point_reached'],
            'next_point_reached': extra_obs['next_point_reached'],
            'cur_pos': extra_obs['cur_pos'],
            'boundary_points_left': extra_obs['boundary_points_left']
        }

        #print("obs: ", obs)
        for key, value in obs.items():
            space = self._obs_space[key]
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            #print("val: ", value, " space: ", space, " key: ", key, " (key, value, @dtype@, value.shape, space): ", (key, value, value.shape, space))
            assert value in space, (key, value, value.dtype, value.shape, space)
        #print("obs: ", obs)
        #print("_O2")
        return obs, extra_obs

    # Determines if we have little enough left to call it an achieved goal
    def have_we_arrived(self, epsilon = 0.0, eval = False):
        return (len(self.boundary_points) == 0)

import logging, elements, cv2, embodied, traceback, threading
import numpy as np
from ai2_thor_model_training.ae_utils import (action_mapping,
                                              action_to_index, index_to_action, inverted_action_mapping,
                                              AI2THORUtils, get_path_length, get_centre_of_the_room,
                                              room_this_point_belongs_to, get_rooms_ground_truth,
                                              get_all_objects_of_type, is_point_inside_room_ground_truth,
                                              create_full_grid_from_room_layout, add_buffer_to_unreachable, RoomType)

from shapely.geometry import Point
import json, zmq
from PIL import Image

np.float = float
np.int = int
np.bool = bool

class Roomcentre(embodied.Wrapper):

    def __init__(self, *args, **kwargs):
        self.logdir = kwargs["logdir"]
        actions = action_mapping

        # Actions
        actions = actions.copy()
        #if "STOP" in actions:
        #    actions.pop("STOP")  # remove STOP action because that will be treated differently

        length = kwargs.pop('length', 36000)
        env = AI2ThorBase(actions, *args, **kwargs, env_type="RoomCentreFinder")
        self.unwrapped_env = env
        env = embodied.wrappers.TimeLimit(env, length)
        super().__init__(env)

    def step(self, action):
        obs = self.env.step(action)
        reward = 0.0
        obs['reward'] = np.float32(reward)

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

        length = kwargs.pop('length', 36000)
        #print("AE: len", length)
        env = AI2ThorBase(actions, *args, **kwargs, env_type="DoorFinder")
        self.unwrapped_env = env
        env = embodied.wrappers.TimeLimit(env, length)
        super().__init__(env)
        #print("DI2")

    def step(self, action):
        #print("A1")
        obs = self.env.step(action)
        reward = 0
        obs['reward'] = np.float32(reward)
        #print("A2")

        # we may not want to train on distance_left parameter, but if we pop it, then wrappers complain,
        # so perhaps it can stay for now.
        #obs.pop("distance_left")
        return obs

        # Introduce a marker on the image that points towards the door that we want to go to. That would allow input and
        # training guidance to navigate to a specific door, not just a random door. Introduce room field in observation so that we can
        # classify target achieved when we change rooms.

class AI2ThorBase(embodied.Env):
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
                 env_index=-1,
                 env_type="RoomCentreFinder",
                 agent_type="rc",
                 server_ip='192.168.1.100',
                 server_port=9999,
                 encoding='utf-8'
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

        # AE: AI2-Thor simulation stuff
        self.atu = AI2THORUtils()

        self.logdir = logdir

        self.agent_type = agent_type

        # Dreamer stuff
        self._size = size
        self.isFirst = False
        self._step = 0

        self._obs_space = self.obs_space

        self._action_names = tuple(actions.keys())
        self._action_values = tuple(actions.values())
        message = f'Indoor Navigation action space ({len(self._action_values)}):'
        print(message, ', '.join(self._action_names))
        #print("C2")

        # Set up zmq server so that we can connect from lead computer and pass observations into here
        # Remote connection stuff
        self.port = server_port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)  # Changed from PULL to REP
        self.socket.bind(f"tcp://*:{self.port}")
        self.need_run = True
        self.env_retired = False  # in some cases we want to be able to signal to driver.py that this env does not need driving anymore. This will help with that.

        print(f"DreamerV3 navigator server running on port {self.port}, waiting for first handshake...")
        data = self.socket.recv_pyobj()  # This BLOCKS until a request arrives
        # we want it to block here until client has connected and only then to continue on and start receiving observations

        if (data['module'] == 'snp' and data['cmd'] == 'handshake'):
            print(" ...Handshake received")

        #self.socket.send_pyobj({"module": "snp", "cmd": "handshake2"})

    def run(self):
        while self.need_run:
            # 1. Receive the image data
            data = self.socket.recv_pyobj()  # This BLOCKS until a request arrives

            if (data['module'] == 'snp'):
                response = self.unpack_remote_obs(data)

            self.socket.send_pyobj(response)

    def unpack_remote_obs(self, data):
        received_array = np.frombuffer(data['pov']['bytes'], dtype=data['pov']['dtype'])
        received_img = received_array.reshape(data['pov']['shape'])[0]
        pil_image = Image.fromarray(received_img)

        obs = dict(
            reward = 0.0,
            pov = pil_image,
            is_first = np.bool(data['is_first']),
            is_last = np.bool(data['is_last']),
            is_terminal = np.bool(data['is_last']),
        )

        return obs

    @property
    def obs_space(self):
        return {
            'image': elements.Space(np.uint8, self._size + (3,)),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool)
        }

    @property
    def act_space(self):
        return {
            'action': elements.Space(np.int32, (), 0, len(self._action_values)),
            'reset': elements.Space(bool),
        }

    def step(self, action):
        action = action.copy()
        try:
            #{"action": 0, "reset": False}
            #action_cmd = {"command": "ACT", "action_bits": action}
            action_cmd = {"command": "ACT", "action_bits": {'action': int(action['action']), 'reset': bool(action['reset'])}}
            print("AE1: ", action_cmd, " ", self._step)

            try:
                self.socket.send_pyobj(action_cmd)
            except zmq.ZMQError as e:
                print(f"Error sending data: {e}")

            # Receive the image data
            try:
                data = self.socket.recv_pyobj()  # This BLOCKS until a request arrives
            except zmq.ZMQError as e:
                print(f"Error receiving data: {e}")

            if (data['module'] == 'snp'):
                obs = self.unpack_remote_obs(data)

            #print(f"-> Action metadata: {data}")
        except Exception as e:
            print(f"An error occurred3: {e}")
            #self.close_client_socket()

        if action['reset']:
            print('R', end='', sep='')

        # Now we turn the obs that was returned by the environment into obs that we use for training,
        # and to not confuse the two, make sure that 'pov' field is not there, because it should be 'image'.
        if obs: obs = self._obs(obs)
        self._step += 1
        assert 'pov' not in obs, list(obs.keys())
        return obs

    def _obs(self, obs):
        #print("_O1")
        obs = {
            'image': obs['pov'],
            'reward': np.float32(0.0), # reward will not be required because we will only be evaluatuing
            'is_first': obs['is_first'],
            'is_last': obs['is_last'],
            'is_terminal': obs['is_terminal']
        }
        for key, value in obs.items():
            space = self._obs_space[key]
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            #print("val: ", value, " space: ", space, " key: ", key, " (key, value, @dtype@, value.shape, space): ", (key, value, value.shape, space))
            assert value in space, (key, value, value.dtype, value.shape, space)
        #print("obs: ", obs)
        #print("_O2")
        return obs

    def close(self):
        #if (self.controller != None):
        #    self.controller.stop()
        pass

if __name__ == "__main__":
    rc = Roomcentre(logdir = "aaa")

    els = elements.Space(np.int32, (), 0, 3)
    act_space = {
        'action': els,
        'reset': elements.Space(bool),
    }

    for i in range(10):
        act = {k: v.sample() for k, v in act_space.items()}
        print(act)
        a = index_to_action(int(act['action']))
        print(a)
        act['action'] = int(act['action'])
        act['reset'] = False
        observation = rc.step(act)

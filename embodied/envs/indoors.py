import importlib

import embodied


class Indoors(embodied.Wrapper):

  def __init__(self, task, *args, **kwargs):
    module, cls = {
        'roomcentre': 'indoors_flat:Roomcentre',
        'door': 'indoors_flat:Door',
        'perimeter': 'indoors_flat:Perimeter'
    }[task].split(':')
    module = importlib.import_module(f'.{module}', __package__)
    cls = getattr(module, cls)
    env = cls(*args, **kwargs)
    super().__init__(env)

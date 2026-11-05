import importlib

import embodied


class IndoorsEval(embodied.Wrapper):

  def __init__(self, task, *args, **kwargs):
    module, cls = {
        'roomcentre': 'indoors_flat:Roomcentre',
        'door': 'indoors_flat:Door',
        'perimeter': 'indoors_flat:Perimeter',
        'straightpath': 'indoors_flat:StraightPath',
#        'remoteroomcentre': 'indoors_flat_remote:Roomcentre',
#        'remotedoor': 'indoors_flat_remote:Door',
        'remoteroomcentre': 'indoors_flat_zmqremote:Roomcentre',
        'remotedoor': 'indoors_flat_zmqremote:Door',
        'remoteperimeter': 'indoors_flat_zmqremote:Perimeter',
        'remotestraightpath': 'indoors_flat_zmqremote:StraightPath'
    }[task].split(':')
    module = importlib.import_module(f'.{module}', __package__)
    cls = getattr(module, cls)
    env = cls(*args, **kwargs)
    super().__init__(env)

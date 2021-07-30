# Channels background tasks in oTree
## Description
I have modified oTree to run channels background tasks with the runprodserver (+1of2 / +2of2) commands.

Background tasks can be defined very much like additional websocket consumers and websocket routes in the otree_extension dir of your project. They are then automatically picked up by oTree.

Basically, with this you can have long running processes in the background that continuously create or process data and then send it to the players over websockets.

## Requirements
You need to have redis-server running.

## Demo
```bash
git clone --recurse-submodules https://github.com/chkgk/otree-bg-mod.git
python3 -m venv venv
source venv/bin/activate
cd otree-bg-mod/otree-core
pip install -e .
cd ../oTree
otree resetdb --noinput
otree runprodserver
```

## Installation 
Head over to my fork of otree-core, then follow the development setup to use the forked and modified otree version instead of the original.

## Defining background tasks
Add an ```otree_extensions``` folder to your app. Then create ```consumers.py``` and ```routing.py```.

The demo below prints a random integer to the standard output every second. (The oTree app in this repository additionally sends it to players via websockets.)

```python
# routing.py
from .consumers import BackgroundTask

channel_name_routes = {
    "backgroundworker": BackgroundTask,
}
```

```python
# consumers.py
from channels.consumer import SyncConsumer
from time import sleep
from random import randint

class BackgroundTask(SyncConsumer):
    def random_values(self, event):
        while True:
            print(randint(0, 100))
            sleep(1)
```

Start it from anywhere in oTree:
```python
# models.py
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
channel_layer = get_channel_layer()

class Subsession(BaseSubsession):
    def creating_session(self):
        async_to_sync(channel_layer.send)(
            "backgroundworker",
            {
                "type": "random_values"
            },
        )

```

## Using devserver
It is possible to use ```devserver``` instead of ```runprodserver```, but you need to force oTree to use REDIS (make sure redis-server is running). First, set the environmental variable ```OTREE_USE_REDIS``` to ```True```, then start the backgroundworker separately.

```bash
# terminal 1
export OTREE_USE_REDIS=True
otree devserver
```

```bash
# terminal 2
export OTREE_USE_REDIS=True
otree runworker backgroundworker
```

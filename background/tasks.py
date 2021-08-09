from huey import SqliteHuey, RedisHuey
import time
import os
import redis
from mcts_.mcts_main import mcts_live_simu


# setup
REDIS_URL = os.environ.get("REDIS_URL", None)
print(REDIS_URL)
if not REDIS_URL:
    huey = SqliteHuey()
else:
    huey = RedisHuey(url=REDIS_URL)


# Task definitions
@huey.task()
def start_mcts(df, round_number):
    # make this task longer than it really is
#     action = mcts_live_simu(df, round_number)
    print('start_mcts')
    time.sleep(3)
    action = 5
    return action

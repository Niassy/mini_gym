
from  enum import Enum

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

GD_SCREEN = (352,352)

GD_TILE_WIDTH = 32
GD_TILE_HEIGHT = 32

GD_MAP_WIDTH = 11
GD_MAP_HEIGHT = 11

GD_X_LIMIT = 320
GD_Y_LIMIT = 320

class GD_Actions(Enum):

    left = 0,
    right = 1,
    up = 2,
    down = 3

MODEL_PATH = "torch_model"

#GD_GOAL_TILE = ()


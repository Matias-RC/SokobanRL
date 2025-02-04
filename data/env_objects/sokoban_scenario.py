
import numpy as np
import pickle
import SokoSource as source
from Logic import master

with open("templates/templates.pkl", "rb") as f:
    templates = pickle.load(f)


class Scenario:
    def __init__(self,height=8,width=8):

        self.logic = master(source.heuristic,source.cost)
        self.height = height
        self.width = width
    
        # Generate room
        inner_room = source.BuildRoom(self.width,self.height, templates)
        self.room = np.ones((self.height + 2, width + 2), dtype=inner_room.dtype)
        self.room[1:-1, 1:-1] = inner_room
        
    
    def fill_scenario(self):
        room = source.FillWithGoalsThenBoxes(np.argwhere(room == 0),room, 3, seed=None)
        room = source.RandomPlacePlayer(np.argwhere(room == 0), room)

        self.posPlayer = self.logic.PosOfPlayer(room)
        self.posBox = self.logic.PosOfBoxes(room)
        self.envPosGoals = self.logic.PosOfGoals(room)
        self.envPosWalls = self.logic.PosOfWalls(room)

        self.logic.posGoals = self.envPosGoals
        self.logic.posWalls = self.envPosWalls
    
    def get_objetive(self):
        return self.envPosGoals
    
    def get_pos_player(self):
        return self.posPlayer





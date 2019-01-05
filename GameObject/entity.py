
from abc import ABC,abstractmethod

class Entity(ABC):

    def __init__(self,x,y):

        self.x = x
        self.y = y

        # tracking last position
        self.lastX = x
        self.lastY = y

        # for map
        self.xMap = -1
        self.yMap = -1

    def setPosition(self, pos):

        self.lastX,self.lastY = self.x,self.y
        self.x, self.y = pos

    def getPosition(self):

        return self.x,self.y


    @abstractmethod
    def render(self,screen):

        pass

    # colllision test
    @abstractmethod
    def collide(self,entity):

        pass



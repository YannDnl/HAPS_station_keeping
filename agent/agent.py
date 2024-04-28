import random as rd

class RandomAgent:
    def __init__(self, n) -> None:
        self.n = n

    def get_action(self, inputs):
        return [rd.randint(-1, 1) for _ in range(self.n)]
    
class PassiveAgent:
    def __init__(self, n) -> None:
        self.n = n
    
    def get_action(self, inputs):
        return [0 for _ in range(self.n)]
    
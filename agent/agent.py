import random as rd

class Agent:
    def __init__(self, n) -> None:
        self.n = n

    def get_action(self, inputs):
        return [rd.randint(-1, 1) for _ in range(self.n)]
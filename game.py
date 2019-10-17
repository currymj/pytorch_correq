import torch
import itertools

class ChickenGame:
    def __init__(self):
        # K is feature/utility vector dimension
        self.K = 3
        self.num_players = 2
        self.player_action_dims = (2,2)

    def features(self, action_tuple):
        p1, p2 = action_tuple
        # 0 is drive, 1 is swerve
        # for utility vectors first dim is crash, second dim is look cool, third dim is look like a wimp
        if p1 == 0:
            if p2 == 0:
                return torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            if p2 == 1:
                return torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        elif p1 == 1:
            if p2 == 0:
                return torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
            if p2 == 1:
                return torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    def enumerate_joint_actions(self):
        return itertools.product(*[range(d) for d in self.player_action_dims])

# explicit payoffs for util vector [-5.0, 1.0, 0.0], for reference
chicken_payoffs = torch.tensor([
    [[-5.0, 1.0], [0.0, 0.0]],
    [[-5.0, 0.0], [1.0, 0.0]]
])
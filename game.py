import torch
import itertools


class RoutingGame:
    def __init__(self, N):
        self.K = 4
        self.num_players = N
        self.player_action_dims = (4, 4, 4, 4)

    def features(self, action_tuple):
        v = torch.tensor([1.5 + .2*self.num_players, 9.0, 1.0/8.0 + 0.04*self.num_players, 7 + 0.4*self.num_players])
        left = 0
        back = 0
        for action in action_tuple:
            if action % 2 == 0:
                left += 1
            if action // 2 == 0:
                back += 1

        payoffs = torch.zeros(self.num_players, self.K)
        for player in range(self.num_players):
            if action_tuple[player] % 2 == 0:
                u = v + torch.tensor([1.0, 1.5, 1.0/20.0, 2.0])
            else:
                u = v + torch.tensor([1.0+2.0*left, 1.0, 1.0/20.0 + 0.04*left, 1.5+.4*left])
            if action_tuple[player] // 2 == 0:
                u = v + torch.tensor([6.0+0.5*back, 12.0, 1.0/7.0 + 0.01*back, 10.0+3.0*back])
            else:
                u = v + torch.tensor([2.0, 20.0, 1.0/8.0, 15.0])
            payoffs[player] = -u

        return payoffs

    def enumerate_joint_actions(self):
        return itertools.product(*[range(d) for d in self.player_action_dims])


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
import torch

def chicken_feats(action_tuple):
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


# explicit payoffs for util vector [-5.0, 1.0, 0.0], for reference
chicken_payoffs = torch.tensor([
    [[-5.0, 1.0], [0.0, 0.0]],
    [[-5.0, 0.0], [1.0, 0.0]]
])
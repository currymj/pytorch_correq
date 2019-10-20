import torch
import itertools

class FixedDeviations:
    def __init__(self, player_action_dims):
        self.player_action_dims = player_action_dims

    def deviations_dim(self):
        # problematic, not general enough
        return (len(self.player_action_dims), self.player_action_dims[0])

    def enumerator(self):
        def e():
            for i in range(len(self.player_action_dims)):
                for j in range(self.player_action_dims[i]):
                    yield (i, j)

        return e

    def apply_deviation(self, action_tens, deviation):
        new_action_tens = torch.clone(action_tens)
        player, action = deviation
        new_action_tens[player] = action
        return new_action_tens


class SwitchDeviations:
    """
    This class represents the set of switch deviations, of the form
    switch(player, action1, action2), which causes player 1 to replace action1
    with action2. Therefore the deviations are represented simply as tuples
    of that form.
    """
    def __init__(self, player_action_dims):
        self.player_action_dims = player_action_dims

    def deviations_dim(self):
        return len(self.player_action_dims), self.player_action_dims[0], self.player_action_dims[0]

    def enumerator(self):
        def e():
            for i in range(len(self.player_action_dims)):
                for j in range(self.player_action_dims[i]):
                    for k in range(self.player_action_dims[i]):
                        yield (i, j, k)
        return e

    def apply_deviation(self, action_tens, deviation):
        new_action_tens = torch.clone(action_tens)
        player, actionx, actiony = deviation
        if new_action_tens[player] == actionx:
            new_action_tens[player] = actiony
        return new_action_tens


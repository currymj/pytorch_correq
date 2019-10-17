import torch
import itertools

def external_enumerator(player_action_dims):
    def e():
        for i in range(len(player_action_dims)):
            for j in range(player_action_dims[i]):
                yield (i, j)

    return e

def switch_enumerator(player_action_dims):
    def e():
        for i in range(len(player_action_dims)):
            for j in range(player_action_dims[i]):
                for k in range(player_action_dims[i]):
                    yield (i, j, k)
    return e

def apply_external_deviation(action_tens, deviation):
    new_action_tens = torch.clone(action_tens)
    player, action = deviation
    new_action_tens[player] = action
    return new_action_tens

def apply_switch_deviation(action_tens, deviation):
    new_action_tens = torch.clone(action_tens)
    player, actionx, actiony = deviation
    if new_action_tens[player] == actionx:
        new_action_tens[player] = actiony
    return new_action_tens

import torch
from inverse_correq import InverseCorrelatedEquilibriumProblem
from game import ChickenGame
from deviations import SwitchDeviations

def optimize_analytic(prob_obj, theta, epochs=100, lr=0.1):
    for i in range(epochs):
        g = prob_obj.analytic_gradient(theta)
        theta -= lr*g
    return theta

if __name__ == '__main__':
    player_action_dims = (2,2)
    corr_chicken = torch.tensor([[0.0, 0.4], [0.4, 0.2]])
    corr_chicken_approx = torch.tensor([[0.0,0.42], [0.38, 0.2]])
    correq_theta = torch.zeros(2,2,2,3, requires_grad=False)
    chicken_obj_int = InverseCorrelatedEquilibriumProblem(ChickenGame(),
                                                          corr_chicken_approx,
                                                          SwitchDeviations(player_action_dims))
    print('observed strategy', corr_chicken_approx)
    print('strategy at initialization', chicken_obj_int.predicted_strategy(correq_theta))
    optimize_analytic(chicken_obj_int, correq_theta, epochs=10000, lr=0.1)
    print('Predicted strategy', chicken_obj_int.predicted_strategy(correq_theta))
    print('true strategy', corr_chicken)


    corr_chicken = torch.tensor([[0.0, 0.5], [0.5, 0.0]])
    corr_chicken_approx = torch.tensor([[0.0, 0.46], [0.54, 0.0]])

    correq_theta = torch.ones(2, 2, 2, 3)

    chicken_obj_int = InverseCorrelatedEquilibriumProblem(ChickenGame(),
                                                          corr_chicken_approx,
                                                          SwitchDeviations(player_action_dims))

    print('observed strategy', corr_chicken_approx)
    print('strategy at initialization', chicken_obj_int.predicted_strategy(correq_theta))
    optimize_analytic(chicken_obj_int, correq_theta, epochs=10000, lr=0.1)
    print('Predicted strategy', chicken_obj_int.predicted_strategy(correq_theta))
    print('true strategy', corr_chicken)

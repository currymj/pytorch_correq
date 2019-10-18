import torch
from inverse_correq import InverseCorrelatedEquilibriumProblem
from game import ChickenGame
from deviations import SwitchDeviations
import math

def logloss(truth, prediction, epsilon=0.0):
    loss = 0.0
    truth_flat = truth.view(-1)
    prediction_flat = prediction.view(-1)
    N = len(truth_flat)
    for i in range(N):
        if truth_flat[i] > 1e-15:
            loss -= truth_flat[i] * torch.log((1-epsilon)*prediction_flat[i] + epsilon/N)
    return loss/math.log(2)

def optimize_analytic(prob_obj, theta, epochs=100, lr=0.1):
    for i in range(epochs):
        g = prob_obj.analytic_gradient(theta)
        theta -= lr*g
    return theta

def sampled_observed_dist(true_dist, N=100):
    # sample N points, sum
    true_dist_shape = true_dist.shape
    true_dist_flat = true_dist.view(-1)
    total_dist = torch.zeros_like(true_dist_flat)
    sampled_indices = torch.multinomial(true_dist_flat, num_samples=N, replacement=True)
    for index in sampled_indices:
        total_dist[index] += 1.0
    return (total_dist / N).view(true_dist_shape)


if __name__ == '__main__':
    player_action_dims = (2,2)
    eps = 0.01
    corr_chicken = torch.tensor([[0.0, 0.4], [0.4, 0.2]])
    corr_chicken_approx = sampled_observed_dist(corr_chicken, N=50)
    correq_theta = torch.zeros(2,2,2,3, requires_grad=False)
    chicken_obj_int = InverseCorrelatedEquilibriumProblem(ChickenGame(),
                                                          corr_chicken_approx,
                                                          SwitchDeviations(player_action_dims))
    print('observed strategy', corr_chicken_approx)
    print('strategy at initialization', chicken_obj_int.predicted_strategy(correq_theta))
    optimize_analytic(chicken_obj_int, correq_theta, epochs=20000, lr=1.0)
    predicted_strategy = chicken_obj_int.predicted_strategy(correq_theta)
    print('Predicted strategy', predicted_strategy)
    print('true strategy', corr_chicken)
    print('multinomial loss', logloss(corr_chicken, corr_chicken_approx, epsilon=eps))
    print('ice loss', logloss(corr_chicken, predicted_strategy, epsilon=eps))

    corr_chicken = torch.tensor([[0.0, 0.5], [0.5, 0.0]])
    corr_chicken_approx = sampled_observed_dist(corr_chicken, N=50)

    correq_theta = torch.ones(2, 2, 2, 3)

    chicken_obj_int = InverseCorrelatedEquilibriumProblem(ChickenGame(),
                                                          corr_chicken_approx,
                                                          SwitchDeviations(player_action_dims))

    print('observed strategy', corr_chicken_approx)
    print('strategy at initialization', chicken_obj_int.predicted_strategy(correq_theta))
    optimize_analytic(chicken_obj_int, correq_theta, epochs=20000, lr=1.0)
    predicted_strategy = chicken_obj_int.predicted_strategy(correq_theta)
    print('Predicted strategy', predicted_strategy)
    print('true strategy', corr_chicken)
    print('multinomial loss', logloss(corr_chicken, corr_chicken_approx, epsilon=eps))
    print('ice loss', logloss(corr_chicken, predicted_strategy, epsilon=eps))

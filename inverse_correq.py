import torch
import itertools


class InverseCorrelatedEquilibriumProblem:
    """
    This class takes a game object, an empirical strategy, and an object representing the set of deviations,
    and performs computations related to the maxent inverse correlated equilibrium optimization problem.

    Most of the work is done on arrays associated with the dual variables. The shape of these arrays varies
    depending on the game and deviation set, but in general they are of shape

                    (num_players x ...deviations... x feature_dim)

    For example, for a 2-player game, with 2 actions, and features of dimension 3, with switch deviations,
    these tensors are of size 2 x 2 x 2 x 3.
    """

    def __init__(self,
                 game,
                 observed_strategy,
                 deviations):
        self.game = game
        self.observed_strategy = observed_strategy
        self.deviations = deviations
        self.memoize_regrets_dict = {}

        # preallocation slots

        self._g = None
        self._regret_feats_observed = self.compute_expected_regret_feats(self.observed_strategy)
        self._predicted = None

    def predicted_strategy(self, theta):
        """computes a predicted strategy given values for the dual variables."""
        unnormalized_dist = torch.zeros(*self.game.player_action_dims)
        # dot product of each regret feat with each theta
        for joint_action in self.game.enumerate_joint_actions():
            action_regret_feats = self.compute_phi_regrets_for_action(torch.tensor(list(joint_action)))
            action_regret_scalars = torch.sum(action_regret_feats * theta, dim=len(theta.shape) - 1)
            unnormalized_dist[joint_action] = torch.exp(-torch.sum(action_regret_scalars))
        Z = torch.sum(unnormalized_dist)
        return unnormalized_dist / Z

    def predicted_strategy_(self, unnormalized_dist, theta):
        """computes a predicted strategy in-place on a preallocated array"""
        for joint_action in self.game.enumerate_joint_actions():
            action_regret_feats = self.compute_phi_regrets_for_action(torch.tensor(list(joint_action)))
            action_regret_scalars = torch.sum(action_regret_feats * theta, dim=len(theta.shape) - 1)
            unnormalized_dist[joint_action] = torch.exp(-torch.sum(action_regret_scalars))
        Z = torch.sum(unnormalized_dist)
        return unnormalized_dist / Z

    def compute_phi_regrets_for_action(self, action_tens):
        # this function compute per-action regrets, recording them in a dictionary so
        # they are not recomputed more than once
        key = tuple(action_tens.numpy())
        if key in self.memoize_regrets_dict:
            return self.memoize_regrets_dict[key]

        # these are the instantaneous regrets for all the specific deviations
        regret_feats = torch.zeros(*self.deviations.deviations_dim(), self.game.K, requires_grad=False)
        dev_iter = self.deviations.enumerator()
        for deviation in dev_iter():
            deviation_applied = self.deviations.apply_deviation(action_tens, deviation)
            # get regrets for specific player only (player is specified by 0 of deviation)
            regret_feats[deviation] = self.game.features(deviation_applied)[deviation[0]] - \
                                      self.game.features(action_tens)[deviation[0]]
        self.memoize_regrets_dict[key] = regret_feats
        return regret_feats

    def compute_expected_regret_feats(self, action_dist):
        total_regret_feats = torch.zeros(*self.deviations.deviations_dim(), self.game.K, requires_grad=False)
        n = 0
        for joint_action in self.game.enumerate_joint_actions():
            n += 1
            total_regret_feats += action_dist[joint_action] * self.compute_phi_regrets_for_action(
                torch.tensor(list(joint_action)))
        return total_regret_feats / n

    def analytic_gradient(self, theta):
        dev_iter = self.deviations.enumerator()
        if self._predicted is not None:
            self.predicted_strategy_(self._predicted, theta)
        else:
            self._predicted = (self.predicted_strategy(theta))
        regret_feats_predicted = self.compute_expected_regret_feats(self._predicted)
        if self._g is not None:
            self._g.zero_()
        else:
            self._g = torch.zeros_like(theta, requires_grad=False)
        for deviation in dev_iter():
            this_deviation_theta = theta[deviation].view(*[1 for _ in deviation], -1)  # unsqueeze to broadcast
            # sorry that is a really hacky way to do it, but i think it does what we want
            # i.e. add one empty dim for all dims of deviations, then -1 for the dim that is size K
            little_scalar_regrets = torch.sum(self._regret_feats_observed * this_deviation_theta, dim=len(theta.shape) - 1)
            #  now argmax

            fstar_ind = little_scalar_regrets.argmax()

            self._g[deviation] = self._regret_feats_observed.view(-1, self.game.K)[fstar_ind] - regret_feats_predicted[deviation]
        return self._g

    def maxent_dual_objective(self, theta):
        bigZ = torch.tensor(0.0, requires_grad=True)

        # for each joint action in A
        for joint_action in self.game.enumerate_joint_actions():
            little_r_a_feats = self.compute_phi_regrets_for_action(torch.tensor(list(joint_action)))
            # scalar features for all deviations f with their own theta_fs
            little_r_a_scalar = torch.sum(little_r_a_feats * theta, dim=len(theta.shape) - 1)
            # sum up, exp, add to Z
            bigZ = bigZ + torch.exp(-torch.sum(little_r_a_scalar))
        obj = torch.log(bigZ)
        # computing expected big regret for theta_f is max over phi_f of r_f(predicted | theta_f)
        # phi_f here is just the whole phi
        expected_er_feats = self.compute_expected_regret_feats(self.observed_strategy)

        # for each deviation
        dev_iter = self.deviations.enumerator()
        for deviation in dev_iter():
            this_deviation_theta = theta[deviation].view(*[1 for _ in deviation], -1)  # unsqueeze to broadcast
            # sorry that is a really hacky way to do it, but i think it does what we want
            # i.e. add one empty dim for all dims of deviations, then -1 for the dim that is size K
            little_scalar_regrets = torch.sum(expected_er_feats * this_deviation_theta, dim=len(theta.shape) - 1)
            # little_scalar_regrets contains the regret for theta_f for all the different fs
            big_Regret = torch.max(little_scalar_regrets)
            obj = obj + big_Regret
        return obj



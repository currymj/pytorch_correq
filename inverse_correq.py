import torch
import itertools


class InverseCorrelatedEquilibriumProblem:

    def __init__(self,
                 K,
                 player_action_dims,
                 observed_strategy,
                 payoff_features,
                 deviations_dim,
                 get_deviation_iter,
                 apply_deviation):
        self.num_players = len(player_action_dims)
        self.player_action_dims = player_action_dims
        self.observed_strategy = observed_strategy
        self.payoff_features_fn = payoff_features
        self.deviations_dim = deviations_dim
        self.get_deviation_iter = get_deviation_iter
        self.apply_deviation_fn = apply_deviation
        self.memoize_regrets_dict = {}
        assert self.deviations_dim[0] == self.num_players
        self.K = K

    def enumerate_joint_actions(self):
        return itertools.product(*[range(d) for d in self.player_action_dims])

    def predicted_strategy(self, theta):
        unnormalized_dist = torch.zeros(*self.player_action_dims)
        # dot product of each regret feat with each theta
        for joint_action in self.enumerate_joint_actions():
            action_regret_feats = self.compute_phi_regrets_for_action(torch.tensor(list(joint_action)))
            action_regret_scalars = torch.sum(action_regret_feats * theta, dim=len(theta.shape) - 1)
            unnormalized_dist[joint_action] = torch.exp(-torch.sum(action_regret_scalars))
        Z = torch.sum(unnormalized_dist)
        return unnormalized_dist / Z

    def compute_phi_regrets_for_action(self, action_tens):
        key = tuple(action_tens.numpy())
        if key in self.memoize_regrets_dict:
            return self.memoize_regrets_dict[key]

        # these are the instantaneous regrets for all the specific deviations
        regret_feats = torch.zeros(*self.deviations_dim, self.K, requires_grad=False)
        dev_iter = self.get_deviation_iter(self.player_action_dims)
        for deviation in dev_iter():
            deviation_applied = self.apply_deviation_fn(action_tens, deviation)
            # get regrets for specific player only (player is specified by 0 of deviation)
            regret_feats[deviation] = self.payoff_features_fn(deviation_applied)[deviation[0]] - \
                                      self.payoff_features_fn(action_tens)[deviation[0]]
        self.memoize_regrets_dict[key] = regret_feats
        return regret_feats

    def compute_expected_regret_feats(self, action_dist):
        total_regret_feats = torch.zeros(*self.deviations_dim, self.K, requires_grad=False)
        n = 0
        for joint_action in self.enumerate_joint_actions():
            n += 1
            total_regret_feats += action_dist[joint_action] * self.compute_phi_regrets_for_action(
                torch.tensor(list(joint_action)))
        return total_regret_feats / n

    def analytic_gradient(self, theta):
        dev_iter = self.get_deviation_iter(self.player_action_dims)
        regret_feats_observed = self.compute_expected_regret_feats(self.observed_strategy)
        regret_feats_predicted = self.compute_expected_regret_feats(self.predicted_strategy(theta))
        g = torch.zeros_like(theta, requires_grad=False)
        for deviation in dev_iter():
            this_deviation_theta = theta[deviation].view(*[1 for _ in deviation], -1)  # unsqueeze to broadcast
            # sorry that is a really hacky way to do it, but i think it does what we want
            # i.e. add one empty dim for all dims of deviations, then -1 for the dim that is size K
            little_scalar_regrets = torch.sum(regret_feats_observed * this_deviation_theta, dim=len(theta.shape) - 1)
            #  now argmax

            fstar_ind = little_scalar_regrets.argmax()

            g[deviation] = regret_feats_observed.view(-1, self.K)[fstar_ind] - regret_feats_predicted[deviation]
        return g

    def maxent_dual_objective(self, theta):
        bigZ = torch.tensor(0.0, requires_grad=True)

        # for each joint action in A
        for joint_action in self.enumerate_joint_actions():
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
        dev_iter = self.get_deviation_iter(self.player_action_dims)
        for deviation in dev_iter():
            this_deviation_theta = theta[deviation].view(*[1 for _ in deviation], -1)  # unsqueeze to broadcast
            # sorry that is a really hacky way to do it, but i think it does what we want
            # i.e. add one empty dim for all dims of deviations, then -1 for the dim that is size K
            little_scalar_regrets = torch.sum(expected_er_feats * this_deviation_theta, dim=len(theta.shape) - 1)
            # little_scalar_regrets contains the regret for theta_f for all the different fs
            big_Regret = torch.max(little_scalar_regrets)
            obj = obj + big_Regret
        return obj



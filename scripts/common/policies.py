from stable_baselines.common.policies import ActorCriticPolicy, register_policy
from scripts.common.distributions import BoundedDiagGaussianDistributionType
from stable_baselines.a2c.utils import linear, ortho_init
from scripts.common.utils import log
from scripts.common import config as cfg
import tensorflow as tf
import numpy as np


class CustomPolicy(ActorCriticPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, **kwargs)
        log("Using CustomPolicy.")

        if cfg.is_mod(cfg.MOD_TANH_ACTS):
            self._pdtype = BoundedDiagGaussianDistributionType(ac_space.shape[0])
            log("Using Bounded Gaussian Distribution")

        with tf.variable_scope("model", reuse=reuse):
            obs = self.processed_obs
            act_func_hid = tf.nn.relu

            # build the policy network's hidden layers
            pi_h = self.fc_hidden_layers('pi_fc_hid', obs, cfg.hid_layer_sizes, act_func_hid)
            # build the value network's hidden layers
            vf_h = self.fc_hidden_layers('vf_fc_hid', obs, cfg.hid_layer_sizes, act_func_hid)
            # build the output layer of the policy (init_scale as proposed by stable-baselines)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_h, vf_h, init_scale=0.01)
            # build the output layer of the value function
            vf_out = self.fc('vf_out', vf_h, 1)
            self._value_fn = vf_out
            # required to set up additional attributes
            self._setup_init()

    def fc_hidden_layers(self, name, input, hid_sizes, act_func):
        """Fully connected MLP. Number of layers determined by len(hid_sizes)."""
        hid = input
        for i, size in enumerate(hid_sizes):
            hid = act_func(self.fc(f'{name}{i}', hid, size))
        return hid

    def fc(self, name, input, size):
        """Builds a single fully connected layer. Initial values taken from stable-baselines."""
        return linear(input, name, size, init_scale=np.sqrt(2), init_bias=0)

    # ----------------------------------------------
    # Obligatory method definitions from stable-baselines (proba_step, step, value)
    # cf https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html
    # ----------------------------------------------

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

# Register the policy
register_policy('CustomPolicy', CustomPolicy)
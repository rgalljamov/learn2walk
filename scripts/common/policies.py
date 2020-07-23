from stable_baselines.common.policies import ActorCriticPolicy, register_policy
from scripts.common.distributions import BoundedDiagGaussianDistributionType, \
    CustomDiagGaussianDistributionType
from stable_baselines.a2c.utils import linear, ortho_init
from scripts.behavior_cloning.models import load_weights
from scripts.common.utils import log
from scripts.common import config as cfg
import tensorflow as tf
import numpy as np


class CustomPolicy(ActorCriticPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, **kwargs)
        log("Using CustomPolicy.")

        if cfg.is_mod(cfg.MOD_PRETRAIN_PI):
            self._pdtype = CustomDiagGaussianDistributionType(ac_space.shape[0])
            log("Using Custom Gaussian Distribution\nwith pretrained mean weights and biases!")
        elif cfg.is_mod(cfg.MOD_BOUND_MEAN) or cfg.is_mod(cfg.MOD_SAC_ACTS):
            self._pdtype = BoundedDiagGaussianDistributionType(ac_space.shape[0])
            log("Using Bounded Gaussian Distribution")

        with tf.variable_scope("model", reuse=reuse):
            obs = self.processed_obs
            act_func_hid = tf.nn.relu

            # build the policy network's hidden layers
            if cfg.is_mod(cfg.MOD_PRETRAIN_PI):
                pi_h = self.load_pretrained_policy_hid_layers('pi_fc_hid', obs, act_func_hid)
                log('Loading pretrained policy HIDDEN LAYER weights!')
            else:
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

    def load_pretrained_policy_hid_layers(self, scope, input, act_func=tf.nn.relu):
        """
        Loads the hidden layers of the pretrained policy model (Behavior Cloning).
        Adopted linear layer from stable_baselines/a2c/utils.py.
        :param input: (tf tensor) The input tensor for the fully connected layer
        :param scope: (str) The TensorFlow variable scope
        """

        input_dim = input.get_shape()[1].value
        # load weights
        w_hid1, b_hid1, w_hid2, b_hid2, w_out, b_out = load_weights()
        # check dimensions
        assert input_dim == w_hid1.shape[0]
        hid_layer_sizes = [b_hid1.size, b_hid2.size]
        assert cfg.hid_layer_sizes == hid_layer_sizes
        # organize weights and biases in lists
        ws = [w_hid1, w_hid2]
        bs = [b_hid1, b_hid2]

        # construct the hidden layers with relu activation
        for i, n_hidden in enumerate(hid_layer_sizes):
            with tf.variable_scope(f'{scope}{i}'):
                weight = tf.get_variable("w", initializer=ws[i])
                bias = tf.get_variable("b",initializer=bs[i])
                output = act_func(tf.matmul(input, weight) + bias)
                input = output

        return output

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
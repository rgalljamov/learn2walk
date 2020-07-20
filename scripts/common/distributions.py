import tensorflow as tf
from scripts.common import config as cfg
from scripts.common.utils import log
from stable_baselines.common.tf_layers import linear
from stable_baselines.common.distributions import \
    DiagGaussianProbabilityDistribution, DiagGaussianProbabilityDistributionType

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class BoundedDiagGaussianDistribution(DiagGaussianProbabilityDistribution):
    def __init__(self, flat):
        super(BoundedDiagGaussianDistribution, self).__init__(flat)

    def neglogp(self, sampled_action):
        """
        Computes log[pi(a|s)] of a given sampled action a.
        """
        neg_log_pi = super(BoundedDiagGaussianDistribution, self).neglogp(sampled_action)
        if cfg.is_mod(cfg.MOD_SAC_ACTS):
            log('Using custom distribution with SAC neglogp.')
            from stable_baselines.sac.policies import clip_but_pass_gradient
            # account for squashing the sampled action by a tahn
            if cfg.DEBUG: print('neg_log_pi:',neg_log_pi)
            neg_log_pi += tf.reduce_sum(
                tf.log(clip_but_pass_gradient(1 - tf.tanh(sampled_action) ** 2, 0, 1) + 1e-6), axis=-1)
        return neg_log_pi

    def sample(self):
        sampled_action = super(BoundedDiagGaussianDistribution, self).sample()
        if cfg.is_mod(cfg.MOD_SAC_ACTS):
            log('Using custom distribution with custom SAC sampling!')
            sampled_action = tf.tanh(sampled_action)
        return sampled_action

    # def entropy(self):
    #     """Return constant value as entropy is not used."""
    #     return 1.0

    # def kl(self, other):
    #     """Also KL is not used during training, only monitoring."""
    #     return 3.33e-3

class BoundedDiagGaussianDistributionType(DiagGaussianProbabilityDistributionType):
    def __init__(self, size):
        self.size = size

    def probability_distribution_class(self):
        return BoundedDiagGaussianDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        mean = linear(pi_latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        if cfg.is_mod(cfg.MOD_BOUND_MEAN):
            with tf.variable_scope('pi'):
                mean = tf.tanh(mean)  # squashing mean only
        logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.zeros_initializer())
        # inspired by sac
        logstd = tf.clip_by_value(logstd, LOG_STD_MIN, LOG_STD_MAX)
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        q_values = linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), mean, q_values

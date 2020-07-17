import tensorflow as tf
from stable_baselines.common.tf_layers import linear
from stable_baselines.common.distributions import \
    DiagGaussianProbabilityDistribution, DiagGaussianProbabilityDistributionType

class BoundedDiagGaussianDistribution(DiagGaussianProbabilityDistribution):
    def __init__(self, flat):
        super(BoundedDiagGaussianDistribution, self).__init__(flat)

class BoundedDiagGaussianDistributionType(DiagGaussianProbabilityDistributionType):
    def __init__(self, size):
        self.size = size

    def probability_distribution_class(self):
        return BoundedDiagGaussianDistribution

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        mean = linear(pi_latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
        logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        q_values = linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), mean, q_values

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Dense

from utils import open_config

tf.keras.backend.set_floatx('float64')

tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers

class Symmetrize(tf.keras.layers.Layer):
   """
   Custom Keras Layer to symmetrize input tensors for Mixture Density Networks (MDN).
   This layer creates two matrices (`mat_a` and `mat`) that are used to transform the input tensor
   in a way that enforces symmetry constraints.
   Args:
      nr_gaussians (int): Number of Gaussian components in the mixture model.
      param_size (int): Size of the parameter vector for each Gaussian.
   Attributes:
      nr_gaussians (int): Stores the number of Gaussians.
      params_size (int): Stores the parameter size.
      mat_a (tf.Tensor): Matrix used for symmetric transformation.
      mat (tf.Tensor): Matrix used for alternate diagonal transformation.
   Methods:
      call(inputs):
         Applies the symmetric transformations to the input tensor.
      create_sym_matrix():
         Constructs the permutation matrices (`mat_a` and `mat`) used in the layer.
   """
   def __init__(self, nr_gaussians, param_size):
      super(Symmetrize, self).__init__()     
      self.nr_gaussians = nr_gaussians
      self.nr_gauss_sym = self.nr_gaussians // 2
      self.params_size = param_size
      self.mat_alpha, self.mat_mu_sigma, self.mat_mu, self.mat_sigma = self.create_sym_matrix()

   def call(self, x, inputs):
      # Gather eps; eps_tr = inputs[:, 1], eps_rA = inputs[:, 2]
      inputs = tf.pad(inputs[:, 1:3], [[0,0],[0,2]])
      
      # Apply the symmetric transformations
      alpha = tf.linalg.matmul(x, self.mat_alpha) 
      mu_sigma = tf.linalg.matmul(x, self.mat_mu_sigma)
      mu = tf.linalg.matmul(x, self.mat_mu)   
      sigma = tf.linalg.matmul(x, self.mat_sigma)  
      # k = tf.linalg.matmul(inputs, self.mat)
      mu_sym = 2 * tf.tile(inputs, [1, self.nr_gauss_sym]) - mu

      mu_sigma_sym = mu_sym + sigma
      
      return tf.concat([alpha, mu_sigma, mu_sigma_sym], axis=1)   

   def create_sym_matrix(self):      
      # Create permutation matrix for weight paramater
      mat_alpha = tf.pad(tf.eye(self.nr_gauss_sym), [[0, self.params_size - self.nr_gauss_sym], [0, 0]])
      mat_alpha  = tf.concat((mat_alpha, mat_alpha), axis=1)

      # Create matrix for mu and sigma
      mat_mu_sigma = tf.eye(self.params_size-self.nr_gauss_sym)
      # alt_diag = tf.linalg.diag(tf.tile([-1.0, -1.0, 1.0, 1.0], [nr_gauss_sym])) # mirror mu_1, mu_2 around 0
      # mat = tf.concat((i_mat, alt_diag), axis=1)
      mat_mu_sigma = tf.pad(mat_mu_sigma, [[self.nr_gauss_sym, 0], [0, 0]])
      
      diag_mu = tf.linalg.diag(tf.tile([1.0, 1.0, 0.0, 0.0], [self.nr_gauss_sym]))
      mat_mu = tf.pad(diag_mu, [[self.nr_gauss_sym, 0], [0,0]])
      
      diag_sigma = tf.linalg.diag(tf.tile([0.0, 0.0, 1.0, 1.0], [self.nr_gauss_sym]))
      # mat_sigma = tf.concat((diag_sigma, diag_sigma), axis=1)
      mat_sigma = tf.pad(diag_sigma, [[self.nr_gauss_sym,0], [0,0]])
      
      mat_alpha = tf.cast(mat_alpha, tf.float64)
      mat_mu_sigma = tf.cast(mat_mu_sigma, tf.float64)
      mat_mu = tf.cast(mat_mu, tf.float64)
      mat_sigma = tf.cast(mat_sigma, tf.float64)
            
      return mat_alpha, mat_mu_sigma, mat_mu, mat_sigma

class MDN(tf.keras.Model):
   def __init__(self, nr_gaussians, nr_neurons, activation_function, symmetrize=False, include_b=False):
      super(MDN, self).__init__()
      self.nr_gaussians = nr_gaussians
      self.symmetrize = symmetrize
      self.include_b = include_b
      
      self.params_size = tfpl.MixtureSameFamily.params_size(self.nr_gaussians, component_params_size=tfpl.IndependentNormal.params_size(2))
      
      if self.symmetrize:
         self.params_size = self.params_size //2 # half output of MLP
         
      self.hidden1 = Dense(nr_neurons, activation=activation_function)
      self.hidden2 = Dense(self.params_size, activation=None)
      
      # Initialize Symmetrize layer after dense layers, for correct use in dsmc.py
      if self.symmetrize:
         self.sym = Symmetrize(self.nr_gaussians, self.params_size)
      
      self.mdn = tfpl.MixtureSameFamily(self.nr_gaussians, tfpl.IndependentNormal(event_shape=[2]))
        
   def call(self, inputs):
      x = self.hidden1(inputs)
      x = self.hidden2(x)
      
      if self.symmetrize:
         x = self.sym(x, inputs)
      
      return self.mdn(x)
   

def build_model(nr_gaussians=20, activation_function='relu', nr_neurons=8, learning_rate=1e-4, symmetrize=False, include_b=False): 
    negloglik = lambda y, p_y: -p_y.log_prob(y)

    model = MDN(nr_gaussians, nr_neurons, activation_function, symmetrize=symmetrize, include_b=include_b)
    
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=negloglik)

    return model
 
def load_model(model_path, config_path=None):
   if config_path is None:
      config_path = model_path.replace('.h5', '_config.json')
   
   config = open_config(config_path)
   model = build_model(config.get("nr_gaussians"), 
                       config.get("activation_function"), 
                       config.get("nr_neurons"), 
                       symmetrize=config.get("symmetrize"), 
                       include_b=config.get("include_b"))
   
   # Initialize model
   if config.get("include_b"):
      x = np.ones((1,4))
   else:
      x = np.ones((1,3))
   model(x)  
   
   model.load_weights(model_path)
   model.summary()
   return model
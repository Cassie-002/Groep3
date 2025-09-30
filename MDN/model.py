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
   def __init__(self, nr_gaussians, param_size):
      super(Symmetrize, self).__init__()     
      self.nr_gaussians = nr_gaussians
      self.params_size = param_size
      self.mat_a, self.mat = self.create_sym_matrix()

   def call(self, inputs):
      a = tf.linalg.matmul(inputs, self.mat_a)
      k = tf.linalg.matmul(inputs, self.mat)
      return tf.concat([a, k], axis=1)   

   def create_sym_matrix(self):
      nr_gauss_sym = self.nr_gaussians // 2
      mat_a = tf.pad(tf.eye(nr_gauss_sym), [[0, self.params_size - nr_gauss_sym], [0, 0]])

      mat_a  = tf.concat((mat_a, mat_a), axis=1)

      i_mat = tf.eye(self.params_size-nr_gauss_sym)
      alt_diag = tf.linalg.diag(tf.tile([-1.0, -1.0, 1.0, 1.0], [nr_gauss_sym]))
      mat = tf.concat((i_mat, alt_diag), axis=1)
      mat = tf.pad(mat, [[nr_gauss_sym,0], [0,0]])

      mat_a = tf.cast(mat_a, tf.float64)
      mat = tf.cast(mat, tf.float64)
      
      return mat_a, mat  

class MDN(tf.keras.Model):
   def __init__(self, nr_gaussians, nr_neurons, activation_function, symmetrize=False):
      super(MDN, self).__init__()
      self.nr_gaussians = nr_gaussians
      self.symmetrize = symmetrize
      
      self.params_size = tfpl.MixtureSameFamily.params_size(self.nr_gaussians, component_params_size=tfpl.IndependentNormal.params_size(2))
      if self.symmetrize:
         self.params_size = self.params_size //2 
         
      self.hidden1 = Dense(nr_neurons, activation=activation_function)
      self.hidden2 = Dense(self.params_size, activation=activation_function)
      
      if self.symmetrize:
         self.sym = Symmetrize(self.nr_gaussians, self.params_size)
      
      self.mdn = tfpl.MixtureSameFamily(self.nr_gaussians, tfpl.IndependentNormal(event_shape=[2]))
        
   def call(self, inputs):
      x = self.hidden1(inputs)
      x = self.hidden2(x)
      
      if self.symmetrize:
         x = self.sym(x)
      
      return self.mdn(x)
            
   

def build_model(nr_gaussians=20, activation_function='relu', nr_neurons=8, learning_rate=1e-4, symmetrize=False):
    event_shape = [2]
    num_components = nr_gaussians
 
    negloglik = lambda y, p_y: -p_y.log_prob(y)

    model = MDN(nr_gaussians, nr_neurons, activation_function, symmetrize=symmetrize)
    
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=negloglik)

    return model
 
def load_model(model_path, config_path=None):
   if config_path is None:
      config_path = model_path.replace('.h5', '_config.json')
   
   config = open_config(config_path)
   model = build_model(config.get("nr_gaussians"), config.get("activation_function"), config.get("nr_neurons"), symmetrize=config.get("symmetrize"))
   model(np.ones((1,3)))  # Initialize model
   model.load_weights(model_path)
   model.summary()
   return model
import tensorflow as tf
import tensorflow_probability as tfp

from utils import open_config

tf.keras.backend.set_floatx('float64')

tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers

def build_model(nr_gaussians=20, activation_function='relu', nr_neurons=8, learning_rate=1e-4):
    event_shape = [2]
    num_components = nr_gaussians
    params_size = tfpl.MixtureSameFamily.params_size(num_components,
                    component_params_size=tfpl.IndependentNormal.params_size(event_shape))

    negloglik = lambda y, p_y: -p_y.log_prob(y)

    model = tf.keras.models.Sequential([
       tf.keras.layers.Dense(nr_neurons, activation=activation_function, kernel_initializer='he_normal'),
       tf.keras.layers.Dense(params_size, activation=None, kernel_initializer='he_normal'),
       tfpl.MixtureSameFamily(num_components, tfpl.IndependentNormal(event_shape)),
    ])
    
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=negloglik)

    return model
 
def load_model(model_path, config_path, x):
   config = open_config(config_path)
   model = build_model(config.get("nr_gaussians"), config.get("activation_function"), config.get("nr_neurons"))
   model(x[:1])  # Initialize model
   model.load_weights(model_path)
   return model
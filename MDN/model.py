import tensorflow as tf
import tensorflow_probability as tfp

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
 
def load_model(path, x, nr_gaussians=20, activation_function='relu', nr_neurons=8):
   model = build_model(nr_gaussians, activation_function, nr_neurons)
   model(x[:1])  # Initialize model
   model.load_weights(path)
   return model
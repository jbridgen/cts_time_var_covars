from llik import ContinuousTimeModel
import transition_rates as tr

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

# Test data
event_times_table = np.array(
    [
        [0.0, 1.2, 3.0, 4.2, 1.5],  # SE
        [1.0, 2.5, 4.5, 5.2, 3.0],  # EI
        [3.7, 4.0, 5.75, 6.2, 4.5],  # IR
    ],
    dtype=np.float32,
)

event_table_units = [6, 2, 5, 8, 4]

event_list = ContinuousTimeModel.event_table_to_list(
    event_times_table, event_table_units
)
covariate_change_times = np.array([0.0, 1.2, 2.5, 5.2], np.float32)

incidence_matrix = np.array(
    [
        [-1, 0, 0, 0],
        [1, -1, 0, 0],
        [0, 1, -1, 0],
        [0, 0, 1, 0],
    ],
    dtype=np.float32,
)
num_individuals = 300
num_states, num_events = incidence_matrix.shape

initial_conditions = np.concatenate(
    [np.ones([1, num_individuals]), np.zeros([num_states - 1, num_individuals])],
    dtype=np.float32,
    axis=-2,
)  # Start all off as susceptible
initial_conditions[0, 3] = 0
initial_conditions[2, 3] = 1  # Set one individual as infectious


# create contact network tensors
tf.random.set_seed(23)
adj_mats = tf.random.uniform(
    shape=[covariate_change_times.shape[0], num_individuals, num_individuals],
    minval=0,
    maxval=3,
    dtype=tf.float32,
    seed=142,
)
spatial_conn = tf.random.uniform(
    shape=[covariate_change_times.shape[0], num_individuals, num_individuals],
    minval=0,
    maxval=3,
    dtype=tf.float32,
    seed=198,
)
covar_mats = dict(adj_mats=adj_mats, spatial_conn=spatial_conn)

# define model parameters
parms = {
    "alpha": 1 / 4,
    "gamma": 1 / 5,
    "beta1": 1.2,
    "beta2": 0.8,
}

# user defined transition rate functon
transition_rate_fn = tr.make_compute_transition_rates(covar_mats, parms)

# Instantiate the class
cts_model = ContinuousTimeModel(
    initial_conditions=initial_conditions,
    num_individuals=num_individuals,
    global_ids=tf.range(num_individuals),
    covariate_change_times=covariate_change_times,
    transition_rate_fn=transition_rate_fn,
    incidence_matrix=incidence_matrix,
)

# compute log prob
cts_model.log_prob(event_list)

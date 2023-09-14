import tensorflow as tf


def make_compute_transition_rates(covar_mats, parms):
    """Make a transition rate computation function

    :param covar_mats: a `dict` of covariate matrices
    :param parms: a `dict` of parameters
    :returns: a function
    """

    parms = {k: tf.convert_to_tensor(v) for k, v in parms.items()}

    def compute_transition_rates(args):
        state, covariate_pointers, event_id, unit = args

        adj_mat_t = tf.gather(
            covar_mats["adj_mats"],
            covariate_pointers,
            name="gather_adj_mats",
        )

        spatial_conn_t = tf.gather(
            covar_mats["spatial_conn"],
            covariate_pointers,
            name="gather_spatial_conn_t",
        )

        # Calculate transition rates
        se_rate = (
            parms["beta1"] * tf.linalg.matvec(adj_mat_t, tf.gather(state, 2))
            + parms["beta2"] * tf.linalg.matvec(spatial_conn_t, tf.gather(state, 2))
        ) * tf.gather(state, 0)

        ei_rate = tf.gather(state, 1) * parms["alpha"]
        ir_rate = tf.gather(state, 2) * parms["gamma"]
        cov_rate = tf.ones_like(ir_rate)
        transition_rates = tf.stack([se_rate, ei_rate, ir_rate, cov_rate])

        event_rate = tf.gather_nd(
            transition_rates, indices=[[event_id, tf.cast(unit, tf.int32)]]
        )
        total_event_rate = tf.reduce_sum(tf.stack([se_rate, ei_rate, ir_rate]))
        return total_event_rate, event_rate

    return compute_transition_rates

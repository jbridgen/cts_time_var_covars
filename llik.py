import numpy as np
import tensorflow as tf
from collections import namedtuple


def expand_event_list(event_times, event_unit, event_id, num_individuals, num_events):
    """Expands coordinates [event_unit, event_times, event_id] into a dense tensor of
    0s and 1s."""
    event_unit = tf.cast(event_unit, tf.int32)
    num_times = tf.shape(event_times)[0]

    # Create an array of [event_unit, event_times, event_id] coordinates
    i = tf.range(num_times)
    indices = tf.stack([i, event_id, event_unit], axis=-1)

    # Scatter a vector of 1s into an array of 0s
    dense_events = tf.scatter_nd(
        indices,
        updates=tf.ones(num_times),
        shape=[num_times, num_events, num_individuals],
    )

    return dense_events


def compute_state(initial_conditions, dense_events, incidence_matrix):
    """Computes a [num_times, num_states, num_individuals] state
    tensor.

    :param initial_conditions: a [num_states, num_individuals] tensor denoting
                               the initial conditions.
    :param dense_events: a [num_times, num_events, num_individuals] tensor
                         denoting events.
    :param incidence_matrix: a [num_states, num_events] tensor denoting the
                             state transition model.
    :returns: a [num_times, num_states, num_individuals] tensor denoting the
              state.
    """

    # First compute the increments we need to add to state
    increments = tf.einsum("trk,sr->tsk", dense_events, incidence_matrix)

    # Reconstructs the state by taking a cumsum along the time axis and
    # adding the initial conditions
    state = initial_conditions + tf.cumsum(increments, axis=-3, exclusive=True)

    return state


def compute_loglik(args):
    total_event_rate, event_rate, time_delta = args
    loglik_t = -total_event_rate * time_delta + tf.math.log(event_rate)

    return loglik_t


class ContinuousTimeModel:
    EventList = namedtuple("EventList", ["time", "unit", "event"])

    def __init__(
        self,
        initial_conditions,
        num_individuals,
        global_ids,
        covariate_change_times,
        transition_rate_fn,
        incidence_matrix,
    ):
        self._parameters = locals()

    @classmethod
    def event_table_to_list(cls, event_times_table, event_table_units):
        event_times_table = tf.convert_to_tensor(event_times_table)
        event_table_units = tf.convert_to_tensor(event_table_units)
        num_transitions, num_individuals = tf.shape(event_times_table)

        EventList = namedtuple("EventList", ["time", "unit", "event"])
        event_list = cls.EventList(
            time=tf.reshape(event_times_table, -1),
            unit=tf.tile(event_table_units, [num_transitions]),
            event=tf.repeat([0, 1, 2], [num_individuals]),
        )

        # Mask out any NaN event times so they don't appear in the event list
        is_not_nan = ~tf.math.is_nan(event_list.time)
        event_list = tf.nest.map_structure(
            lambda x: tf.boolean_mask(x, is_not_nan), event_list
        )

        # Sort by time
        sort_idx = tf.argsort(event_list.time)
        event_list = tf.nest.map_structure(lambda x: tf.gather(x, sort_idx), event_list)

        return event_list

    @property
    def transition_rate_fn(self):
        return self._parameters["transition_rate_fn"]

    @property
    def initial_conditions(self):
        return self._parameters["initial_conditions"]

    @property
    def num_individuals(self):
        return self._parameters["num_individuals"]

    @property
    def num_states(self):
        return self.incidence_matrix.shape[0]

    @property
    def num_events(self):
        return self.incidence_matrix.shape[1]

    @property
    def covariate_change_times(self):
        return self._parameters["covariate_change_times"]

    @property
    def global_ids(self):
        return self._parameters["global_ids"]

    @property
    def incidence_matrix(self):
        return self._parameters["incidence_matrix"]

    def _concatonate_event_lists(self, event_list):
        # Concatenate the fixed and changed event list with covariate times
        num_covariate_changes = tf.shape(self.covariate_change_times)[0]

        new_event_list = self.EventList(
            time=tf.concat(
                [
                    event_list.time,
                    self.covariate_change_times,
                ],
                axis=0,
            ),
            unit=tf.concat(
                [
                    event_list.unit,
                    tf.fill(
                        (num_covariate_changes,),
                        tf.constant(-1, event_list.unit.dtype),
                    ),
                ],
                axis=0,
            ),
            event=tf.concat(
                [
                    event_list.event,
                    tf.fill((num_covariate_changes,), self.num_events - 1),
                ],
                axis=0,
            ),
        )

        # Sort by time
        sorted_idx = tf.argsort(new_event_list.time, name="sort_event_list")
        event_list = tf.nest.map_structure(
            lambda x: tf.gather(x, sorted_idx), new_event_list
        )

        # Generate a corresponding sequence of pointers to indices in the
        # first dimension of the covariate structure
        covariate_pointers = tf.cumsum(
            tf.cast(event_list.unit == -1, event_list.unit.dtype)
        )
        # Reset -1s to 0s
        event_list = event_list._replace(
            unit=tf.clip_by_value(
                event_list.unit,
                clip_value_min=0,
                clip_value_max=event_list.unit.dtype.max,
            )
        )

        return event_list, covariate_pointers

    @tf.function(jit_compile=True)
    def _log_prob_chunk(
        self, event_times, event_units, event_id, covariate_pointers, initial_state
    ):
        with tf.name_scope("log_prob_chunk"):
            # Create dense event tensor
            dense_events = expand_event_list(
                event_times,
                event_units,
                event_id,
                self.num_individuals,
                self.num_events,
            )

            # Compute the state tensor
            state = compute_state(initial_state, dense_events, self.incidence_matrix)

            # Compute time between each event
            times = tf.concat([[event_times[0]], event_times], 0)
            time_delta = times[1:] - times[:-1]

            # Compute likelihood
            def llik_t_fn(args):
                with tf.name_scope("llik_t_fn"):
                    (
                        state,
                        covariate_pointers,
                        event_id,
                        event_pids,
                        time_delta,
                    ) = args
                    total_event_rate, event_rate = self.transition_rate_fn(
                        (state, covariate_pointers, event_id, event_pids)
                    )

                    return compute_loglik((total_event_rate, event_rate, time_delta))

            loglik_t = tf.vectorized_map(
                llik_t_fn,
                elems=(
                    state,
                    covariate_pointers,
                    event_id,
                    event_units,
                    time_delta,
                ),
            )
            return state[-1], tf.reduce_sum(loglik_t)

    def log_prob(self, event_list):
        """Return the log probability density of observing `event_list`

        :param event_list: an EventList object
        :returns: the log probability density of observing `event_list`
        """
        # 1.Concatonate event list with covariate times
        (event_list, covariate_pointers) = self._concatonate_event_lists(event_list)

        # Calculate chunks of likelihood
        initial_state = self.initial_conditions

        def chunk_fn(accum, elems):
            initial_state, log_prob = accum
            event_times, event_units, event_id, covariate_pointers = elems
            next_state, log_lik_chunk = self._log_prob_chunk(
                event_times,
                event_units,
                event_id,
                covariate_pointers,
                initial_state,
            )

            return next_state, log_lik_chunk + log_prob

        _, log_prob = tf.scan(
            chunk_fn,
            elems=[
                tf.expand_dims(x, 0)  # Use tf.split to chunk up, maybe.
                for x in (
                    event_list.time,
                    event_list.unit,
                    event_list.event,
                    covariate_pointers,
                )
            ],
            initializer=(initial_state, 0.0),
        )

        return (log_prob,)

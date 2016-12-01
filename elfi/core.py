import itertools
from functools import partial

import numpy as np

from elfi.utils import *
from elfi.storage import ElfiStore, LocalDataStore, MemoryStore
from elfi.graph import Node
from elfi import env
from elfi.inference_task import InferenceTask


# TODO: enforce this?
DEFAULT_DATATYPE = np.float32


def prepare_store(store):
    """Takes in user-originated specifier for 'store' and
    returns a corresponding ElfiStore derivative or raises
    a value error.

    Parameters
    ----------
    store : various
        None : means data is not stored.
        ElfiStore derivative : stores data according to specification.
        String identifiers :
        "cache" : Creates a MemoryStore()
        Sliceable object : is converted to LocalDataStore(obj)

        Examples: local numpy array, h5py instance.
        The size of the object must be at least (n_samples, )  data.shape
        The slicing must be consistent:
            obj[sl] = d must guarantee that obj[sl] == d
            For example, an empty list will not guarantee this, but a pre-allocated will.
        See also: LocalDataStore

    Returns
    -------
    `ElfiStore` instance or None is store is None
    """

    if store is None:
        return None
    if isinstance(store, ElfiStore):
        return store
    if type(store) == str:
        if store.lower() == "cache":
            return MemoryStore()
        raise ValueError("Unknown store identifier '{}'".format(store))
    return LocalDataStore(store)


class DelayedOutputCache:
    """Handles a continuous list of delayed outputs for a node.
    """
    def __init__(self, node_id, store=None):
        """

        Parameters
        ----------
        node_id : str
            id of the node (`node.id`)
        store : various
            See prepare_store
        """
        self._delayed_outputs = []
        self._stored_mask = []
        self._store = prepare_store(store)
        self._node_id = node_id

    def __len__(self):
        l = 0
        for o in self._delayed_outputs:
            l += slen(get_key_slice(o.key))
        return l

    def append(self, output):
        """Appends output to cache/store

        """
        if len(self) != get_key_slice(output.key).start:
            raise ValueError('Appending a non matching slice')

        self._delayed_outputs.append(output)
        self._stored_mask.append(False)
        if self._store:
            self._store.write(output, done_callback=self._set_stored)

    def reset(self, new_node_id):
        del self._delayed_outputs[:]
        del self._stored_mask[:]
        if self._store is not None:
            self._store.reset(self._node_id)
        self._node_id = new_node_id

    def __getitem__(self, sl):
        """
        Returns the delayed data in slice `sl`
        """
        sl = to_slice(sl)
        outputs = self._get_output_datalist(sl)

        # Return the data_slice
        if len(outputs) == 0:
            empty = np.zeros(shape=(0,0))
            output = delayed(empty)
        elif len(outputs) == 1:
            output = outputs[0]
        else:
            key = reset_key_slice(outputs[0].key, sl)
            output = delayed(np.vstack)(tuple(outputs), dask_key_name=key)
        return output

    def _get_output_datalist(self, sl):
        data_list = []
        for i, output in enumerate(self._delayed_outputs):
            output_sl = get_key_slice(output.key)
            intsect_sl = slice_intersect(output_sl, sl)
            if slen(intsect_sl) == 0:
                continue

            if self._stored_mask[i] == True:
                output_data = self._store.read_data(self._node_id, output_sl)
            else:
                output_data = get_named_item(output, 'data')

            if slen(intsect_sl) != slen(output_sl):
                # Take a subset of the data-slice
                intsect_key = reset_key_slice(output_data.key, intsect_sl)
                sub_sl = slice_intersect(intsect_sl, offset=output_sl.start)
                output_data = delayed(operator.getitem)(output_data, sub_sl, dask_key_name=intsect_key)
            data_list.append(output_data)
        return data_list

    def _set_stored(self, key, result):
        """Inform that the result is computed for the `key`.

        Allows self to start using the stored delayed object

        Parameters
        ----------
        key : key of the original output
        result : future or concrete result of the output (currently not used)
        """
        output = [i for i,o in enumerate(self._delayed_outputs) if o.key == key]
        if len(output) != 1:
            # TODO: this error doesn't actually currently propagate into the main thread
            # Also make a separate case for len > 1
            raise LookupError('Cannot find output with the given key')
        i = output[0]
        self._stored_mask[i] = True


def to_output_dict(input_dict, **kwargs):
    output_dict = input_dict.copy()
    for k, v in kwargs.items():
        output_dict[k] = v
    return output_dict


substreams = itertools.count()


def normalize_data(data, n=1):
    """Translates user-originated data into format compatible with the core.

    Parameters
    ----------
    data : any object
        User-originated data.
    n : int
        Number of times to replicate data (vectorization).

    Returns
    -------
    ret : np.ndarray

    If type(data) is not list, tuple or numpy.ndarray:
        ret.shape == (n, 1), ret[i][0] == data for all i
    If type(data) is list or tuple:
        data is converted to atleast 1D numpy array, after which
    If data.ndim == 1:
        If len(data) == n:
            ret.shape == (n, 1), ret[i][0] == data[i] for all i
        If len(data) != n:
            ret.shape == (n, len(data), ret[i] == data for all i
    If data.ndim > 1:
        If len(data) == n:
            ret == data
        If len(data) != n:
            ret.shape == (n, ) + data.shape, ret[i] == data for all i

    Examples
    --------
    Plain data
    >>> normalize_data(1, n=1)
    array([[1]])
    >>> normalize_data(1, n=2)
    array([[1],
           [1]])

    1D data
    >>> normalize_data([1], n=1)
    array([[1]])
    >>> normalize_data([1], n=2)
    array([[1],
           [1]])
    >>> normalize_data([1, 2], n=1)
    array([[1, 2]])
    >>> normalize_data([1, 2], n=2)
    array([[1],
           [2]])

    2D data
    >>> normalize_data([[1]], n=1)
    array([[1]])
    >>> normalize_data([[1]], n=2)
    array([[[1]],
    <BLANKLINE>
           [[1]]])
    >>> normalize_data([[1], [2]], n=1)
    array([[[1],
            [2]]])
    >>> normalize_data([[1], [2]], n=2)
    array([[1],
           [2]])
    """
    if isinstance(data, str):
        # numpy array initialization works unintuitively with strings
        data = np.array([[data]], dtype=object)
    else:
        data = np.atleast_1d(data)

    if data.ndim == 1:
        if data.shape[0] == n:
            data = data[:, None]
        else:
            data = data[None, :]
            if n > 1:
                data = np.vstack((data, ) * n)
    else:
        if data.shape[0] != n:
            data = data[None, :]
            if n > 1:
                data = np.vstack((data, ) * n)
    return data


def normalize_data_dict(dict, n):
    if dict is None:
        return None
    normalized = {}
    for k, v in dict.items():
        normalized[k] = normalize_data(v, n)
    return normalized


class Operation(Node):
    def __init__(self, name, operation, *parents, inference_task=None, store=None):
        """Operation node transforms data from parents to output
        that is given to the node's children.

        The operation takes `input_dict` from the parent nodes as input. The subclasses
        of `Operation` usually abstract `input_dict` away and instead define a more
        straightforward function signature tailored for the subclasses purpose.

        The `input_dict` will have a key "data", that contains a tuple where each parent
        in `parents` is replaced by the parent data.

        Parameters
        ----------
        name : string
            Name of the node
        operation : node operation function
            `operation(input_dict)` returns `output_dict`
            `input_dict` and `output_dict` must contain a key `"data"`
        *parents : tuple or list
            Parents of the operation node
        store : `OutputStore` instance
        """
        inference_task = inference_task or env.inference_task()
        super(Operation, self).__init__(name, *parents, graph=inference_task)
        self.operation = operation

        self._generate_index = 0
        # Keeps track of the resets
        self._num_resets = 0
        self._delayed_outputs = DelayedOutputCache(self.id, store)

    def acquire(self, n, starting=0, batch_size=None, **kwargs):
        """Acquires values from the start or from starting index.
        Generates new ones if needed and updates the _generate_index.

        Parameters
        ----------
        n : int
            number of samples
        starting : int
        batch_size : int

        Returns
        -------
        n samples in numpy array
        """
        sl = slice(starting, starting+n)
        if self._generate_index < sl.stop:
            self.generate(sl.stop - self._generate_index, batch_size=batch_size, **kwargs)
        return self.get_slice(sl)

    # TODO: better documentation for `with_values`
    def generate(self, n, batch_size=None, with_values=None):
        """Generate n new values from the node. If all of the n values are going to be the same value,
        it is allowed to return just one value (see e.g. Constant).

        Parameters
        ----------
        n : int
            number of samples
        batch_size : int
        with_values : dict(node_name: np.array)

        Returns
        -------
        n new values or a 1 value if all n values are the same
        """
        a = self._generate_index
        b = a + n
        batch_size = batch_size or n
        with_values = normalize_data_dict(with_values, n)

        # TODO: with_values cannot be used with already generated values
        # Ensure store is filled up to `b`
        while len(self._delayed_outputs) < b:
            l = len(self._delayed_outputs)
            n_batch = min(b-l, batch_size)
            batch_sl = slice(l, l+n_batch)
            batch_values = None
            if with_values is not None:
                batch_values = {k: v[(l-a):(l-a)+n_batch] for k,v in with_values.items()}
            self.get_slice(batch_sl, with_values=batch_values)

        self._generate_index = b
        return self[slice(a, b)]

    def __getitem__(self, sl):
        sl = to_slice(sl)
        return self._delayed_outputs[sl]

    def __str__(self):
        return "{}".format(self.__class__.__name__)

    # TODO: better documentation for `with_values`
    def get_slice(self, sl, with_values=None):
        """
        This function is ensured to give a slice anywhere (already generated or not)
        Does not update _generate_index

        Parameters
        ----------
        sl : slice
            continuous slice
        with_values : dict(node_name: np.array)

        Returns
        -------
        numpy.array of samples in the slice `sl`

        """
        # TODO: prevent using with_values with already generated values
        # Check if we need to generate new
        if len(self._delayed_outputs) < sl.stop:
            with_values = normalize_data_dict(with_values,
                                              sl.stop - len(self._delayed_outputs))
            new_sl = slice(len(self._delayed_outputs), sl.stop)
            new_input = self._create_input_dict(new_sl, with_values=with_values)
            new_output = self._create_delayed_output(new_sl, new_input, with_values)
            self._delayed_outputs.append(new_output)
        return self[sl]

    @property
    def id(self):
        return make_key_id(self.inference_task.name, self.name, self.version)

    @property
    def inference_task(self):
        return self.graph

    @property
    def version(self):
        """Version of the node (currently number of resets)"""
        return self._num_resets

    def reset(self, propagate=True):
        """Resets the data of the node

        Resets the node to a state as if no data was generated from it.
        If propagate is True (default) also resets its descendants

        Parameters
        ----------
        propagate : bool

        """
        if propagate:
            for c in self.children:
                c.reset()
        self._generate_index = 0
        self._num_resets += 1
        self._delayed_outputs.reset(self.id)

    def _create_input_dict(self, sl, with_values=None):
        n = sl.stop - sl.start
        input_data = tuple([p.get_slice(sl, with_values) for p in self.parents])
        return {
            "data": input_data,
            "n": n,
            "index": sl.start,
        }

    def _create_delayed_output(self, sl, input_dict, with_values=None):
        """

        Parameters
        ----------
        sl : slice
        input_dict : dict
        with_values : dict {'node_name': np.array}

        Returns
        -------
        out : dask.delayed object

        """
        with_values = with_values or {}
        dask_key = make_key(self.id, sl)
        if self.name in with_values:
            # Set the data to with_values
            output = to_output_dict(input_dict, data=with_values[self.name])
            return delayed(output, name=dask_key)
        else:
            dinput = delayed(input_dict, pure=True)
            return delayed(self.operation)(dinput,
                                           dask_key_name=dask_key)

    def _convert_to_node(self, obj, name):
        return Constant(name, obj)


"""
Operation mixins add additional functionality to the Operation class.
They do not define the actual operation but may add add keyword arguments
for the constructor. They may also add keys to `input_dict` and `output_dict`.
"""


def get_substream_state(master_seed, substream_index):
    """Returns PRNG internal state for the sub stream

    Parameters
    ----------
    master_seed : uint32
    substream_index : uint

    Returns
    -------
    out : tuple
    Random state for the sub stream as defined by numpy

    See Also
    --------
    'numpy.random.RandomState.get_state' for the representation of MT19937 state
    """
    # Fixme: In the future, allow MRG32K3a from https://pypi.python.org/pypi/randomstate
    seeds = np.random.RandomState(master_seed)\
        .randint(np.iinfo(np.uint32).max, size=substream_index+1)
    return np.random.RandomState(seeds[substream_index]).get_state()


class RandomStateMixin(Operation):
    """Makes Operation node stochastic.
    """
    def __init__(self, *args, **kwargs):
        super(RandomStateMixin, self).__init__(*args, **kwargs)

    def _create_input_dict(self, sl, **kwargs):
        dct = super(RandomStateMixin, self)._create_input_dict(sl, **kwargs)
        dct["random_state"] = self._get_random_state()
        return dct

    def _get_random_state(self):
        it = self.inference_task
        return delayed(get_substream_state, pure=True)(it.seed, it.new_substream_index())


class ObservedMixin(Operation):
    """Adds observed data to the class.
    """

    def __init__(self, *args, observed=None, **kwargs):
        super(ObservedMixin, self).__init__(*args, **kwargs)
        if observed is None:
            self.observed = self._inherit_observed()
        else:
            self.observed = normalize_data(observed, 1)

    def _inherit_observed(self):
        if len(self.parents) < 1:
            raise ValueError("There are no parents to inherit from")
        for parent in self.parents:
            if not hasattr(parent, "observed"):
                raise ValueError("Parent {} has no observed value to inherit".format(parent))
        observed = tuple([p.observed for p in self.parents])
        observed = self.operation({"data": observed, "n": 1})["data"]
        return observed


"""
Operation nodes
"""


class Constant(ObservedMixin, Operation):
    """
    Constant. Holds a constant value and returns only that when asked to generate data.
    Observed value is set also to the same value.
    """
    def __init__(self, name, value):
        """

        Parameters
        ----------
        value : constant value returned from generate
        """
        if type(value) in (tuple, list, np.ndarray):
            self.value = normalize_data(value, len(value))
        else:
            self.value = normalize_data(value, 1)
        v = self.value.copy()
        super(Constant, self).__init__(name, lambda input_dict: {"data": v}, observed=v)


def vectorize_simulator(simulator, *input_data, n_sim=1, prng=None):
    """Used to vectorize a sequential simulation operation
    """
    data = None
    for i in range(n_sim):
        inputs = [v[i] for v in input_data]
        d = simulator(*inputs, prng=prng)
        if not isinstance(d, np.ndarray):
            raise ValueError("Simulation operation output type incorrect." +
                "Expected type np.ndarray, received type {}".format(type(d)))
        if data is None:
            data = np.zeros((n_sim,) + d.shape)
        data[i] = d
    return data


# For python simulators using numpy random variables
def simulator_operation(simulator, vectorized, input_dict):
    """Calls the simulator to produce output

    Vectorized simulators
    ---------------------
    Calls the simulator(*vectorized_args, n_sim, prng) to create output.
    Each vectorized argument to simulator is a numpy array with shape[0] == 'n_sim'.
    Simulator should return a numpy array with shape[0] == 'n_sim'.

    Sequential simulators
    ---------------------
    Calls the simulator(*args, prng) 'n_sim' times to create output.
    Each argument to simulator is of the dtype of the original array[i].
    Simulator should return a numpy array.

    Parameters
    ----------
    simulator: function
    vectorized: bool
    input_dict: dict
        "n": number of parallel simulations
        "data": list of args as numpy arrays
    """
    # set the random state
    prng = np.random.RandomState(0)
    prng.set_state(input_dict["random_state"])
    n_sim = input_dict["n"]
    data = simulator(*input_dict["data"], n_sim=n_sim, prng=prng)
    if not isinstance(data, np.ndarray):
        raise ValueError("Simulation operation output type incorrect." +
                "Expected type np.ndarray, received type {}".format(type(data)))
    if data.shape[0] != n_sim or len(data.shape) < 2:
        raise ValueError("Simulation operation output format incorrect." +
                " Expected shape == ({}, ...).".format(n_sim) +
                " Received shape == {}.".format(data.shape))
    return to_output_dict(input_dict, data=data, random_state=prng.get_state())


class Simulator(ObservedMixin, RandomStateMixin, Operation):
    """Simulator node

    Parameters
    ----------
    name: string
    simulator: function
    vectorized: bool
        whether the simulator function is vectorized or not
        see definition of simulator_operation for more information
    """
    def __init__(self, name, simulator, *args, vectorized=True, **kwargs):
        if vectorized is False:
            simulator = partial(vectorize_simulator, simulator)
        operation = partial(simulator_operation, simulator, vectorized)
        super(Simulator, self).__init__(name, operation, *args, **kwargs)


def vectorize_summary(summary, *input_data):
    """Used to vectorize a sequential summary operation
    """
    data = None
    # TODO: should summary operations also get n_sim as parameter?
    n_sim = input_data[0].shape[0]
    for i in range(n_sim):
        inputs = [v[i] for v in input_data]
        d = summary(*inputs)
        if not isinstance(d, np.ndarray):
            raise ValueError("Summary operation output type incorrect." +
                "Expected type np.ndarray, received type {}".format(type(d)))
        if data is None:
            data = np.zeros((n_sim,) + d.shape)
        data[i] = d
    return data


def summary_operation(operation, input):
    data = operation(*input["data"])
    vec_len = input["n"]
    if not isinstance(data, np.ndarray):
        raise ValueError("Summary operation output type incorrect." +
                "Expected type np.ndarray, received type {}".format(type(data)))
    if data.shape[0] != vec_len or len(data.shape) < 2:
        raise ValueError("Summary operation output format incorrect." +
                " Expected shape == ({}, ...).".format(vec_len) +
                " Received shape == {}.".format(data.shape))
    return to_output_dict(input, data=data)


class Summary(ObservedMixin, Operation):
    def __init__(self, name, summary, *args, vectorized=True, **kwargs):
        if vectorized is False:
            summary = partial(vectorize_summary, summary)
        operation = partial(summary_operation, summary)
        super(Summary, self).__init__(name, operation, *args, **kwargs)


def vectorize_discrepancy(discrepancy, x, y):
    """Used to vectorize a sequential discrepancy operation
    """
    # TODO: should discrepancy operations also get n_sim as parameter?
    n_sim = x[0].shape[0]
    data = np.zeros((n_sim, 1))
    for i in range(n_sim):
        xi = tuple([v[i] for v in x])
        yi = tuple([v[0] for v in y])
        d = discrepancy(x, y)
        if not isinstance(d, np.ndarray):
            raise ValueError("Discrepancy operation output type incorrect." +
                "Expected type np.ndarray, received type {}".format(type(d)))
        if d.shape != (1,):
            raise ValueError("Discrepancy operation output format incorrect." +
                " Expected shape == (1,)." +
                " Received shape == {}.".format(data.shape))
        data[i] = d
    return data

def discrepancy_operation(operation, input):
    data = operation(input["data"], input["observed"])
    vec_len = input["n"]
    if not isinstance(data, np.ndarray):
        raise ValueError("Discrepancy operation output type incorrect." +
                "Expected type np.ndarray, received type {}".format(type(data)))
    if data.shape != (vec_len, 1):
        raise ValueError("Discrepancy operation output format incorrect." +
                " Expected shape == ({}, 1).".format(vec_len) +
                " Received shape == {}.".format(data.shape))
    return to_output_dict(input, data=data)


class Discrepancy(Operation):
    """The operation input has a tuple of data and tuple of observed
    """
    def __init__(self, name, discrepancy, *args, vectorized=True, **kwargs):
        if vectorized is False:
            discrepancy = partial(vectorize_discrepancy, discrepancy)
        operation = partial(discrepancy_operation, discrepancy)
        super(Discrepancy, self).__init__(name, operation, *args, **kwargs)

    def _create_input_dict(self, sl, **kwargs):
        dct = super(Discrepancy, self)._create_input_dict(sl, **kwargs)
        dct["observed"] = observed = tuple([p.observed for p in self.parents])
        return dct


def threshold_operation(threshold, input):
    data = input['data'][0] < threshold
    return to_output_dict(input, data=data)


class Threshold(Operation):
    def __init__(self, name, threshold, *args, **kwargs):
        operation = partial(threshold_operation, threshold)
        super(Threshold, self).__init__(name, operation, *args, **kwargs)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

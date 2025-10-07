# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) joblib contributors.
import cloudpickle


class PicklableWrapper(object):
    """
    Wrap an object to make it more picklable, note that it uses
    heavy weight serialization libraries that are slower than pickle.
    It's best to use it only on closures (which are usually not picklable).

    This is a simplified version of joblib
    """

    def __init__(self, obj):
        while isinstance(obj, PicklableWrapper):
            # Wrapping an object twice is no-op
            obj = obj._obj
        self._obj = obj

    def __reduce__(self):
        s = cloudpickle.dumps(self._obj)
        return cloudpickle.loads, (s,)

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __getattr__(self, attr):
        # Ensure that the wrapped object can be used seamlessly as the previous object.
        if attr not in ["_obj"]:
            return getattr(self._obj, attr)
        return getattr(self, attr)

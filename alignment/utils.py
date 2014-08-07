import collections

def iflatten(iterable):
    """
    Returns a flattened version of an arbitrarily deeply nested iterable.
    :param iterable: some fixed order iterable (list, tuple)
    :return: generator
    """
    for elt in iterable:
        if isinstance(elt, collections.Iterable) and not isinstance(elt, str):
            for sub in flatten(elt):
                yield sub
        else:
            yield elt

def flatten(iterable):
    return list(iflatten(iterable))

def merge(*sequences):
    return list(map(flatten, zip(*sequences)))
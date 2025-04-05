def compute_max(iterable, evaluation_function):
    best = None
    best_value = None
    for e in iterable:
        value = evaluation_function(e)
        if best_value is None or value > best_value:
            best_value = value
            best = e
    return best, best_value

def strategy_factory(strategy_type):
    if strategy_type == "incremental":
        return incremental
    elif strategy_type == "reverse-incremental":
        return revIncremental
    elif strategy_type == "exponential":
        return exponential
    elif strategy_type == "dense":
        return dense
    elif strategy_type == "void":
        return void
    else:
        raise NotImplementedError


def void(step, max_steps):
    return False, None


def dense(step, max_steps):
    return True, 1


def incremental(step, max_steps, min_interval=1, max_interval=100, n_increments=10):
    step_size = max_steps // n_increments
    increment_size = int((max_interval - min_interval + 1) // n_increments * (step // step_size))
    
    if step // step_size == 0:
        return True, 1
    else:
        if step % increment_size == 0:
            return True, increment_size
        else:
            return False, increment_size


def revIncremental(step, max_steps, min_interval=1, max_interval=100, n_increments=10):
    step_size = max_steps // n_increments
    curr_increment = (n_increments - (step // step_size) - 1)
    increment_size = int((max_interval - min_interval + 1) // n_increments * curr_increment)
    
    if curr_increment == 0:
        return True, 1
    else:
        if step % increment_size == 0:
            return True, increment_size
        else:
            return False, increment_size


def exponential():
    pass

def listify(value):
    if isinstance(value, list):
        return value
    elif isinstance(value, tuple):
        return [*value]
    else:
        return [value]

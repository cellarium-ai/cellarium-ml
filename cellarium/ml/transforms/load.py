def to_device(x):
    """Transfer a module from a checkpoint to the meta device"""
    x = x.to("meta")
    return x
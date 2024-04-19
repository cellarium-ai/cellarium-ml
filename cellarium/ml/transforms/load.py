def to_device(x):
    x = x.to("meta")
    x.training = False
    #TODO: Trainer class? assign subcommands?
    return x
from .horse import HorseReID

__imgreid_factory = {
    "horse": HorseReID,
}

def get_names():
    return list(__imgreid_factory.keys())

def init_imgreid_dataset(name, **kwargs):
    if name not in __imgreid_factory:
        raise KeyError(f"Unknown dataset: {name}. Available: {list(__imgreid_factory.keys())}")
    return __imgreid_factory[name](**kwargs)

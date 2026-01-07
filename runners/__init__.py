from .STFRunner import STFRunner



def runner_select(name):
    name = name.upper()

    if name in ("STF", "BASIC", "DEFAULT"):
        return STFRunner

    else:
        raise NotImplementedError

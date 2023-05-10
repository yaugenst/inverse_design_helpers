import inspect

from autograd import grad, value_and_grad


def grad_dict(fun):
    """Calculates gradient w.r.t. all function arguments and returns them as a dict.
    This is an updated version of `multigrad_dict` from autograd.
    See https://github.com/HIPS/autograd/blob/cecca22f59f4d62dd0df6b54cc57608078ca6df0/autograd/differential_operators.py#L152 for reference.
    """
    sig = inspect.signature(fun)

    def select(preds, lst):
        def idx(item):
            return next((i for i, pred in enumerate(preds) if pred(item)), len(preds))

        results = [[] for _ in preds] + [[]]
        for item in lst:
            results[idx(item)].append(item)
        return results

    def is_var_pos(name):
        return sig.parameters[name].kind == sig.parameters[name].VAR_POSITIONAL

    def is_var_kwd(name):
        return sig.parameters[name].kind == sig.parameters[name].VAR_KEYWORD

    var_pos, var_kwd, argnames = select([is_var_pos, is_var_kwd], sig.parameters)

    def todict(dct):
        return {key: dct[key] for key in dct}

    def apply_defaults(arguments):
        defaults = {
            name: param.default
            for name, param in sig.parameters.items()
            if param.default is not param.empty
        }
        return {
            name: arguments[name] if name in arguments else defaults[name]
            for name in sig.parameters
        }

    def gradfun(*args, **kwargs):
        bindings = sig.bind(*args, **kwargs)

        def args(dct):
            return tuple(dct[var_pos[0]]) if var_pos else ()

        def kwargs(dct):
            return todict(dct[var_kwd[0]]) if var_kwd else {}

        def others(dct):
            return tuple(
                dct[argname] for argname in argnames if argname not in var_kwd + var_pos
            )

        def newfun(dct):
            return fun(*(others(dct) + args(dct)), **kwargs(dct))

        argdict = apply_defaults(bindings.arguments)
        return grad(newfun)(argdict)

    return gradfun


def value_and_grad_dict(fun):
    """Calculates value and gradient w.r.t. all function arguments.
    This is the `value_and_grad` version of autograd's `multigrad_dict`.
    """
    sig = inspect.signature(fun)

    def select(preds, lst):
        def idx(item):
            return next((i for i, pred in enumerate(preds) if pred(item)), len(preds))

        results = [[] for _ in preds] + [[]]
        for item in lst:
            results[idx(item)].append(item)
        return results

    def is_var_pos(name):
        return sig.parameters[name].kind == sig.parameters[name].VAR_POSITIONAL

    def is_var_kwd(name):
        return sig.parameters[name].kind == sig.parameters[name].VAR_KEYWORD

    var_pos, var_kwd, argnames = select([is_var_pos, is_var_kwd], sig.parameters)

    def todict(dct):
        return {key: dct[key] for key in dct}

    def apply_defaults(arguments):
        defaults = {
            name: param.default
            for name, param in sig.parameters.items()
            if param.default is not param.empty
        }
        return {
            name: arguments[name] if name in arguments else defaults[name]
            for name in sig.parameters
        }

    def gradfun(*args, **kwargs):
        bindings = sig.bind(*args, **kwargs)

        def args(dct):
            return tuple(dct[var_pos[0]]) if var_pos else ()

        def kwargs(dct):
            return todict(dct[var_kwd[0]]) if var_kwd else {}

        def others(dct):
            return tuple(
                dct[argname] for argname in argnames if argname not in var_kwd + var_pos
            )

        def newfun(dct):
            return fun(*(others(dct) + args(dct)), **kwargs(dct))

        argdict = apply_defaults(bindings.arguments)
        return value_and_grad(newfun)(argdict)

    return gradfun

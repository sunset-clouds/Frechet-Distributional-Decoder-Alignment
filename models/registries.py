def family_registry(generator_name="", tokenizer_name=""):
    """The models/<family>/registry module whose prefix matches either name, or None."""
    for family in FAMILIES:
        if generator_name.startswith(family) or tokenizer_name.startswith(family):
            return __import__(f"models.{family}.registry", fromlist=["registry"])
    return None


FAMILIES = ("llamagen", "gigatok", "titok", "var", "imf")


def _registries():
    for family in FAMILIES:
        yield __import__(f"models.{family}.registry", fromlist=["registry"])


def model_names(kind):
    """Sorted list of every registered name for kind 'tokenizer' or 'generator'."""
    return sorted(name for registry in _registries() for name in registry.MODELS[kind])


def resolve_model_paths(args):
    registry = family_registry(args.generator_name, args.tokenizer_name)
    if registry is None:
        return args

    if not args.tokenizer_ckpt_path:
        args.tokenizer_ckpt_path = registry.ckpt_path("tokenizer", args.tokenizer_name)
    if not args.generator_ckpt_path:
        args.generator_ckpt_path = registry.ckpt_path(
            "generator", args.generator_name, getattr(args, "resolution", None))
    return args

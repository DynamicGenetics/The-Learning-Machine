from .vgg import VGGMachine
from .unet import UNetMachine
from .learning_machine import LearningMachine

VGG_MODEL = "vgg"
UNET_MODEL = "unet"
MODELS_PROXY = dict()


def setup_learning_machine(model_key: str, pretrained: bool):
    """
    Utility method to initialise a new Learning Machine Model
    """
    if model_key == VGG_MODEL:
        return VGGMachine(pretrained=pretrained)
    if model_key == UNET_MODEL:
        return UNetMachine(pretrained=pretrained)
    raise ValueError(f"Model Key {model_key} is not currently supported")


def get_model(
    key: str, force_init: bool = False, pretrained: bool = False
) -> LearningMachine:
    """
    Instantiate a Learning Machine Model given a key for Models Proxy.

    Parameters
    ----------
    key : str
        A model proxy key
    force_init: bool (default False)
        If True, the model will be re-initialised before being returned.
    pretrained: bool (default False)
        This flag determines whether the re-initialised model should be returned
        pre-trained or not.
        NOTE: This parameter will only be used if `force_init` is True.

    Returns
    -------
    LearningMachine
        Instance of Machine subclass implementing the learning machine.
        This is basically a torch.nn.Module subclass, with unified API for an
        easier integration with ReSTful endpoints.

    Raises
    ------
    ValueError
        Raised if input key is not valid. No default fall-back model implemented.
    """
    net = MODELS_PROXY.get(key, None)
    if net is None or force_init:
        net = setup_learning_machine(key, pretrained=pretrained)
        MODELS_PROXY[key] = net
    print(
        f"[LOG]: Loading Model: {net.name} (Pretrained: {'yes' if net.is_pretrained else 'no'})"
    )
    return net

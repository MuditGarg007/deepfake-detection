from __future__ import annotations


DF40_HELD_IN = (
    "faceswap", "blendface", "facedancer", "fsgan", "mobileswap",
    "fomm", "facevid2vid", "lia", "tpsm", "wav2lip",
    "StyleGAN2", "ddim", "VQGAN", "DiT",
)

DF40_VAL_UNSEEN = ("mobileswap", "tpsm", "VQGAN")

DF40_HELD_OUT = (
    "inswap", "simswap", "uniface", "e4s", "deepfacelab",
    "sadtalker", "heygen_new", "heygen", "hyperreenact", "mcnet",
    "SiT", "StyleGAN3", "MidJourney", "CollabDiff",
    "starganv2", "styleclip", "whichfaceisreal", "stargan",
)


FFPP_HELD_IN = ("Deepfakes", "Face2Face")
FFPP_VAL_UNSEEN = ("FaceSwap",)
FFPP_HELD_OUT = ("NeuralTextures", "FaceShifter", "DeepFakeDetection")


TRAIN_REAL_DATASETS = ("ffpp", "cdf")
HELD_OUT_DATASETS = ("dfdcp", "uadfv")

HELD_IN = frozenset(DF40_HELD_IN) | frozenset(FFPP_HELD_IN)
VAL_UNSEEN = frozenset(DF40_VAL_UNSEEN) | frozenset(FFPP_VAL_UNSEEN)
HELD_OUT = frozenset(DF40_HELD_OUT) | frozenset(FFPP_HELD_OUT)

HELD_IN = HELD_IN - VAL_UNSEEN


def group_for(method: str, dataset: str) -> str:
    if method == "real":
        return "heldout" if dataset in HELD_OUT_DATASETS else "heldin"
    if dataset in HELD_OUT_DATASETS:
        return "heldout"
    if method in VAL_UNSEEN:
        return "valunseen"
    if method in HELD_IN:
        return "heldin"
    return "heldout"


FAMILY: dict[str, str] = {
    "Deepfakes": "swap_graphics",
    "FaceSwap": "swap_graphics",
    "faceswap": "swap_graphics",
    "deepfacelab": "swap_graphics",
    "DeepFakeDetection": "swap_graphics",
    "uadfv_fake": "swap_graphics",
    "blendface": "swap_learned",
    "facedancer": "swap_learned",
    "fsgan": "swap_learned",
    "inswap": "swap_learned",
    "simswap": "swap_learned",
    "uniface": "swap_learned",
    "e4s": "swap_learned",
    "mobileswap": "swap_learned",
    "FaceShifter": "swap_learned",
    "cdf_synthesis": "swap_learned",
    "dfdcp_fake": "swap_learned",
    "Face2Face": "reenactment",
    "NeuralTextures": "reenactment",
    "fomm": "reenactment",
    "facevid2vid": "reenactment",
    "lia": "reenactment",
    "tpsm": "reenactment",
    "wav2lip": "reenactment",
    "sadtalker": "reenactment",
    "heygen_new": "reenactment",
    "heygen": "reenactment",
    "hyperreenact": "reenactment",
    "mcnet": "reenactment",
    "StyleGAN2": "synthesis",
    "StyleGAN3": "synthesis",
    "ddim": "synthesis",
    "DiT": "synthesis",
    "SiT": "synthesis",
    "VQGAN": "synthesis",
    "MidJourney": "synthesis",
    "CollabDiff": "synthesis",
    "stargan": "synthesis",
    "starganv2": "synthesis",
    "styleclip": "synthesis",
    "whichfaceisreal": "synthesis",
}


def family_for(method: str) -> str:
    if method == "real":
        return "real"
    return FAMILY.get(method, "unknown")

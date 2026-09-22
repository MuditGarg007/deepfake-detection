"""Which manipulation methods the detector is allowed to see.

The whole point of v2 is generalization, and a detector evaluated on methods it
trained on tells you nothing about that. So methods are partitioned once, here,
and every script reads the partition from this module:

``HELD_IN``
    Trainable. A deliberately wide spread — classic graphics face-swaps, modern
    encoder swaps, reenactment/talking-head, GAN synthesis and diffusion
    synthesis — because the single biggest driver of cross-method AUC is how
    many *families* of artifact the model has seen, not how many frames.

``VAL_UNSEEN``
    Excluded from training and used only for model selection / early stopping.
    Selecting on in-domain val AUC picks the most overfitted checkpoint; these
    methods make the selection signal itself a generalization signal.

``HELD_OUT``
    Never touched until the final report. Modern face swaps (InSwapper,
    SimSwap, UniFace, E4S, DeepFaceLab), modern talking heads (SadTalker,
    HeyGen, HyperReenact, MCNet), modern generative synthesis (SiT, StyleGAN3,
    MidJourney, CollabDiff, StyleCLIP, StarGAN-v2) and whole unseen corpora
    (Celeb-DF-v2 fakes, DFDCP, UADFV).

Note the asymmetry: ``inswap`` and ``simswap`` are the two face swaps most
likely to be what someone actually runs today, and they are held out precisely
because they are the interesting question.
"""

from __future__ import annotations

# --- DF40 manipulations -----------------------------------------------------

DF40_HELD_IN = (
    # face swap (graphics + learned)
    "faceswap", "blendface", "facedancer", "fsgan", "mobileswap",
    # face reenactment / talking head
    "fomm", "facevid2vid", "lia", "tpsm", "wav2lip",
    # entire face synthesis (GAN + diffusion)
    "StyleGAN2", "ddim", "VQGAN", "DiT",
)

DF40_VAL_UNSEEN = ("mobileswap", "tpsm", "VQGAN")

DF40_HELD_OUT = (
    "inswap", "simswap", "uniface", "e4s", "deepfacelab",
    "sadtalker", "heygen_new", "heygen", "hyperreenact", "mcnet",
    "SiT", "StyleGAN3", "MidJourney", "CollabDiff",
    "starganv2", "styleclip", "whichfaceisreal", "stargan",
)

# --- FaceForensics++ manipulations -----------------------------------------
# Deepfakes is what v1 trained on; keeping it held-in makes the v1-vs-v2
# comparison fair. NeuralTextures is held out as the hardest FF++ method.

FFPP_HELD_IN = ("Deepfakes", "Face2Face")
# FaceSwap is a graphics-pipeline swap, unlike either held-in FF++ method, so
# it makes a usable unseen-method signal on its own — which also means model
# selection works before the DF40 corpus has finished downloading.
FFPP_VAL_UNSEEN = ("FaceSwap",)
FFPP_HELD_OUT = ("NeuralTextures", "FaceShifter", "DeepFakeDetection")

# --- Whole corpora ----------------------------------------------------------
# Real frames from these corpora are trainable (real faces are not a forgery
# method); their *fakes* are held out to give a clean cross-dataset number.

TRAIN_REAL_DATASETS = ("ffpp", "cdf")
HELD_OUT_DATASETS = ("dfdcp", "uadfv")

HELD_IN = frozenset(DF40_HELD_IN) | frozenset(FFPP_HELD_IN)
VAL_UNSEEN = frozenset(DF40_VAL_UNSEEN) | frozenset(FFPP_VAL_UNSEEN)
HELD_OUT = frozenset(DF40_HELD_OUT) | frozenset(FFPP_HELD_OUT)

# VAL_UNSEEN methods must not also be trainable.
HELD_IN = HELD_IN - VAL_UNSEEN


def group_for(method: str, dataset: str) -> str:
    """Return ``heldin`` / ``valunseen`` / ``heldout`` for a manifest row."""
    if method == "real":
        return "heldout" if dataset in HELD_OUT_DATASETS else "heldin"
    if dataset in HELD_OUT_DATASETS:
        return "heldout"
    if method in VAL_UNSEEN:
        return "valunseen"
    if method in HELD_IN:
        return "heldin"
    return "heldout"


# --- Generation families -----------------------------------------------------
# A pooled average over 31 manipulations hides a *class* of generation the
# detector cannot see at all — which is exactly what happened with SadTalker
# (AUC 0.4646, i.e. the model voting "real" on it). Grouping methods by how the
# pixels were produced turns "one weak method" into a testable claim about a
# family, and says whether the training corpus is missing a family rather than
# a method.
#
#   swap_graphics   classical pipeline: detect, warp, blend, colour-correct.
#                   Leaves a blending boundary around a face-shaped region.
#   swap_learned    encoder/decoder or identity-embedding swaps. Also blended,
#                   but the interior is generated.
#   reenactment     the identity is kept and the motion is driven: expression
#                   transfer, lip sync, audio-driven talking heads. Some
#                   re-render the whole head from a still, leaving no seam.
#   synthesis       the whole image is generated — GAN or diffusion. No source
#                   face, so no blending cue of any kind.

FAMILY: dict[str, str] = {
    # classical graphics swaps
    "Deepfakes": "swap_graphics",
    "FaceSwap": "swap_graphics",
    "faceswap": "swap_graphics",
    "deepfacelab": "swap_graphics",
    "DeepFakeDetection": "swap_graphics",
    "uadfv_fake": "swap_graphics",
    # learned swaps
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
    # reenactment / talking head
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
    # entire-face synthesis
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
    """Generation family for a manifest method, ``unknown`` if unclassified."""
    if method == "real":
        return "real"
    return FAMILY.get(method, "unknown")

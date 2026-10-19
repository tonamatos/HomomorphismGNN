"""Spasm, homomorphism-basis coefficients and (multi-)rooted hom counts."""
from .patterns import ROOTS, cycle, describe, make_pattern, path, root_nodes, star
from .spasm import SpasmTerm, rooted_automorphisms, spasm
from .hom import hom_bruteforce, hom_tensor, sub_tensor
from .hom_sparse import hom_sparse

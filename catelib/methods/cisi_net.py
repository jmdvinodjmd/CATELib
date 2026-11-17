"""CISI-Net: task embeddings and pairwise Wasserstein representation balancing.

Paper: https://causal-machine-learning.github.io/kdd2025-workshop/papers/8.pdf
Comparison specification: https://arxiv.org/html/2608.22024v1#S0.SS3
Official code: not verified; independent implementation based on the papers.

Uses the three-network architecture shared with CIHSI-Net, replacing barycentric
FGW with the mean pairwise W1 discrepancy across present binary patterns and
ordinary factual squared error. Pairwise mean normalisation is an explicit
implementation convention; penalty strength must be tuned accordingly. Missing
batch patterns are omitted, never fabricated. Binary-outcome and multi-output
support are extensions; independent outcome copies retain scalar prediction.
Solver gradient and validation conventions are documented in cihsi_net.py.
"""
from catelib.methods.cihsi_net import CIHSINet


class CISINet(CIHSINet):
    def _build_network(self):
        network = super()._build_network()
        network.mode = 'pairwise'
        return network

"""surfaces/__init__.py — Hybrid surface API: inject, retract, compose, provenance.
"""
from hybrid.surfaces.registry import ComponentRegistry
from hybrid.surfaces.inject import inject_logit_bias, inject_concept_pack
from hybrid.surfaces.retract import retract
from hybrid.surfaces.compose import compose

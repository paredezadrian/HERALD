"""
HERALD Reasoning Module
Logical inference and reasoning capabilities

This module contains the reasoning components:
- Logic Engine: Boolean and first-order logic processing
- Causal Reasoning: Dependency graph and intervention analysis
- Temporal Logic: Event sequence and duration estimation
- MoE Router: Mixture-of-experts routing system
"""

__version__ = "1.0.0"
__author__ = "HERALD Development Team"

# Reasoning components
from .logic import LogicEngine
from .causal import CausalReasoningEngine
from .temporal import TemporalLogicEngine
# from .router import MoERouter  # TODO: Implement in Week 12

__all__ = [
    "LogicEngine",
    "CausalReasoningEngine",
    "TemporalLogicEngine",
    # "MoERouter"  # TODO: Implement in Week 12
] 
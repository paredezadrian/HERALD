"""
HERALD Causal Reasoning System
Causal Inference and Analysis

This module implements:
- Dependency Graph Constructor (DAG)
- Intervention Analysis for hypothetical changes
- Confounding Variable Detection
- Temporal Causality handling
- Causal inference accuracy testing
"""

import logging
import time
from typing import Dict, List, Set, Tuple, Optional, Union, Any, NamedTuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from numba import jit, prange
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import json


class CausalError(Exception):
    """Raised when causal operations fail."""
    pass


class GraphError(Exception):
    """Raised when graph operations fail."""
    pass


class InterventionError(Exception):
    """Raised when intervention analysis fails."""
    pass


class RelationType(Enum):
    """Types of causal relationships."""
    DIRECT = "direct"
    INDIRECT = "indirect"
    CONFOUNDING = "confounding"
    COLLIDER = "collider"
    BACKDOOR = "backdoor"


class TemporalRelation(Enum):
    """Types of temporal relationships."""
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    SIMULTANEOUS = "simultaneous"
    OVERLAPS = "overlaps"


@dataclass
class CausalVariable:
    """Represents a variable in the causal graph."""
    name: str
    domain: List[Any] = field(default_factory=list)
    temporal_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.temporal_info is None:
            self.temporal_info = {}
    
    def __str__(self) -> str:
        return self.name
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, CausalVariable):
            return False
        return self.name == other.name


@dataclass
class CausalEdge:
    """Represents a causal relationship between variables."""
    source: CausalVariable
    target: CausalVariable
    relation_type: RelationType = RelationType.DIRECT
    strength: float = 1.0
    confidence: float = 1.0
    temporal_constraint: Optional[TemporalRelation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.source} -> {self.target} ({self.relation_type.value})"
    
    def __hash__(self) -> int:
        return hash((self.source.name, self.target.name, self.relation_type))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, CausalEdge):
            return False
        return (self.source == other.source and 
                self.target == other.target and 
                self.relation_type == other.relation_type)


@dataclass
class Intervention:
    """Represents a causal intervention."""
    variable: CausalVariable
    value: Any
    intervention_type: str = "do"  # "do" or "see"
    temporal_constraint: Optional[TemporalRelation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.intervention_type}({self.variable} = {self.value})"


@dataclass
class CausalPath:
    """Represents a causal path between variables."""
    path: List[CausalEdge]
    strength: float = 1.0
    path_type: str = "causal"  # "causal", "backdoor", "frontdoor"
    
    def __str__(self) -> str:
        if not self.path:
            return "empty_path"
        return " -> ".join(f"{edge.source}" for edge in self.path) + f" -> {self.path[-1].target}"
    
    def get_variables(self) -> List[CausalVariable]:
        """Get all variables in the path."""
        variables = []
        if self.path:
            variables.append(self.path[0].source)
            for edge in self.path:
                variables.append(edge.target)
        return variables


class DependencyGraphConstructor:
    """Constructs and manages causal dependency graphs (DAGs)."""
    
    def __init__(self, max_variables: int = 1000):
        self.max_variables = max_variables
        self.variables: Dict[str, CausalVariable] = {}
        self.edges: List[CausalEdge] = []
        self.graph = nx.DiGraph()
        self.logger = logging.getLogger(__name__)
        
    def add_variable(self, name: str, domain: List[Any] = None, 
                    temporal_info: Dict[str, Any] = None) -> CausalVariable:
        """Add a variable to the causal graph."""
        if len(self.variables) >= self.max_variables:
            raise GraphError(f"Maximum number of variables ({self.max_variables}) exceeded")
        
        if name in self.variables:
            raise GraphError(f"Variable '{name}' already exists")
        
        variable = CausalVariable(name=name, domain=domain or [], temporal_info=temporal_info or {})
        self.variables[name] = variable
        self.graph.add_node(name, variable=variable)
        
        self.logger.debug(f"Added variable: {name}")
        return variable
    
    def add_edge(self, source_name: str, target_name: str, 
                relation_type: RelationType = RelationType.DIRECT,
                strength: float = 1.0, confidence: float = 1.0,
                temporal_constraint: TemporalRelation = None) -> CausalEdge:
        """Add a causal edge between variables."""
        if source_name not in self.variables:
            raise GraphError(f"Source variable '{source_name}' not found")
        if target_name not in self.variables:
            raise GraphError(f"Target variable '{target_name}' not found")
        
        source = self.variables[source_name]
        target = self.variables[target_name]
        
        # Check for cycles
        if self._would_create_cycle(source, target):
            raise GraphError(f"Adding edge {source} -> {target} would create a cycle")
        
        edge = CausalEdge(
            source=source,
            target=target,
            relation_type=relation_type,
            strength=strength,
            confidence=confidence,
            temporal_constraint=temporal_constraint
        )
        
        self.edges.append(edge)
        self.graph.add_edge(source_name, target_name, edge=edge)
        
        self.logger.debug(f"Added edge: {edge}")
        return edge
    
    def _would_create_cycle(self, source: CausalVariable, target: CausalVariable) -> bool:
        """Check if adding an edge would create a cycle."""
        # Create a temporary graph to test
        temp_graph = self.graph.copy()
        temp_graph.add_edge(source.name, target.name)
        
        try:
            # Check if the graph is still acyclic
            return not nx.is_directed_acyclic_graph(temp_graph)
        except nx.NetworkXError:
            return True
    
    def remove_edge(self, source_name: str, target_name: str) -> bool:
        """Remove a causal edge."""
        edge_to_remove = None
        for edge in self.edges:
            if edge.source.name == source_name and edge.target.name == target_name:
                edge_to_remove = edge
                break
        
        if edge_to_remove:
            self.edges.remove(edge_to_remove)
            self.graph.remove_edge(source_name, target_name)
            self.logger.debug(f"Removed edge: {edge_to_remove}")
            return True
        
        return False
    
    def get_ancestors(self, variable_name: str) -> Set[str]:
        """Get all ancestors of a variable."""
        if variable_name not in self.variables:
            raise GraphError(f"Variable '{variable_name}' not found")
        
        ancestors = set()
        queue = deque([variable_name])
        visited = set()
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            for pred in self.graph.predecessors(current):
                if pred not in ancestors:
                    ancestors.add(pred)
                    queue.append(pred)
        
        return ancestors
    
    def get_descendants(self, variable_name: str) -> Set[str]:
        """Get all descendants of a variable."""
        if variable_name not in self.variables:
            raise GraphError(f"Variable '{variable_name}' not found")
        
        descendants = set()
        queue = deque([variable_name])
        visited = set()
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            for succ in self.graph.successors(current):
                if succ not in descendants:
                    descendants.add(succ)
                    queue.append(succ)
        
        return descendants
    
    def get_parents(self, variable_name: str) -> Set[str]:
        """Get direct parents of a variable."""
        if variable_name not in self.variables:
            raise GraphError(f"Variable '{variable_name}' not found")
        
        return set(self.graph.predecessors(variable_name))
    
    def get_children(self, variable_name: str) -> Set[str]:
        """Get direct children of a variable."""
        if variable_name not in self.variables:
            raise GraphError(f"Variable '{variable_name}' not found")
        
        return set(self.graph.successors(variable_name))
    
    def is_acyclic(self) -> bool:
        """Check if the graph is acyclic."""
        try:
            return nx.is_directed_acyclic_graph(self.graph)
        except nx.NetworkXError:
            return False
    
    def get_topological_order(self) -> List[str]:
        """Get topological ordering of variables."""
        if not self.is_acyclic():
            raise GraphError("Graph contains cycles, cannot compute topological order")
        
        return list(nx.topological_sort(self.graph))
    
    def get_causal_paths(self, source_name: str, target_name: str) -> List[CausalPath]:
        """Get all causal paths from source to target."""
        if source_name not in self.variables or target_name not in self.variables:
            raise GraphError("Source or target variable not found")
        
        paths = []
        all_paths = list(nx.all_simple_paths(self.graph, source_name, target_name))
        
        for path_nodes in all_paths:
            path_edges = []
            for i in range(len(path_nodes) - 1):
                source = path_nodes[i]
                target = path_nodes[i + 1]
                edge_data = self.graph.get_edge_data(source, target)
                if edge_data and 'edge' in edge_data:
                    path_edges.append(edge_data['edge'])
            
            if path_edges:
                # Calculate path strength as product of edge strengths
                path_strength = 1.0
                for edge in path_edges:
                    path_strength *= edge.strength
                
                causal_path = CausalPath(path=path_edges, strength=path_strength)
                paths.append(causal_path)
        
        return paths
    
    def export_graph(self) -> Dict[str, Any]:
        """Export the graph structure for serialization."""
        return {
            'variables': {name: {
                'name': var.name,
                'domain': var.domain,
                'temporal_info': var.temporal_info,
                'metadata': var.metadata
            } for name, var in self.variables.items()},
            'edges': [{
                'source': edge.source.name,
                'target': edge.target.name,
                'relation_type': edge.relation_type.value,
                'strength': edge.strength,
                'confidence': edge.confidence,
                'temporal_constraint': edge.temporal_constraint.value if edge.temporal_constraint else None,
                'metadata': edge.metadata
            } for edge in self.edges]
        }
    
    def import_graph(self, graph_data: Dict[str, Any]) -> None:
        """Import a graph structure from serialized data."""
        # Clear existing graph
        self.variables.clear()
        self.edges.clear()
        self.graph.clear()
        
        # Import variables
        for name, var_data in graph_data['variables'].items():
            variable = CausalVariable(
                name=var_data['name'],
                domain=var_data['domain'],
                temporal_info=var_data['temporal_info'],
                metadata=var_data['metadata']
            )
            self.variables[name] = variable
            self.graph.add_node(name, variable=variable)
        
        # Import edges
        for edge_data in graph_data['edges']:
            source = self.variables[edge_data['source']]
            target = self.variables[edge_data['target']]
            
            temporal_constraint = None
            if edge_data['temporal_constraint']:
                temporal_constraint = TemporalRelation(edge_data['temporal_constraint'])
            
            edge = CausalEdge(
                source=source,
                target=target,
                relation_type=RelationType(edge_data['relation_type']),
                strength=edge_data['strength'],
                confidence=edge_data['confidence'],
                temporal_constraint=temporal_constraint,
                metadata=edge_data['metadata']
            )
            
            self.edges.append(edge)
            self.graph.add_edge(source.name, target.name, edge=edge)


class InterventionAnalysis:
    """Performs causal intervention analysis."""
    
    def __init__(self, graph_constructor: DependencyGraphConstructor):
        self.graph = graph_constructor
        self.logger = logging.getLogger(__name__)
        
    def perform_intervention(self, intervention: Intervention) -> Dict[str, Any]:
        """Perform a causal intervention and analyze its effects."""
        if intervention.variable.name not in self.graph.variables:
            raise InterventionError(f"Variable '{intervention.variable.name}' not found in graph")
        
        # Get all descendants of the intervened variable
        descendants = self.graph.get_descendants(intervention.variable.name)
        
        # Simulate the intervention effect
        effects = {
            'intervened_variable': intervention.variable.name,
            'intervention_value': intervention.value,
            'affected_variables': list(descendants),
            'causal_paths': [],
            'estimated_effects': {}
        }
        
        # Analyze causal paths from intervened variable to each descendant
        for descendant in descendants:
            paths = self.graph.get_causal_paths(intervention.variable.name, descendant)
            effects['causal_paths'].append({
                'target': descendant,
                'paths': [str(path) for path in paths],
                'total_strength': sum(path.strength for path in paths)
            })
        
        # Estimate effects based on path strengths
        for path_info in effects['causal_paths']:
            target = path_info['target']
            total_strength = path_info['total_strength']
            
            # Simple effect estimation (in practice, this would use more sophisticated methods)
            estimated_effect = total_strength * 0.5  # Placeholder calculation
            effects['estimated_effects'][target] = estimated_effect
        
        self.logger.info(f"Performed intervention: {intervention}")
        return effects
    
    def counterfactual_analysis(self, intervention: Intervention, 
                              observed_outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Perform counterfactual analysis."""
        # Get the intervention effects
        intervention_effects = self.perform_intervention(intervention)
        
        # Calculate counterfactual outcomes
        counterfactual_outcomes = {}
        for variable, observed_value in observed_outcome.items():
            if variable in intervention_effects['estimated_effects']:
                estimated_effect = intervention_effects['estimated_effects'][variable]
                # Simple counterfactual calculation
                counterfactual_value = observed_value - estimated_effect
                counterfactual_outcomes[variable] = counterfactual_value
        
        return {
            'intervention': str(intervention),
            'observed_outcome': observed_outcome,
            'intervention_effects': intervention_effects,
            'counterfactual_outcomes': counterfactual_outcomes
        }
    
    def sensitivity_analysis(self, intervention: Intervention, 
                           parameter_range: Tuple[float, float], 
                           steps: int = 10) -> Dict[str, Any]:
        """Perform sensitivity analysis for an intervention."""
        effects_over_range = []
        
        for i in range(steps):
            # Vary the intervention strength
            strength_factor = parameter_range[0] + (parameter_range[1] - parameter_range[0]) * i / (steps - 1)
            
            # Create modified intervention
            modified_intervention = Intervention(
                variable=intervention.variable,
                value=intervention.value,
                intervention_type=intervention.intervention_type,
                temporal_constraint=intervention.temporal_constraint,
                metadata={**intervention.metadata, 'strength_factor': strength_factor}
            )
            
            effects = self.perform_intervention(modified_intervention)
            effects_over_range.append({
                'strength_factor': strength_factor,
                'effects': effects
            })
        
        return {
            'intervention': str(intervention),
            'parameter_range': parameter_range,
            'sensitivity_results': effects_over_range
        }


class ConfoundingVariableDetection:
    """Detects confounding variables in causal relationships."""
    
    def __init__(self, graph_constructor: DependencyGraphConstructor):
        self.graph = graph_constructor
        self.logger = logging.getLogger(__name__)
        
    def detect_confounders(self, treatment: str, outcome: str) -> List[Dict[str, Any]]:
        """Detect confounding variables between treatment and outcome."""
        if treatment not in self.graph.variables or outcome not in self.graph.variables:
            raise CausalError("Treatment or outcome variable not found")
        
        confounders = []
        
        # Get all variables in the graph
        all_variables = set(self.graph.variables.keys())
        
        for variable in all_variables:
            if variable in [treatment, outcome]:
                continue
            
            # Check if variable is a confounder
            if self._is_confounder(variable, treatment, outcome):
                confounder_info = {
                    'variable': variable,
                    'confounding_strength': self._calculate_confounding_strength(variable, treatment, outcome),
                    'backdoor_paths': self._find_backdoor_paths(variable, treatment, outcome),
                    'adjustment_set': self._find_adjustment_set(variable, treatment, outcome)
                }
                confounders.append(confounder_info)
        
        # Sort by confounding strength
        confounders.sort(key=lambda x: x['confounding_strength'], reverse=True)
        
        self.logger.info(f"Detected {len(confounders)} confounders for {treatment} -> {outcome}")
        return confounders
    
    def _is_confounder(self, variable: str, treatment: str, outcome: str) -> bool:
        """Check if a variable is a confounder."""
        # A confounder is a common cause of both treatment and outcome
        treatment_ancestors = self.graph.get_ancestors(treatment)
        outcome_ancestors = self.graph.get_ancestors(outcome)
        
        return variable in treatment_ancestors and variable in outcome_ancestors
    
    def _calculate_confounding_strength(self, confounder: str, treatment: str, outcome: str) -> float:
        """Calculate the strength of confounding."""
        # Find paths from confounder to treatment and outcome
        paths_to_treatment = self.graph.get_causal_paths(confounder, treatment)
        paths_to_outcome = self.graph.get_causal_paths(confounder, outcome)
        
        # Calculate total path strengths
        treatment_strength = sum(path.strength for path in paths_to_treatment)
        outcome_strength = sum(path.strength for path in paths_to_outcome)
        
        # Confounding strength is the product of path strengths
        return treatment_strength * outcome_strength
    
    def _find_backdoor_paths(self, confounder: str, treatment: str, outcome: str) -> List[List[str]]:
        """Find backdoor paths from confounder to treatment and outcome."""
        backdoor_paths = []
        
        # Paths from confounder to treatment
        treatment_paths = list(nx.all_simple_paths(self.graph.graph, confounder, treatment))
        # Paths from confounder to outcome
        outcome_paths = list(nx.all_simple_paths(self.graph.graph, confounder, outcome))
        
        for t_path in treatment_paths:
            for o_path in outcome_paths:
                # Check if paths are backdoor (don't go through treatment)
                if treatment not in t_path[1:]:  # Exclude the start node
                    backdoor_paths.append(t_path + o_path[1:])  # Avoid double counting confounder
        
        return backdoor_paths
    
    def _find_adjustment_set(self, confounder: str, treatment: str, outcome: str) -> Set[str]:
        """Find the adjustment set for a confounder."""
        # For simplicity, return the confounder itself
        # In practice, this would implement more sophisticated algorithms
        return {confounder}
    
    def detect_colliders(self, treatment: str, outcome: str) -> List[Dict[str, Any]]:
        """Detect collider variables between treatment and outcome."""
        if treatment not in self.graph.variables or outcome not in self.graph.variables:
            raise CausalError("Treatment or outcome variable not found")
        
        colliders = []
        
        # Find variables that are descendants of both treatment and outcome
        treatment_descendants = self.graph.get_descendants(treatment)
        outcome_descendants = self.graph.get_descendants(outcome)
        
        collider_variables = treatment_descendants.intersection(outcome_descendants)
        
        for collider in collider_variables:
            collider_info = {
                'variable': collider,
                'collider_strength': self._calculate_collider_strength(collider, treatment, outcome),
                'blocking_paths': self._find_blocking_paths(collider, treatment, outcome)
            }
            colliders.append(collider_info)
        
        # Sort by collider strength
        colliders.sort(key=lambda x: x['collider_strength'], reverse=True)
        
        self.logger.info(f"Detected {len(colliders)} colliders for {treatment} -> {outcome}")
        return colliders
    
    def _calculate_collider_strength(self, collider: str, treatment: str, outcome: str) -> float:
        """Calculate the strength of collider effect."""
        # Find paths from treatment and outcome to collider
        paths_from_treatment = self.graph.get_causal_paths(treatment, collider)
        paths_from_outcome = self.graph.get_causal_paths(outcome, collider)
        
        # Calculate total path strengths
        treatment_strength = sum(path.strength for path in paths_from_treatment)
        outcome_strength = sum(path.strength for path in paths_from_outcome)
        
        # Collider strength is the product of path strengths
        return treatment_strength * outcome_strength
    
    def _find_blocking_paths(self, collider: str, treatment: str, outcome: str) -> List[List[str]]:
        """Find paths that are blocked by the collider."""
        # This is a simplified implementation
        # In practice, this would identify paths that are blocked when conditioning on the collider
        return []


class TemporalCausality:
    """Handles temporal causality and temporal relationships."""
    
    def __init__(self, graph_constructor: DependencyGraphConstructor):
        self.graph = graph_constructor
        self.logger = logging.getLogger(__name__)
        
    def add_temporal_constraint(self, source: str, target: str, 
                              temporal_relation: TemporalRelation) -> None:
        """Add a temporal constraint to a causal relationship."""
        if source not in self.graph.variables or target not in self.graph.variables:
            raise CausalError("Source or target variable not found")
        
        # Find the edge and update its temporal constraint
        for edge in self.graph.edges:
            if edge.source.name == source and edge.target.name == target:
                edge.temporal_constraint = temporal_relation
                self.logger.debug(f"Added temporal constraint: {source} {temporal_relation.value} {target}")
                return
        
        raise CausalError(f"No edge found between {source} and {target}")
    
    def check_temporal_consistency(self) -> bool:
        """Check if all temporal constraints are consistent."""
        # Check for temporal cycles
        temporal_graph = nx.DiGraph()
        
        for edge in self.graph.edges:
            if edge.temporal_constraint:
                source = edge.source.name
                target = edge.target.name
                
                if edge.temporal_constraint == TemporalRelation.BEFORE:
                    temporal_graph.add_edge(source, target)
                elif edge.temporal_constraint == TemporalRelation.AFTER:
                    temporal_graph.add_edge(target, source)
                elif edge.temporal_constraint == TemporalRelation.SIMULTANEOUS:
                    # Add bidirectional edge for simultaneous events
                    temporal_graph.add_edge(source, target)
                    temporal_graph.add_edge(target, source)
        
        # Check for cycles in temporal graph
        try:
            return nx.is_directed_acyclic_graph(temporal_graph)
        except nx.NetworkXError:
            return False
    
    def get_temporal_order(self) -> List[List[str]]:
        """Get temporal ordering of variables."""
        if not self.check_temporal_consistency():
            raise CausalError("Temporal constraints are inconsistent")
        
        temporal_graph = nx.DiGraph()
        
        for edge in self.graph.edges:
            if edge.temporal_constraint:
                source = edge.source.name
                target = edge.target.name
                
                if edge.temporal_constraint == TemporalRelation.BEFORE:
                    temporal_graph.add_edge(source, target)
                elif edge.temporal_constraint == TemporalRelation.AFTER:
                    temporal_graph.add_edge(target, source)
        
        # Get topological sort
        try:
            return list(nx.topological_sort(temporal_graph))
        except nx.NetworkXError:
            return []
    
    def analyze_temporal_causality(self, event_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal causality in an event sequence."""
        analysis = {
            'event_sequence': event_sequence,
            'temporal_relationships': [],
            'causal_chains': [],
            'temporal_consistency': self.check_temporal_consistency()
        }
        
        # Analyze temporal relationships between events
        for i, event1 in enumerate(event_sequence):
            for j, event2 in enumerate(event_sequence[i+1:], i+1):
                relationship = self._analyze_temporal_relationship(event1, event2)
                if relationship:
                    analysis['temporal_relationships'].append(relationship)
        
        # Find causal chains
        analysis['causal_chains'] = self._find_causal_chains(event_sequence)
        
        return analysis
    
    def _analyze_temporal_relationship(self, event1: Dict[str, Any], 
                                    event2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze temporal relationship between two events."""
        # Extract temporal information
        time1 = event1.get('timestamp', 0)
        time2 = event2.get('timestamp', 0)
        
        if time1 < time2:
            relation = TemporalRelation.BEFORE
        elif time1 > time2:
            relation = TemporalRelation.AFTER
        else:
            relation = TemporalRelation.SIMULTANEOUS
        
        return {
            'event1': event1.get('name', 'unknown'),
            'event2': event2.get('name', 'unknown'),
            'temporal_relation': relation.value,
            'time_difference': abs(time2 - time1)
        }
    
    def _find_causal_chains(self, event_sequence: List[Dict[str, Any]]) -> List[List[str]]:
        """Find causal chains in the event sequence."""
        chains = []
        
        # Simple implementation: find sequences where events are causally related
        for i in range(len(event_sequence)):
            chain = [event_sequence[i].get('name', f'event_{i}')]
            
            for j in range(i + 1, len(event_sequence)):
                # Check if there's a causal relationship
                if self._are_causally_related(event_sequence[i], event_sequence[j]):
                    chain.append(event_sequence[j].get('name', f'event_{j}'))
            
            if len(chain) > 1:
                chains.append(chain)
        
        return chains
    
    def _are_causally_related(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> bool:
        """Check if two events are causally related."""
        # Simple heuristic: events are causally related if they involve the same variables
        # and have appropriate temporal ordering
        variables1 = set(event1.get('variables', []))
        variables2 = set(event2.get('variables', []))
        
        # Check for variable overlap
        if variables1.intersection(variables2):
            time1 = event1.get('timestamp', 0)
            time2 = event2.get('timestamp', 0)
            return time1 < time2  # Event1 must precede Event2
        
        return False


class CausalInferenceAccuracy:
    """Tests and validates causal inference accuracy."""
    
    def __init__(self, graph_constructor: DependencyGraphConstructor):
        self.graph = graph_constructor
        self.logger = logging.getLogger(__name__)
        
    def test_causal_inference(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test causal inference accuracy on a set of test cases."""
        results = {
            'total_tests': len(test_cases),
            'passed_tests': 0,
            'failed_tests': 0,
            'test_results': [],
            'accuracy': 0.0
        }
        
        for i, test_case in enumerate(test_cases):
            test_result = self._run_single_test(test_case)
            results['test_results'].append(test_result)
            
            if test_result['passed']:
                results['passed_tests'] += 1
            else:
                results['failed_tests'] += 1
        
        results['accuracy'] = results['passed_tests'] / results['total_tests']
        
        self.logger.info(f"Causal inference accuracy: {results['accuracy']:.2%}")
        return results
    
    def _run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single causal inference test."""
        test_type = test_case.get('type', 'unknown')
        
        if test_type == 'intervention':
            return self._test_intervention(test_case)
        elif test_type == 'confounding':
            return self._test_confounding(test_case)
        elif test_type == 'temporal':
            return self._test_temporal(test_case)
        else:
            return {
                'test_id': test_case.get('id', 'unknown'),
                'type': test_type,
                'passed': False,
                'error': f"Unknown test type: {test_type}"
            }
    
    def _test_intervention(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Test intervention analysis."""
        try:
            intervention_data = test_case['intervention']
            variable = self.graph.variables.get(intervention_data['variable'])
            
            if not variable:
                return {
                    'test_id': test_case.get('id', 'unknown'),
                    'type': 'intervention',
                    'passed': False,
                    'error': f"Variable '{intervention_data['variable']}' not found"
                }
            
            intervention = Intervention(
                variable=variable,
                value=intervention_data['value'],
                intervention_type=intervention_data.get('type', 'do')
            )
            
            intervention_analysis = InterventionAnalysis(self.graph)
            effects = intervention_analysis.perform_intervention(intervention)
            
            # Check if results match expected
            expected_effects = test_case.get('expected_effects', {})
            passed = self._compare_effects(effects, expected_effects)
            
            return {
                'test_id': test_case.get('id', 'unknown'),
                'type': 'intervention',
                'passed': passed,
                'effects': effects,
                'expected_effects': expected_effects
            }
            
        except Exception as e:
            return {
                'test_id': test_case.get('id', 'unknown'),
                'type': 'intervention',
                'passed': False,
                'error': str(e)
            }
    
    def _test_confounding(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Test confounding variable detection."""
        try:
            treatment = test_case['treatment']
            outcome = test_case['outcome']
            
            confounding_detector = ConfoundingVariableDetection(self.graph)
            confounders = confounding_detector.detect_confounders(treatment, outcome)
            
            # Check if expected confounders are detected
            expected_confounders = test_case.get('expected_confounders', [])
            detected_confounder_names = [c['variable'] for c in confounders]
            
            passed = all(conf in detected_confounder_names for conf in expected_confounders)
            
            return {
                'test_id': test_case.get('id', 'unknown'),
                'type': 'confounding',
                'passed': passed,
                'detected_confounders': confounders,
                'expected_confounders': expected_confounders
            }
            
        except Exception as e:
            return {
                'test_id': test_case.get('id', 'unknown'),
                'type': 'confounding',
                'passed': False,
                'error': str(e)
            }
    
    def _test_temporal(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Test temporal causality."""
        try:
            event_sequence = test_case['event_sequence']
            
            temporal_analyzer = TemporalCausality(self.graph)
            analysis = temporal_analyzer.analyze_temporal_causality(event_sequence)
            
            # Check if temporal consistency is maintained
            passed = analysis['temporal_consistency']
            
            return {
                'test_id': test_case.get('id', 'unknown'),
                'type': 'temporal',
                'passed': passed,
                'analysis': analysis
            }
            
        except Exception as e:
            return {
                'test_id': test_case.get('id', 'unknown'),
                'type': 'temporal',
                'passed': False,
                'error': str(e)
            }
    
    def _compare_effects(self, actual_effects: Dict[str, Any], 
                        expected_effects: Dict[str, Any]) -> bool:
        """Compare actual effects with expected effects."""
        # Simple comparison - in practice, this would be more sophisticated
        for key, expected_value in expected_effects.items():
            if key not in actual_effects:
                return False
            
            actual_value = actual_effects[key]
            if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                # Allow for some tolerance in numerical comparisons
                if abs(actual_value - expected_value) > 0.1:
                    return False
            elif actual_value != expected_value:
                return False
        
        return True
    
    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate comprehensive test cases for causal inference."""
        test_cases = []
        
        # Test case 1: Simple intervention
        test_cases.append({
            'id': 'test_intervention_1',
            'type': 'intervention',
            'intervention': {
                'variable': 'X',
                'value': 1,
                'type': 'do'
            },
            'expected_effects': {
                'affected_variables': ['Y', 'Z']
            }
        })
        
        # Test case 2: Confounding detection
        test_cases.append({
            'id': 'test_confounding_1',
            'type': 'confounding',
            'treatment': 'X',
            'outcome': 'Y',
            'expected_confounders': ['C']
        })
        
        # Test case 3: Temporal causality
        test_cases.append({
            'id': 'test_temporal_1',
            'type': 'temporal',
            'event_sequence': [
                {'name': 'A', 'timestamp': 1, 'variables': ['X']},
                {'name': 'B', 'timestamp': 2, 'variables': ['Y']},
                {'name': 'C', 'timestamp': 3, 'variables': ['Z']}
            ]
        })
        
        return test_cases


class CausalReasoningEngine:
    """Main causal reasoning engine that integrates all components."""
    
    def __init__(self, max_variables: int = 1000):
        self.graph_constructor = DependencyGraphConstructor(max_variables)
        self.intervention_analyzer = InterventionAnalysis(self.graph_constructor)
        self.confounding_detector = ConfoundingVariableDetection(self.graph_constructor)
        self.temporal_analyzer = TemporalCausality(self.graph_constructor)
        self.accuracy_tester = CausalInferenceAccuracy(self.graph_constructor)
        self.logger = logging.getLogger(__name__)
        
    def add_variable(self, name: str, domain: List[Any] = None, 
                    temporal_info: Dict[str, Any] = None) -> CausalVariable:
        """Add a variable to the causal graph."""
        return self.graph_constructor.add_variable(name, domain, temporal_info)
    
    def add_causal_relationship(self, source: str, target: str, 
                              relation_type: RelationType = RelationType.DIRECT,
                              strength: float = 1.0, confidence: float = 1.0,
                              temporal_constraint: TemporalRelation = None) -> CausalEdge:
        """Add a causal relationship between variables."""
        return self.graph_constructor.add_edge(source, target, relation_type, 
                                            strength, confidence, temporal_constraint)
    
    def perform_intervention_analysis(self, intervention: Intervention) -> Dict[str, Any]:
        """Perform intervention analysis."""
        return self.intervention_analyzer.perform_intervention(intervention)
    
    def detect_confounding_variables(self, treatment: str, outcome: str) -> List[Dict[str, Any]]:
        """Detect confounding variables."""
        return self.confounding_detector.detect_confounders(treatment, outcome)
    
    def analyze_temporal_causality(self, event_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal causality."""
        return self.temporal_analyzer.analyze_temporal_causality(event_sequence)
    
    def test_causal_inference_accuracy(self, test_cases: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Test causal inference accuracy."""
        if test_cases is None:
            test_cases = self.accuracy_tester.generate_test_cases()
        
        return self.accuracy_tester.test_causal_inference(test_cases)
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the causal graph."""
        return {
            'num_variables': len(self.graph_constructor.variables),
            'num_edges': len(self.graph_constructor.edges),
            'is_acyclic': self.graph_constructor.is_acyclic(),
            'topological_order': self.graph_constructor.get_topological_order(),
            'variables': list(self.graph_constructor.variables.keys())
        }
    
    def export_graph(self) -> Dict[str, Any]:
        """Export the causal graph."""
        return self.graph_constructor.export_graph()
    
    def import_graph(self, graph_data: Dict[str, Any]) -> None:
        """Import a causal graph."""
        self.graph_constructor.import_graph(graph_data)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'graph_size': len(self.graph_constructor.variables),
            'edge_count': len(self.graph_constructor.edges),
            'memory_usage': 'N/A',  # Would implement actual memory tracking
            'computation_time': 'N/A'  # Would implement actual timing
        }


# Optimized functions for performance
@jit(nopython=True)
def _optimized_path_strength_calculation(path_strengths: np.ndarray) -> float:
    """Optimized calculation of path strength."""
    result = 1.0
    for strength in path_strengths:
        result *= strength
    return result


@jit(nopython=True, parallel=True)
def _optimized_confounding_detection(adjacency_matrix: np.ndarray, 
                                   treatment_idx: int, 
                                   outcome_idx: int) -> np.ndarray:
    """Optimized confounding variable detection."""
    num_variables = adjacency_matrix.shape[0]
    confounders = np.zeros(num_variables, dtype=np.bool_)
    
    for i in prange(num_variables):
        if i != treatment_idx and i != outcome_idx:
            # Check if variable is ancestor of both treatment and outcome
            is_treatment_ancestor = adjacency_matrix[i, treatment_idx] > 0
            is_outcome_ancestor = adjacency_matrix[i, outcome_idx] > 0
            confounders[i] = is_treatment_ancestor and is_outcome_ancestor
    
    return confounders


if __name__ == "__main__":
    # Example usage
    engine = CausalReasoningEngine()
    
    # Add variables
    engine.add_variable("X", domain=[0, 1])
    engine.add_variable("Y", domain=[0, 1])
    engine.add_variable("Z", domain=[0, 1])
    engine.add_variable("C", domain=[0, 1])
    
    # Add causal relationships
    engine.add_causal_relationship("C", "X", RelationType.DIRECT, 0.8)
    engine.add_causal_relationship("C", "Y", RelationType.DIRECT, 0.7)
    engine.add_causal_relationship("X", "Y", RelationType.DIRECT, 0.6)
    engine.add_causal_relationship("Y", "Z", RelationType.DIRECT, 0.5)
    
    # Test causal inference
    test_results = engine.test_causal_inference_accuracy()
    print(f"Causal inference accuracy: {test_results['accuracy']:.2%}")
    
    # Get graph summary
    summary = engine.get_graph_summary()
    print(f"Graph has {summary['num_variables']} variables and {summary['num_edges']} edges") 
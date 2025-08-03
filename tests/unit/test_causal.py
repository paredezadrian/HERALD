"""
Unit tests for HERALD Causal Reasoning System
Tests all components: DependencyGraphConstructor, InterventionAnalysis, 
ConfoundingVariableDetection, TemporalCausality, and CausalInferenceAccuracy
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from reasoning.causal import (
    CausalReasoningEngine, DependencyGraphConstructor, InterventionAnalysis,
    ConfoundingVariableDetection, TemporalCausality, CausalInferenceAccuracy,
    CausalVariable, CausalEdge, Intervention, CausalPath,
    RelationType, TemporalRelation, CausalError, GraphError, InterventionError
)


class TestCausalVariable:
    """Test CausalVariable class."""
    
    def test_causal_variable_creation(self):
        """Test creating a causal variable."""
        var = CausalVariable("X", domain=[0, 1, 2])
        assert var.name == "X"
        assert var.domain == [0, 1, 2]
        assert var.temporal_info == {}
        assert var.metadata == {}
    
    def test_causal_variable_string_representation(self):
        """Test string representation of causal variable."""
        var = CausalVariable("Y")
        assert str(var) == "Y"
    
    def test_causal_variable_equality(self):
        """Test equality comparison of causal variables."""
        var1 = CausalVariable("X")
        var2 = CausalVariable("X")
        var3 = CausalVariable("Y")
        
        assert var1 == var2
        assert var1 != var3
        assert var1 != "not a variable"
    
    def test_causal_variable_hash(self):
        """Test hashing of causal variables."""
        var1 = CausalVariable("X")
        var2 = CausalVariable("X")
        
        assert hash(var1) == hash(var2)
        
        # Test that variables can be used in sets
        var_set = {var1, var2}
        assert len(var_set) == 1


class TestCausalEdge:
    """Test CausalEdge class."""
    
    def test_causal_edge_creation(self):
        """Test creating a causal edge."""
        source = CausalVariable("X")
        target = CausalVariable("Y")
        edge = CausalEdge(source, target, RelationType.DIRECT, 0.8)
        
        assert edge.source == source
        assert edge.target == target
        assert edge.relation_type == RelationType.DIRECT
        assert edge.strength == 0.8
        assert edge.confidence == 1.0
    
    def test_causal_edge_string_representation(self):
        """Test string representation of causal edge."""
        source = CausalVariable("X")
        target = CausalVariable("Y")
        edge = CausalEdge(source, target, RelationType.DIRECT)
        
        assert str(edge) == "X -> Y (direct)"
    
    def test_causal_edge_equality(self):
        """Test equality comparison of causal edges."""
        source1 = CausalVariable("X")
        target1 = CausalVariable("Y")
        source2 = CausalVariable("X")
        target2 = CausalVariable("Y")
        
        edge1 = CausalEdge(source1, target1, RelationType.DIRECT)
        edge2 = CausalEdge(source2, target2, RelationType.DIRECT)
        edge3 = CausalEdge(source1, target1, RelationType.INDIRECT)
        
        assert edge1 == edge2
        assert edge1 != edge3


class TestDependencyGraphConstructor:
    """Test DependencyGraphConstructor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.graph = DependencyGraphConstructor()
    
    def test_add_variable(self):
        """Test adding variables to the graph."""
        var = self.graph.add_variable("X", domain=[0, 1])
        
        assert var.name == "X"
        assert var.domain == [0, 1]
        assert "X" in self.graph.variables
        assert len(self.graph.variables) == 1
    
    def test_add_variable_duplicate(self):
        """Test adding duplicate variables raises error."""
        self.graph.add_variable("X")
        
        with pytest.raises(GraphError, match="already exists"):
            self.graph.add_variable("X")
    
    def test_add_variable_max_limit(self):
        """Test adding variables beyond max limit raises error."""
        # Create a graph with max 2 variables
        small_graph = DependencyGraphConstructor(max_variables=2)
        small_graph.add_variable("X")
        small_graph.add_variable("Y")
        
        with pytest.raises(GraphError, match="Maximum number of variables"):
            small_graph.add_variable("Z")
    
    def test_add_edge(self):
        """Test adding edges to the graph."""
        self.graph.add_variable("X")
        self.graph.add_variable("Y")
        
        edge = self.graph.add_edge("X", "Y", RelationType.DIRECT, 0.8)
        
        assert edge.source.name == "X"
        assert edge.target.name == "Y"
        assert edge.relation_type == RelationType.DIRECT
        assert edge.strength == 0.8
        assert len(self.graph.edges) == 1
    
    def test_add_edge_nonexistent_variables(self):
        """Test adding edge with nonexistent variables raises error."""
        with pytest.raises(GraphError, match="not found"):
            self.graph.add_edge("X", "Y")
    
    def test_add_edge_creates_cycle(self):
        """Test adding edge that creates cycle raises error."""
        self.graph.add_variable("X")
        self.graph.add_variable("Y")
        self.graph.add_variable("Z")
        
        # Create a cycle: X -> Y -> Z -> X
        self.graph.add_edge("X", "Y")
        self.graph.add_edge("Y", "Z")
        
        with pytest.raises(GraphError, match="would create a cycle"):
            self.graph.add_edge("Z", "X")
    
    def test_get_ancestors(self):
        """Test getting ancestors of a variable."""
        self.graph.add_variable("A")
        self.graph.add_variable("B")
        self.graph.add_variable("C")
        self.graph.add_variable("D")
        
        # Create: A -> B -> C, A -> D
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        self.graph.add_edge("A", "D")
        
        ancestors = self.graph.get_ancestors("C")
        assert "A" in ancestors
        assert "B" in ancestors
        assert "C" not in ancestors
    
    def test_get_descendants(self):
        """Test getting descendants of a variable."""
        self.graph.add_variable("A")
        self.graph.add_variable("B")
        self.graph.add_variable("C")
        self.graph.add_variable("D")
        
        # Create: A -> B -> C, A -> D
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        self.graph.add_edge("A", "D")
        
        descendants = self.graph.get_descendants("A")
        assert "B" in descendants
        assert "C" in descendants
        assert "D" in descendants
        assert "A" not in descendants
    
    def test_is_acyclic(self):
        """Test checking if graph is acyclic."""
        self.graph.add_variable("X")
        self.graph.add_variable("Y")
        
        # Acyclic graph
        self.graph.add_edge("X", "Y")
        assert self.graph.is_acyclic()
        
        # Create cycle
        self.graph.add_edge("Y", "X")
        assert not self.graph.is_acyclic()
    
    def test_get_topological_order(self):
        """Test getting topological order."""
        self.graph.add_variable("A")
        self.graph.add_variable("B")
        self.graph.add_variable("C")
        
        # Create: A -> B -> C
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        
        order = self.graph.get_topological_order()
        assert order == ["A", "B", "C"]
    
    def test_get_causal_paths(self):
        """Test getting causal paths between variables."""
        self.graph.add_variable("A")
        self.graph.add_variable("B")
        self.graph.add_variable("C")
        
        # Create: A -> B -> C
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        
        paths = self.graph.get_causal_paths("A", "C")
        assert len(paths) == 1
        assert str(paths[0]) == "A -> B -> C"


class TestInterventionAnalysis:
    """Test InterventionAnalysis class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.graph = DependencyGraphConstructor()
        self.analyzer = InterventionAnalysis(self.graph)
        
        # Create a simple graph: X -> Y -> Z
        self.graph.add_variable("X")
        self.graph.add_variable("Y")
        self.graph.add_variable("Z")
        self.graph.add_edge("X", "Y")
        self.graph.add_edge("Y", "Z")
    
    def test_perform_intervention(self):
        """Test performing an intervention."""
        intervention = Intervention(
            variable=self.graph.variables["X"],
            value=1
        )
        
        effects = self.analyzer.perform_intervention(intervention)
        
        assert effects['intervened_variable'] == "X"
        assert effects['intervention_value'] == 1
        assert "Y" in effects['affected_variables']
        assert "Z" in effects['affected_variables']
        assert len(effects['causal_paths']) == 2
    
    def test_perform_intervention_nonexistent_variable(self):
        """Test intervention with nonexistent variable raises error."""
        nonexistent_var = CausalVariable("Nonexistent")
        intervention = Intervention(variable=nonexistent_var, value=1)
        
        with pytest.raises(InterventionError, match="not found in graph"):
            self.analyzer.perform_intervention(intervention)
    
    def test_counterfactual_analysis(self):
        """Test counterfactual analysis."""
        intervention = Intervention(
            variable=self.graph.variables["X"],
            value=1
        )
        observed_outcome = {"Y": 0.5, "Z": 0.3}
        
        result = self.analyzer.counterfactual_analysis(intervention, observed_outcome)
        
        assert result['intervention'] == "do(X = 1)"
        assert result['observed_outcome'] == observed_outcome
        assert 'counterfactual_outcomes' in result
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        intervention = Intervention(
            variable=self.graph.variables["X"],
            value=1
        )
        
        result = self.analyzer.sensitivity_analysis(
            intervention, 
            parameter_range=(0.1, 1.0), 
            steps=5
        )
        
        assert len(result['sensitivity_results']) == 5
        assert result['parameter_range'] == (0.1, 1.0)


class TestConfoundingVariableDetection:
    """Test ConfoundingVariableDetection class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.graph = DependencyGraphConstructor()
        self.detector = ConfoundingVariableDetection(self.graph)
        
        # Create graph with confounder: C -> X, C -> Y, X -> Y
        self.graph.add_variable("C")
        self.graph.add_variable("X")
        self.graph.add_variable("Y")
        self.graph.add_edge("C", "X")
        self.graph.add_edge("C", "Y")
        self.graph.add_edge("X", "Y")
    
    def test_detect_confounders(self):
        """Test detecting confounding variables."""
        confounders = self.detector.detect_confounders("X", "Y")
        
        assert len(confounders) == 1
        assert confounders[0]['variable'] == "C"
        assert confounders[0]['confounding_strength'] > 0
    
    def test_detect_confounders_nonexistent_variables(self):
        """Test detecting confounders with nonexistent variables raises error."""
        with pytest.raises(CausalError, match="not found"):
            self.detector.detect_confounders("Nonexistent", "Y")
    
    def test_is_confounder(self):
        """Test checking if a variable is a confounder."""
        # C is a confounder of X and Y
        assert self.detector._is_confounder("C", "X", "Y")
        
        # X is not a confounder of X and Y
        assert not self.detector._is_confounder("X", "X", "Y")
    
    def test_detect_colliders(self):
        """Test detecting collider variables."""
        # Add a collider: X -> Z, Y -> Z
        self.graph.add_variable("Z")
        self.graph.add_edge("X", "Z")
        self.graph.add_edge("Y", "Z")
        
        colliders = self.detector.detect_colliders("X", "Y")
        
        assert len(colliders) == 1
        assert colliders[0]['variable'] == "Z"


class TestTemporalCausality:
    """Test TemporalCausality class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.graph = DependencyGraphConstructor()
        self.temporal = TemporalCausality(self.graph)
        
        # Create a simple graph
        self.graph.add_variable("A")
        self.graph.add_variable("B")
        self.graph.add_edge("A", "B")
    
    def test_add_temporal_constraint(self):
        """Test adding temporal constraints."""
        self.temporal.add_temporal_constraint("A", "B", TemporalRelation.BEFORE)
        
        # Check that the edge has the temporal constraint
        edge = self.graph.edges[0]
        assert edge.temporal_constraint == TemporalRelation.BEFORE
    
    def test_add_temporal_constraint_nonexistent_edge(self):
        """Test adding temporal constraint to nonexistent edge raises error."""
        with pytest.raises(CausalError, match="No edge found"):
            self.temporal.add_temporal_constraint("A", "C", TemporalRelation.BEFORE)
    
    def test_check_temporal_consistency(self):
        """Test checking temporal consistency."""
        # Add temporal constraints
        self.temporal.add_temporal_constraint("A", "B", TemporalRelation.BEFORE)
        
        assert self.temporal.check_temporal_consistency()
    
    def test_get_temporal_order(self):
        """Test getting temporal order."""
        self.temporal.add_temporal_constraint("A", "B", TemporalRelation.BEFORE)
        
        order = self.temporal.get_temporal_order()
        assert "A" in order
        assert "B" in order
        assert order.index("A") < order.index("B")
    
    def test_analyze_temporal_causality(self):
        """Test analyzing temporal causality."""
        event_sequence = [
            {"name": "A", "timestamp": 1, "variables": ["X"]},
            {"name": "B", "timestamp": 2, "variables": ["Y"]},
            {"name": "C", "timestamp": 3, "variables": ["Z"]}
        ]
        
        analysis = self.temporal.analyze_temporal_causality(event_sequence)
        
        assert analysis['temporal_consistency']
        assert len(analysis['temporal_relationships']) > 0
        assert 'causal_chains' in analysis


class TestCausalInferenceAccuracy:
    """Test CausalInferenceAccuracy class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.graph = DependencyGraphConstructor()
        self.tester = CausalInferenceAccuracy(self.graph)
        
        # Create a simple graph for testing
        self.graph.add_variable("X")
        self.graph.add_variable("Y")
        self.graph.add_variable("C")
        self.graph.add_edge("C", "X")
        self.graph.add_edge("C", "Y")
        self.graph.add_edge("X", "Y")
    
    def test_test_causal_inference(self):
        """Test causal inference testing."""
        test_cases = self.tester.generate_test_cases()
        results = self.tester.test_causal_inference(test_cases)
        
        assert 'total_tests' in results
        assert 'passed_tests' in results
        assert 'failed_tests' in results
        assert 'accuracy' in results
        assert results['accuracy'] >= 0.0
        assert results['accuracy'] <= 1.0
    
    def test_generate_test_cases(self):
        """Test generating test cases."""
        test_cases = self.tester.generate_test_cases()
        
        assert len(test_cases) > 0
        for test_case in test_cases:
            assert 'id' in test_case
            assert 'type' in test_case
    
    def test_compare_effects(self):
        """Test comparing effects."""
        actual_effects = {"affected_variables": ["Y", "Z"]}
        expected_effects = {"affected_variables": ["Y", "Z"]}
        
        assert self.tester._compare_effects(actual_effects, expected_effects)
        
        # Test with different effects
        different_effects = {"affected_variables": ["Y"]}
        assert not self.tester._compare_effects(actual_effects, different_effects)


class TestCausalReasoningEngine:
    """Test CausalReasoningEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = CausalReasoningEngine()
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        assert self.engine.graph_constructor is not None
        assert self.engine.intervention_analyzer is not None
        assert self.engine.confounding_detector is not None
        assert self.engine.temporal_analyzer is not None
        assert self.engine.accuracy_tester is not None
    
    def test_add_variable(self):
        """Test adding variables through the engine."""
        var = self.engine.add_variable("X", domain=[0, 1])
        
        assert var.name == "X"
        assert "X" in self.engine.graph_constructor.variables
    
    def test_add_causal_relationship(self):
        """Test adding causal relationships through the engine."""
        self.engine.add_variable("X")
        self.engine.add_variable("Y")
        
        edge = self.engine.add_causal_relationship("X", "Y", RelationType.DIRECT, 0.8)
        
        assert edge.source.name == "X"
        assert edge.target.name == "Y"
        assert edge.strength == 0.8
    
    def test_perform_intervention_analysis(self):
        """Test performing intervention analysis through the engine."""
        self.engine.add_variable("X")
        self.engine.add_variable("Y")
        self.engine.add_causal_relationship("X", "Y")
        
        intervention = Intervention(
            variable=self.engine.graph_constructor.variables["X"],
            value=1
        )
        
        effects = self.engine.perform_intervention_analysis(intervention)
        
        assert effects['intervened_variable'] == "X"
        assert "Y" in effects['affected_variables']
    
    def test_detect_confounding_variables(self):
        """Test detecting confounding variables through the engine."""
        self.engine.add_variable("C")
        self.engine.add_variable("X")
        self.engine.add_variable("Y")
        self.engine.add_causal_relationship("C", "X")
        self.engine.add_causal_relationship("C", "Y")
        self.engine.add_causal_relationship("X", "Y")
        
        confounders = self.engine.detect_confounding_variables("X", "Y")
        
        assert len(confounders) == 1
        assert confounders[0]['variable'] == "C"
    
    def test_analyze_temporal_causality(self):
        """Test analyzing temporal causality through the engine."""
        event_sequence = [
            {"name": "A", "timestamp": 1, "variables": ["X"]},
            {"name": "B", "timestamp": 2, "variables": ["Y"]}
        ]
        
        analysis = self.engine.analyze_temporal_causality(event_sequence)
        
        assert 'temporal_relationships' in analysis
        assert 'causal_chains' in analysis
        assert 'temporal_consistency' in analysis
    
    def test_test_causal_inference_accuracy(self):
        """Test causal inference accuracy through the engine."""
        results = self.engine.test_causal_inference_accuracy()
        
        assert 'accuracy' in results
        assert results['accuracy'] >= 0.0
        assert results['accuracy'] <= 1.0
    
    def test_get_graph_summary(self):
        """Test getting graph summary through the engine."""
        self.engine.add_variable("X")
        self.engine.add_variable("Y")
        self.engine.add_causal_relationship("X", "Y")
        
        summary = self.engine.get_graph_summary()
        
        assert summary['num_variables'] == 2
        assert summary['num_edges'] == 1
        assert summary['is_acyclic']
        assert "X" in summary['variables']
        assert "Y" in summary['variables']
    
    def test_export_import_graph(self):
        """Test exporting and importing graphs."""
        self.engine.add_variable("X")
        self.engine.add_variable("Y")
        self.engine.add_causal_relationship("X", "Y")
        
        # Export graph
        graph_data = self.engine.export_graph()
        
        # Create new engine and import graph
        new_engine = CausalReasoningEngine()
        new_engine.import_graph(graph_data)
        
        # Check that the graph was imported correctly
        summary = new_engine.get_graph_summary()
        assert summary['num_variables'] == 2
        assert summary['num_edges'] == 1
    
    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        stats = self.engine.get_performance_stats()
        
        assert 'graph_size' in stats
        assert 'edge_count' in stats
        assert 'memory_usage' in stats
        assert 'computation_time' in stats


class TestOptimizedFunctions:
    """Test optimized functions."""
    
    def test_optimized_path_strength_calculation(self):
        """Test optimized path strength calculation."""
        from reasoning.causal import _optimized_path_strength_calculation
        
        path_strengths = np.array([0.5, 0.8, 0.3])
        result = _optimized_path_strength_calculation(path_strengths)
        
        expected = 0.5 * 0.8 * 0.3
        assert abs(result - expected) < 1e-10
    
    def test_optimized_confounding_detection(self):
        """Test optimized confounding detection."""
        from reasoning.causal import _optimized_confounding_detection
        
        # Create adjacency matrix for: C -> X, C -> Y, X -> Y
        adjacency_matrix = np.array([
            [0, 0, 0, 0],  # C
            [1, 0, 0, 0],  # X
            [1, 1, 0, 0],  # Y
            [0, 0, 0, 0]   # Z
        ])
        
        confounders = _optimized_confounding_detection(adjacency_matrix, 1, 2)
        
        # C (index 0) should be a confounder of X (index 1) and Y (index 2)
        assert confounders[0] == True
        assert confounders[1] == False  # X is not a confounder of itself
        assert confounders[2] == False  # Y is not a confounder of itself


if __name__ == "__main__":
    pytest.main([__file__]) 
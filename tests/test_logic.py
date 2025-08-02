"""
Unit tests for HERALD Logic Engine

Tests:
- Boolean Satisfiability Solver
- First-Order Logic Engine
- Rule Chain Inference
- Consistency checking
"""

import pytest
import time
import numpy as np
from typing import Dict, List, Set

from reasoning.logic import (
    LogicEngine, BooleanSatisfiabilitySolver, FirstOrderLogicEngine, RuleChainInference,
    Literal, Clause, Formula, Predicate, AtomicFormula, QuantifiedFormula, Rule,
    QuantifierType, LogicError, ConsistencyError
)


class TestLiteral:
    """Test Literal class functionality."""
    
    def test_literal_creation(self):
        """Test literal creation and string representation."""
        lit1 = Literal("x", False)
        lit2 = Literal("y", True)
        
        assert str(lit1) == "x"
        assert str(lit2) == "¬y"
        assert lit1.variable == "x"
        assert lit1.negated == False
        assert lit2.variable == "y"
        assert lit2.negated == True
    
    def test_literal_equality(self):
        """Test literal equality and hashing."""
        lit1 = Literal("x", False)
        lit2 = Literal("x", False)
        lit3 = Literal("x", True)
        lit4 = Literal("y", False)
        
        assert lit1 == lit2
        assert lit1 != lit3
        assert lit1 != lit4
        assert hash(lit1) == hash(lit2)
        assert hash(lit1) != hash(lit3)


class TestClause:
    """Test Clause class functionality."""
    
    def test_clause_creation(self):
        """Test clause creation and operations."""
        clause = Clause()
        lit1 = Literal("x", False)
        lit2 = Literal("y", True)
        
        clause = clause.add_literal(lit1)
        clause = clause.add_literal(lit2)
        
        assert len(clause.literals) == 2
        assert lit1 in clause.literals
        assert lit2 in clause.literals
        # Note: string representation is now sorted, so order may vary
        assert "x" in str(clause) and "¬y" in str(clause)
    
    def test_clause_properties(self):
        """Test clause property methods."""
        # Empty clause
        empty_clause = Clause()
        assert empty_clause.is_empty()
        assert not empty_clause.is_unit()
        assert str(empty_clause) == "⊥"
        
        # Unit clause
        unit_clause = Clause()
        unit_clause = unit_clause.add_literal(Literal("x", False))
        assert not unit_clause.is_empty()
        assert unit_clause.is_unit()
        
        # Regular clause
        regular_clause = Clause()
        regular_clause = regular_clause.add_literal(Literal("x", False))
        regular_clause = regular_clause.add_literal(Literal("y", True))
        assert not regular_clause.is_empty()
        assert not regular_clause.is_unit()
    
    def test_clause_tautology(self):
        """Test tautology detection."""
        # Tautological clause (contains complementary literals)
        tautology = Clause()
        tautology = tautology.add_literal(Literal("x", False))
        tautology = tautology.add_literal(Literal("x", True))
        assert tautology.is_tautology()
        
        # Non-tautological clause
        non_tautology = Clause()
        non_tautology = non_tautology.add_literal(Literal("x", False))
        non_tautology = non_tautology.add_literal(Literal("y", True))
        assert not non_tautology.is_tautology()


class TestFormula:
    """Test Formula class functionality."""
    
    def test_formula_creation(self):
        """Test formula creation and operations."""
        formula = Formula()
        clause1 = Clause()
        clause1 = clause1.add_literal(Literal("x", False))
        clause2 = Clause()
        clause2 = clause2.add_literal(Literal("y", True))
        
        formula.add_clause(clause1)
        formula.add_clause(clause2)
        
        assert len(formula.clauses) == 2
        # Note: string representation is now sorted, so order may vary
        assert "(x)" in str(formula) and "(¬y)" in str(formula)
    
    def test_formula_variables(self):
        """Test variable extraction from formula."""
        formula = Formula()
        clause1 = Clause()
        clause1 = clause1.add_literal(Literal("x", False))
        clause1 = clause1.add_literal(Literal("y", True))
        clause2 = Clause()
        clause2 = clause2.add_literal(Literal("z", False))
        
        formula.add_clause(clause1)
        formula.add_clause(clause2)
        
        variables = formula.get_variables()
        assert "x" in variables
        assert "y" in variables
        assert "z" in variables
        assert len(variables) == 3


class TestBooleanSatisfiabilitySolver:
    """Test Boolean Satisfiability Solver."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = BooleanSatisfiabilitySolver()
    
    def test_simple_satisfiable_formula(self):
        """Test solving a simple satisfiable formula."""
        # Formula: (x ∨ y) ∧ (¬x ∨ z)
        formula = Formula()
        
        clause1 = Clause()
        clause1 = clause1.add_literal(Literal("x", False))
        clause1 = clause1.add_literal(Literal("y", False))
        
        clause2 = Clause()
        clause2 = clause2.add_literal(Literal("x", True))
        clause2 = clause2.add_literal(Literal("z", False))
        
        formula.add_clause(clause1)
        formula.add_clause(clause2)
        
        result = self.solver.solve(formula)
        assert result is not None
        assert isinstance(result, dict)
        
        # Verify the assignment satisfies the formula
        for clause in formula.clauses:
            clause_satisfied = False
            for literal in clause.literals:
                if literal.variable in result:
                    if (result[literal.variable] and not literal.negated) or \
                       (not result[literal.variable] and literal.negated):
                        clause_satisfied = True
                        break
            assert clause_satisfied
    
    def test_unsatisfiable_formula(self):
        """Test solving an unsatisfiable formula."""
        # Formula: x ∧ ¬x (contradiction)
        formula = Formula()
        
        clause1 = Clause()
        clause1 = clause1.add_literal(Literal("x", False))
        
        clause2 = Clause()
        clause2 = clause2.add_literal(Literal("x", True))
        
        formula.add_clause(clause1)
        formula.add_clause(clause2)
        
        result = self.solver.solve(formula)
        assert result is None
    
    def test_empty_formula(self):
        """Test solving an empty formula (tautology)."""
        formula = Formula()
        result = self.solver.solve(formula)
        assert result is not None
        assert result == {}
    
    def test_unit_clause_propagation(self):
        """Test unit clause propagation."""
        # Formula: x ∧ (¬x ∨ y) ∧ (¬x ∨ z)
        formula = Formula()
        
        clause1 = Clause()
        clause1 = clause1.add_literal(Literal("x", False))
        
        clause2 = Clause()
        clause2 = clause2.add_literal(Literal("x", True))
        clause2 = clause2.add_literal(Literal("y", False))
        
        clause3 = Clause()
        clause3 = clause3.add_literal(Literal("x", True))
        clause3 = clause3.add_literal(Literal("z", False))
        
        formula.add_clause(clause1)
        formula.add_clause(clause2)
        formula.add_clause(clause3)
        
        result = self.solver.solve(formula)
        assert result is not None
        assert result["x"] == True  # x must be true due to unit clause
        assert result["y"] == True  # y must be true due to propagation
        assert result["z"] == True  # z must be true due to propagation
    
    def test_pure_literal_elimination(self):
        """Test pure literal elimination."""
        # Formula: (x ∨ y) ∧ (¬y ∨ z) ∧ (¬z ∨ w)
        # x appears only positively, w appears only negatively
        formula = Formula()
        
        clause1 = Clause()
        clause1 = clause1.add_literal(Literal("x", False))
        clause1 = clause1.add_literal(Literal("y", False))
        
        clause2 = Clause()
        clause2 = clause2.add_literal(Literal("y", True))
        clause2 = clause2.add_literal(Literal("z", False))
        
        clause3 = Clause()
        clause3 = clause3.add_literal(Literal("z", True))
        clause3 = clause3.add_literal(Literal("w", True))
        
        formula.add_clause(clause1)
        formula.add_clause(clause2)
        formula.add_clause(clause3)
        
        result = self.solver.solve(formula)
        assert result is not None
        assert result["x"] == True  # x should be assigned true (pure positive)
        assert result["w"] == False  # w should be assigned false (pure negative)


class TestFirstOrderLogicEngine:
    """Test First-Order Logic Engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = FirstOrderLogicEngine()
    
    def test_atomic_formula_proof(self):
        """Test proving atomic formulas."""
        # Add axiom: P(a)
        predicate = Predicate("P", ["a"])
        axiom = AtomicFormula(predicate, False)
        self.engine.add_axiom(axiom)
        
        # Try to prove P(a)
        goal = AtomicFormula(predicate, False)
        result = self.engine.prove(goal)
        assert result == True
    
    def test_quantified_formula_proof(self):
        """Test proving quantified formulas."""
        # Add axiom: ∀x P(x)
        predicate = Predicate("P", ["x"])
        subformula = AtomicFormula(predicate, False)
        axiom = QuantifiedFormula(QuantifierType.UNIVERSAL, "x", subformula)
        self.engine.add_axiom(axiom)
        
        # Try to prove ∃x P(x)
        goal = QuantifiedFormula(QuantifierType.EXISTENTIAL, "x", subformula)
        result = self.engine.prove(goal)
        # This should be provable since ∀x P(x) implies ∃x P(x)
        assert result == True
    
    def test_skolemization(self):
        """Test Skolem function creation."""
        # Test existential quantification
        predicate = Predicate("P", ["x"])
        subformula = AtomicFormula(predicate, False)
        formula = QuantifiedFormula(QuantifierType.EXISTENTIAL, "x", subformula)
        
        skolemized = self.engine.skolemize(formula)
        assert isinstance(skolemized, AtomicFormula)
        assert "skolem" in str(skolemized)


class TestRuleChainInference:
    """Test Rule Chain Inference."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RuleChainInference()
    
    def test_forward_chaining(self):
        """Test forward chaining inference."""
        # Add facts
        fact1 = AtomicFormula(Predicate("P", ["a"]), False)
        fact2 = AtomicFormula(Predicate("Q", ["a"]), False)
        self.engine.add_fact(fact1)
        self.engine.add_fact(fact2)
        
        # Add rule: P(x) ∧ Q(x) → R(x)
        premise1 = AtomicFormula(Predicate("P", ["x"]), False)
        premise2 = AtomicFormula(Predicate("Q", ["x"]), False)
        conclusion = AtomicFormula(Predicate("R", ["x"]), False)
        rule = Rule([premise1, premise2], conclusion)
        self.engine.add_rule(rule)
        
        # Try to derive R(a)
        goal = AtomicFormula(Predicate("R", ["a"]), False)
        derived_facts = self.engine.forward_chain(goal)
        
        assert len(derived_facts) > 0
        assert any(str(fact) == "R(a)" for fact in derived_facts)
    
    def test_backward_chaining(self):
        """Test backward chaining inference."""
        # Add facts
        fact1 = AtomicFormula(Predicate("P", ["a"]), False)
        fact2 = AtomicFormula(Predicate("Q", ["a"]), False)
        self.engine.add_fact(fact1)
        self.engine.add_fact(fact2)
        
        # Add rule: P(x) ∧ Q(x) → R(x)
        premise1 = AtomicFormula(Predicate("P", ["x"]), False)
        premise2 = AtomicFormula(Predicate("Q", ["x"]), False)
        conclusion = AtomicFormula(Predicate("R", ["x"]), False)
        rule = Rule([premise1, premise2], conclusion)
        self.engine.add_rule(rule)
        
        # Try to prove R(a)
        goal = AtomicFormula(Predicate("R", ["a"]), False)
        result = self.engine.backward_chain(goal)
        assert result == True
    
    def test_cycle_detection(self):
        """Test cycle detection in rule chains."""
        # Add circular rule: A → B, B → A
        premise_a = AtomicFormula(Predicate("A", []), False)
        conclusion_b = AtomicFormula(Predicate("B", []), False)
        rule1 = Rule([premise_a], conclusion_b)
        
        premise_b = AtomicFormula(Predicate("B", []), False)
        conclusion_a = AtomicFormula(Predicate("A", []), False)
        rule2 = Rule([premise_b], conclusion_a)
        
        self.engine.add_rule(rule1)
        self.engine.add_rule(rule2)
        
        # Add fact A
        fact_a = AtomicFormula(Predicate("A", []), False)
        self.engine.add_fact(fact_a)
        
        # Try to prove A (should detect cycle)
        result = self.engine.backward_chain(fact_a)
        # Should return True since A is already a fact
        assert result == True
        
        # Check that cycles were detected
        assert self.engine.stats['cycles_detected'] > 0
    
    def test_consistency_checking(self):
        """Test consistency checking."""
        # Add consistent facts
        fact1 = AtomicFormula(Predicate("P", ["a"]), False)
        fact2 = AtomicFormula(Predicate("Q", ["a"]), False)
        self.engine.add_fact(fact1)
        self.engine.add_fact(fact2)
        
        # Check consistency
        result = self.engine.check_consistency()
        assert result == True
        
        # Add contradictory facts
        fact3 = AtomicFormula(Predicate("P", ["a"]), True)  # ¬P(a)
        self.engine.add_fact(fact3)
        
        # Check consistency (should fail)
        result = self.engine.check_consistency()
        assert result == False


class TestLogicEngine:
    """Test main Logic Engine integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = LogicEngine()
    
    def test_sat_solving_integration(self):
        """Test SAT solving through main engine."""
        # Create simple satisfiable formula
        formula = Formula()
        clause = Clause()
        clause = clause.add_literal(Literal("x", False))
        clause = clause.add_literal(Literal("y", False))
        formula.add_clause(clause)
        
        result = self.engine.solve_sat(formula)
        assert result is not None
        assert isinstance(result, dict)
    
    def test_fol_proof_integration(self):
        """Test FOL proof through main engine."""
        # Add axiom and try to prove it
        predicate = Predicate("P", ["a"])
        axiom = AtomicFormula(predicate, False)
        self.engine.fol_engine.add_axiom(axiom)
        
        goal = AtomicFormula(predicate, False)
        result = self.engine.prove_fol(goal)
        assert result == True
    
    def test_rule_inference_integration(self):
        """Test rule inference through main engine."""
        # Add fact and rule
        fact = AtomicFormula(Predicate("P", ["a"]), False)
        self.engine.rule_engine.add_fact(fact)
        
        premise = AtomicFormula(Predicate("P", ["x"]), False)
        conclusion = AtomicFormula(Predicate("Q", ["x"]), False)
        rule = Rule([premise], conclusion)
        self.engine.rule_engine.add_rule(rule)
        
        # Try forward chaining
        goal = AtomicFormula(Predicate("Q", ["a"]), False)
        derived_facts = self.engine.forward_chain(goal)
        assert len(derived_facts) > 0
    
    def test_consistency_checking_integration(self):
        """Test consistency checking through main engine."""
        # Add consistent facts
        fact1 = AtomicFormula(Predicate("P", ["a"]), False)
        fact2 = AtomicFormula(Predicate("Q", ["a"]), False)
        self.engine.rule_engine.add_fact(fact1)
        self.engine.rule_engine.add_fact(fact2)
        
        result = self.engine.check_consistency()
        assert result == True
    
    def test_performance_stats(self):
        """Test performance statistics collection."""
        # Perform some operations
        formula = Formula()
        clause = Clause()
        clause = clause.add_literal(Literal("x", False))
        formula.add_clause(clause)
        
        self.engine.solve_sat(formula)
        
        stats = self.engine.get_performance_stats()
        assert 'sat_solves' in stats
        assert 'total_time' in stats
        assert stats['sat_solves'] > 0
    
    def test_stats_reset(self):
        """Test statistics reset functionality."""
        # Perform some operations
        formula = Formula()
        clause = Clause()
        clause = clause.add_literal(Literal("x", False))
        formula.add_clause(clause)
        
        self.engine.solve_sat(formula)
        
        # Check stats are non-zero
        stats = self.engine.get_performance_stats()
        assert stats['sat_solves'] > 0
        
        # Reset stats
        self.engine.reset_stats()
        
        # Check stats are zero
        stats = self.engine.get_performance_stats()
        assert stats['sat_solves'] == 0


class TestPerformanceOptimizations:
    """Test performance optimization functions."""
    
    def test_optimized_clause_evaluation(self):
        """Test optimized clause evaluation with Numba."""
        from reasoning.logic import _optimized_clause_evaluation
        
        # Create test data
        clause_literals = np.array([[0, 0], [1, 1]], dtype=np.int32)  # x, ¬y
        assignment = np.array([True, False], dtype=np.bool_)  # x=True, y=False
        
        result = _optimized_clause_evaluation(clause_literals, assignment)
        assert result == True  # Clause should be satisfied
    
    def test_optimized_variable_selection(self):
        """Test optimized variable selection with Numba."""
        from reasoning.logic import _optimized_variable_selection
        
        # Create test data
        activity_scores = np.array([0.5, 0.8, 0.3], dtype=np.float64)
        unassigned_mask = np.array([True, True, False], dtype=np.bool_)
        
        selected_var = _optimized_variable_selection(activity_scores, unassigned_mask)
        assert selected_var == 1  # Should select variable with highest score (0.8)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_logic_error_handling(self):
        """Test LogicError exception handling."""
        engine = LogicEngine()
        
        # Test with invalid formula
        with pytest.raises(LogicError):
            # This should raise an error due to invalid input
            engine.solve_sat(None)
    
    def test_consistency_error_handling(self):
        """Test ConsistencyError exception handling."""
        engine = LogicEngine()
        
        # Add contradictory facts
        fact1 = AtomicFormula(Predicate("P", ["a"]), False)
        fact2 = AtomicFormula(Predicate("P", ["a"]), True)
        engine.rule_engine.add_fact(fact1)
        engine.rule_engine.add_fact(fact2)
        
        # This should not raise an exception, but return False
        result = engine.check_consistency()
        assert result == False
    
    def test_timeout_handling(self):
        """Test timeout handling in SAT solving."""
        solver = BooleanSatisfiabilitySolver()
        
        # Create a complex formula that might timeout
        formula = Formula()
        for i in range(100):  # Many variables
            clause = Clause()
            clause = clause.add_literal(Literal(f"x{i}", False))
            clause = clause.add_literal(Literal(f"y{i}", True))
            formula.add_clause(clause)
        
        # Try to solve with very short timeout
        result = solver.solve(formula, timeout=0.001)
        # Should return None due to timeout
        assert result is None


class TestThreadSafety:
    """Test thread safety of Logic Engine components."""
    
    def test_concurrent_sat_solving(self):
        """Test concurrent SAT solving operations."""
        import threading
        import time
        
        engine = LogicEngine()
        results = []
        
        def solve_task():
            formula = Formula()
            clause = Clause()
            clause = clause.add_literal(Literal("x", False))
            formula.add_clause(clause)
            
            result = engine.solve_sat(formula)
            results.append(result)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=solve_task)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all threads got results
        assert len(results) == 5
        assert all(result is not None for result in results)
    
    def test_concurrent_rule_inference(self):
        """Test concurrent rule inference operations."""
        import threading
        
        engine = LogicEngine()
        
        # Add facts and rules
        fact = AtomicFormula(Predicate("P", ["a"]), False)
        engine.rule_engine.add_fact(fact)
        
        premise = AtomicFormula(Predicate("P", ["x"]), False)
        conclusion = AtomicFormula(Predicate("Q", ["x"]), False)
        rule = Rule([premise], conclusion)
        engine.rule_engine.add_rule(rule)
        
        results = []
        
        def inference_task():
            goal = AtomicFormula(Predicate("Q", ["a"]), False)
            result = engine.backward_chain(goal)
            results.append(result)
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=inference_task)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all threads got correct results
        assert len(results) == 3
        assert all(result == True for result in results)


if __name__ == "__main__":
    pytest.main([__file__]) 
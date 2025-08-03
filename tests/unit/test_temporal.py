"""
Unit tests for HERALD Temporal Logic System
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, date
from unittest.mock import Mock, patch

from reasoning.temporal import (
    TemporalLogicEngine, Event, TimePoint, Duration, DurationUnit,
    EventType, TemporalRelation, Calendar, TemporalConstraint,
    EventSequenceModeling, DurationEstimation, TemporalRelationshipReasoning,
    CalendarSupport, _optimized_duration_calculation,
    _optimized_temporal_relationship_matrix, _optimized_working_time_check
)


class TestTimePoint:
    """Test TimePoint class."""
    
    def test_timepoint_creation(self):
        """Test TimePoint creation."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        timepoint = TimePoint(timestamp=timestamp, confidence=0.8)
        
        assert timepoint.timestamp == timestamp
        assert timepoint.confidence == 0.8
        assert str(timepoint) == timestamp.isoformat()
    
    def test_timepoint_comparison(self):
        """Test TimePoint comparison."""
        tp1 = TimePoint(timestamp=datetime(2024, 1, 1, 12, 0, 0))
        tp2 = TimePoint(timestamp=datetime(2024, 1, 1, 13, 0, 0))
        
        assert tp1 < tp2
        assert tp2 > tp1
        assert tp1 == TimePoint(timestamp=datetime(2024, 1, 1, 12, 0, 0))


class TestDuration:
    """Test Duration class."""
    
    def test_duration_creation(self):
        """Test Duration creation."""
        duration = Duration(value=30, unit=DurationUnit.MINUTES)
        
        assert duration.value == 30
        assert duration.unit == DurationUnit.MINUTES
        assert str(duration) == "30min"
    
    def test_duration_conversion(self):
        """Test duration conversion to seconds."""
        duration = Duration(value=2, unit=DurationUnit.HOURS)
        seconds = duration.to_seconds()
        
        assert seconds == 7200.0  # 2 hours = 7200 seconds
    
    def test_duration_timedelta(self):
        """Test duration conversion to timedelta."""
        duration = Duration(value=90, unit=DurationUnit.MINUTES)
        delta = duration.to_timedelta()
        
        assert delta == timedelta(minutes=90)


class TestEvent:
    """Test Event class."""
    
    def test_instantaneous_event(self):
        """Test instantaneous event creation."""
        start_time = TimePoint(timestamp=datetime(2024, 1, 1, 12, 0, 0))
        event = Event(
            name="meeting",
            event_type=EventType.INSTANTANEOUS,
            start_time=start_time,
            description="Team meeting"
        )
        
        assert event.name == "meeting"
        assert event.event_type == EventType.INSTANTANEOUS
        assert event.start_time == start_time
        assert event.end_time is None
        assert event.get_duration().value == 0
    
    def test_durative_event(self):
        """Test durative event creation."""
        start_time = TimePoint(timestamp=datetime(2024, 1, 1, 12, 0, 0))
        end_time = TimePoint(timestamp=datetime(2024, 1, 1, 13, 0, 0))
        event = Event(
            name="workshop",
            event_type=EventType.DURATIVE,
            start_time=start_time,
            end_time=end_time,
            description="Training workshop"
        )
        
        assert event.event_type == EventType.DURATIVE
        assert event.start_time == start_time
        assert event.end_time == end_time
        assert event.get_duration().to_seconds() == 3600.0
    
    def test_event_validation(self):
        """Test event validation."""
        # Instantaneous event without start time should raise error
        with pytest.raises(Exception):
            Event(name="invalid", event_type=EventType.INSTANTANEOUS)
        
        # Instantaneous event with end time should raise error
        start_time = TimePoint(timestamp=datetime(2024, 1, 1, 12, 0, 0))
        end_time = TimePoint(timestamp=datetime(2024, 1, 1, 13, 0, 0))
        with pytest.raises(Exception):
            Event(
                name="invalid",
                event_type=EventType.INSTANTANEOUS,
                start_time=start_time,
                end_time=end_time
            )


class TestEventSequenceModeling:
    """Test EventSequenceModeling class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.modeling = EventSequenceModeling()
        
        # Create test events
        self.event1 = Event(
            name="event1",
            event_type=EventType.DURATIVE,
            start_time=TimePoint(timestamp=datetime(2024, 1, 1, 9, 0, 0)),
            end_time=TimePoint(timestamp=datetime(2024, 1, 1, 10, 0, 0))
        )
        self.event2 = Event(
            name="event2",
            event_type=EventType.DURATIVE,
            start_time=TimePoint(timestamp=datetime(2024, 1, 1, 10, 30, 0)),
            end_time=TimePoint(timestamp=datetime(2024, 1, 1, 11, 30, 0))
        )
        self.event3 = Event(
            name="event3",
            event_type=EventType.DURATIVE,
            start_time=TimePoint(timestamp=datetime(2024, 1, 1, 12, 0, 0)),
            end_time=TimePoint(timestamp=datetime(2024, 1, 1, 13, 0, 0))
        )
    
    def test_add_event(self):
        """Test adding events to the model."""
        self.modeling.add_event(self.event1)
        self.modeling.add_event(self.event2)
        
        assert len(self.modeling.events) == 2
        assert "event1" in self.modeling.events
        assert "event2" in self.modeling.events
    
    def test_create_sequence(self):
        """Test creating event sequences."""
        self.modeling.add_event(self.event1)
        self.modeling.add_event(self.event2)
        self.modeling.add_event(self.event3)
        
        sequence = self.modeling.create_sequence(["event1", "event2", "event3"])
        
        assert sequence == ["event1", "event2", "event3"]
        assert len(self.modeling.sequences) == 1
    
    def test_analyze_sequence(self):
        """Test sequence analysis."""
        self.modeling.add_event(self.event1)
        self.modeling.add_event(self.event2)
        self.modeling.add_event(self.event3)
        
        sequence = ["event1", "event2", "event3"]
        analysis = self.modeling.analyze_sequence(sequence)
        
        assert "total_duration" in analysis
        assert "gaps" in analysis
        assert "temporal_order" in analysis
        assert "constraints" in analysis
        assert len(analysis["constraints"]) == 2  # 3 events = 2 constraints
    
    def test_find_patterns(self):
        """Test pattern finding."""
        self.modeling.add_event(self.event1)
        self.modeling.add_event(self.event2)
        self.modeling.add_event(self.event3)
        
        # Create multiple sequences with patterns
        self.modeling.create_sequence(["event1", "event2"])
        self.modeling.create_sequence(["event1", "event2", "event3"])
        self.modeling.create_sequence(["event1", "event2"])
        
        patterns = self.modeling.find_patterns(min_occurrences=2)
        
        assert len(patterns) > 0
        # Should find pattern ["event1", "event2"] occurring 3 times
        event1_event2_pattern = next((p for p in patterns if p["pattern"] == ["event1", "event2"]), None)
        assert event1_event2_pattern is not None
        assert event1_event2_pattern["occurrences"] == 3


class TestDurationEstimation:
    """Test DurationEstimation class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.estimation = DurationEstimation()
    
    def test_learn_duration_pattern(self):
        """Test learning duration patterns."""
        context = {"participants": 5, "complexity": "high"}
        actual_duration = Duration(value=120, unit=DurationUnit.MINUTES)
        
        self.estimation.learn_duration_pattern("meeting", context, actual_duration)
        
        assert self.estimation.stats["patterns_learned"] == 1
    
    def test_estimate_duration(self):
        """Test duration estimation."""
        context = {"participants": 5, "complexity": "high"}
        actual_duration = Duration(value=120, unit=DurationUnit.MINUTES)
        
        self.estimation.learn_duration_pattern("meeting", context, actual_duration)
        estimated = self.estimation.estimate_duration("meeting", context)
        
        assert estimated is not None
        assert estimated.value == 2.0  # 120 minutes = 2 hours
        assert estimated.unit == DurationUnit.HOURS
    
    def test_estimate_from_text(self):
        """Test duration extraction from text."""
        text = "The meeting lasted 2 hours and the workshop took 30 minutes"
        estimations = self.estimation.estimate_from_text(text)
        
        assert len(estimations) == 2
        assert any(e["duration"].value == 2 and e["duration"].unit == DurationUnit.HOURS for e in estimations)
        assert any(e["duration"].value == 30 and e["duration"].unit == DurationUnit.MINUTES for e in estimations)


class TestTemporalRelationshipReasoning:
    """Test TemporalRelationshipReasoning class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reasoning = TemporalRelationshipReasoning()
        
        # Create test events
        self.event1 = Event(
            name="event1",
            event_type=EventType.DURATIVE,
            start_time=TimePoint(timestamp=datetime(2024, 1, 1, 9, 0, 0)),
            end_time=TimePoint(timestamp=datetime(2024, 1, 1, 10, 0, 0))
        )
        self.event2 = Event(
            name="event2",
            event_type=EventType.DURATIVE,
            start_time=TimePoint(timestamp=datetime(2024, 1, 1, 10, 30, 0)),
            end_time=TimePoint(timestamp=datetime(2024, 1, 1, 11, 30, 0))
        )
        self.event3 = Event(
            name="event3",
            event_type=EventType.DURATIVE,
            start_time=TimePoint(timestamp=datetime(2024, 1, 1, 10, 0, 0)),
            end_time=TimePoint(timestamp=datetime(2024, 1, 1, 11, 0, 0))
        )
    
    def test_infer_relationship(self):
        """Test temporal relationship inference."""
        relation = self.reasoning.infer_relationship(self.event1, self.event2)
        
        assert relation == TemporalRelation.BEFORE
    
    def test_infer_overlapping_relationship(self):
        """Test overlapping relationship inference."""
        relation = self.reasoning.infer_relationship(self.event1, self.event3)
        
        assert relation == TemporalRelation.MEETS
    
    def test_add_relationship(self):
        """Test adding explicit relationships."""
        self.reasoning.add_relationship("event1", "event2", TemporalRelation.BEFORE)
        
        assert ("event1", "event2") in self.reasoning.relationships
        assert self.reasoning.relationships[("event1", "event2")] == TemporalRelation.BEFORE
    
    def test_check_consistency(self):
        """Test consistency checking."""
        events = {
            "event1": self.event1,
            "event2": self.event2
        }
        
        # Add a relationship
        self.reasoning.add_relationship("event1", "event2", TemporalRelation.BEFORE)
        
        consistency = self.reasoning.check_consistency(events)
        
        assert "inconsistencies" in consistency
        assert "consistent_relationships" in consistency
        assert len(consistency["consistent_relationships"]) == 1


class TestCalendarSupport:
    """Test CalendarSupport class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calendar_support = CalendarSupport()
        
        # Create test events
        self.event1 = Event(
            name="meeting",
            event_type=EventType.DURATIVE,
            start_time=TimePoint(timestamp=datetime(2024, 1, 1, 9, 0, 0)),
            end_time=TimePoint(timestamp=datetime(2024, 1, 1, 10, 0, 0))
        )
        self.event2 = Event(
            name="workshop",
            event_type=EventType.DURATIVE,
            start_time=TimePoint(timestamp=datetime(2024, 1, 1, 10, 30, 0)),
            end_time=TimePoint(timestamp=datetime(2024, 1, 1, 11, 30, 0))
        )
    
    def test_create_calendar(self):
        """Test calendar creation."""
        calendar = self.calendar_support.create_calendar("test_calendar", "UTC")
        
        assert calendar.name == "test_calendar"
        assert calendar.timezone == "UTC"
        assert "test_calendar" in self.calendar_support.calendars
    
    def test_schedule_event(self):
        """Test event scheduling."""
        self.calendar_support.create_calendar("test_calendar")
        result = self.calendar_support.schedule_event("test_calendar", self.event1)
        
        assert result["success"] is True
        assert self.calendar_support.stats["events_scheduled"] == 1
    
    def test_schedule_conflicting_event(self):
        """Test scheduling conflicting events."""
        self.calendar_support.create_calendar("test_calendar")
        
        # Schedule first event
        self.calendar_support.schedule_event("test_calendar", self.event1)
        
        # Create conflicting event
        conflicting_event = Event(
            name="conflict",
            event_type=EventType.DURATIVE,
            start_time=TimePoint(timestamp=datetime(2024, 1, 1, 9, 30, 0)),
            end_time=TimePoint(timestamp=datetime(2024, 1, 1, 10, 30, 0))
        )
        
        result = self.calendar_support.schedule_event("test_calendar", conflicting_event)
        
        assert result["success"] is False
        assert len(result["conflicts"]) > 0
        assert self.calendar_support.stats["conflicts_detected"] == 1
    
    def test_find_free_slots(self):
        """Test finding free time slots."""
        self.calendar_support.create_calendar("test_calendar")
        
        duration = Duration(value=60, unit=DurationUnit.MINUTES)
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 1)
        
        slots = self.calendar_support.find_free_slots("test_calendar", duration, start_date, end_date)
        
        assert len(slots) > 0
        assert all("start_time" in slot for slot in slots)
        assert all("end_time" in slot for slot in slots)


class TestTemporalLogicEngine:
    """Test TemporalLogicEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = TemporalLogicEngine()
        
        # Create test events
        self.event1 = Event(
            name="meeting",
            event_type=EventType.DURATIVE,
            start_time=TimePoint(timestamp=datetime(2024, 1, 1, 9, 0, 0)),
            end_time=TimePoint(timestamp=datetime(2024, 1, 1, 10, 0, 0))
        )
        self.event2 = Event(
            name="workshop",
            event_type=EventType.DURATIVE,
            start_time=TimePoint(timestamp=datetime(2024, 1, 1, 10, 30, 0)),
            end_time=TimePoint(timestamp=datetime(2024, 1, 1, 11, 30, 0))
        )
    
    def test_add_event(self):
        """Test adding events to the engine."""
        self.engine.add_event(self.event1)
        
        assert self.engine.stats["total_events_processed"] == 1
        assert "meeting" in self.engine.event_sequence_modeling.events
    
    def test_analyze_event_sequence(self):
        """Test event sequence analysis."""
        self.engine.add_event(self.event1)
        self.engine.add_event(self.event2)
        
        sequence = ["meeting", "workshop"]
        analysis = self.engine.analyze_event_sequence(sequence)
        
        assert "total_duration" in analysis
        assert "temporal_order" in analysis
        assert len(analysis["constraints"]) == 1
    
    def test_estimate_duration_from_context(self):
        """Test duration estimation from context."""
        context = {"participants": 5, "complexity": "high"}
        actual_duration = Duration(value=120, unit=DurationUnit.MINUTES)

        self.engine.duration_estimation.learn_duration_pattern("meeting", context, actual_duration)
        estimated = self.engine.estimate_duration_from_context("meeting", context)

        assert estimated is not None
        assert estimated.value == 2.0  # 120 minutes = 2 hours
        assert estimated.unit == DurationUnit.HOURS
    
    def test_extract_durations_from_text(self):
        """Test duration extraction from text."""
        text = "The meeting lasted 2 hours"
        estimations = self.engine.extract_durations_from_text(text)
        
        assert len(estimations) == 1
        assert estimations[0]["duration"].value == 2
        assert estimations[0]["duration"].unit == DurationUnit.HOURS
    
    def test_infer_temporal_relationships(self):
        """Test temporal relationship inference."""
        events = [self.event1, self.event2]
        relationships = self.engine.infer_temporal_relationships(events)
        
        assert len(relationships) == 2  # 2 events = 2 relationships (bidirectional)
        assert self.engine.stats["total_relationships_analyzed"] == 2
    
    def test_create_calendar(self):
        """Test calendar creation."""
        calendar = self.engine.create_calendar("test_calendar")
        
        assert calendar.name == "test_calendar"
        assert self.engine.stats["total_calendar_operations"] == 1
    
    def test_schedule_event_in_calendar(self):
        """Test event scheduling in calendar."""
        self.engine.create_calendar("test_calendar")
        result = self.engine.schedule_event_in_calendar("test_calendar", self.event1)
        
        assert result["success"] is True
        assert self.engine.stats["total_calendar_operations"] == 2
    
    def test_get_performance_stats(self):
        """Test performance statistics."""
        stats = self.engine.get_performance_stats()
        
        assert "total_events_processed" in stats
        assert "total_estimations_made" in stats
        assert "total_relationships_analyzed" in stats
        assert "total_calendar_operations" in stats
        assert "event_sequence_stats" in stats
        assert "duration_estimation_stats" in stats
        assert "temporal_relationship_stats" in stats
        assert "calendar_support_stats" in stats
    
    def test_reset_stats(self):
        """Test statistics reset."""
        self.engine.add_event(self.event1)
        self.engine.reset_stats()
        
        assert self.engine.stats["total_events_processed"] == 0
        assert self.engine.event_sequence_modeling.stats["events_processed"] == 0


class TestOptimizedFunctions:
    """Test optimized Numba functions."""
    
    def test_optimized_duration_calculation(self):
        """Test optimized duration calculation."""
        start_times = np.array([1000.0, 2000.0, 3000.0])
        end_times = np.array([1500.0, 2500.0, 3500.0])
        
        durations = _optimized_duration_calculation(start_times, end_times)
        
        assert len(durations) == 3
        assert durations[0] == 500.0
        assert durations[1] == 500.0
        assert durations[2] == 500.0
    
    def test_optimized_temporal_relationship_matrix(self):
        """Test optimized temporal relationship matrix."""
        start_times = np.array([1000.0, 2000.0, 3000.0])
        end_times = np.array([1500.0, 2500.0, 3500.0])
        
        matrix = _optimized_temporal_relationship_matrix(start_times, end_times)
        
        assert matrix.shape == (3, 3)
        assert matrix[0, 1] == 1  # BEFORE
        assert matrix[1, 0] == 2  # AFTER
    
    def test_optimized_working_time_check(self):
        """Test optimized working time check."""
        timestamps = np.array([3600.0, 7200.0, 10800.0])  # 1h, 2h, 3h
        working_hours_start = 9
        working_hours_end = 17

        is_working = _optimized_working_time_check(timestamps, working_hours_start, working_hours_end)

        assert len(is_working) == 3
        assert isinstance(is_working[0], (bool, np.bool_))


class TestCalendar:
    """Test Calendar class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calendar = Calendar(name="test_calendar")
        
        self.event1 = Event(
            name="meeting",
            event_type=EventType.DURATIVE,
            start_time=TimePoint(timestamp=datetime(2024, 1, 1, 9, 0, 0)),
            end_time=TimePoint(timestamp=datetime(2024, 1, 1, 10, 0, 0))
        )
    
    def test_calendar_creation(self):
        """Test calendar creation."""
        assert self.calendar.name == "test_calendar"
        assert self.calendar.working_hours == (9, 17)
        assert 0 in self.calendar.working_days  # Monday
        assert 4 in self.calendar.working_days  # Friday
    
    def test_add_event(self):
        """Test adding events to calendar."""
        self.calendar.add_event(self.event1)
        
        assert "meeting" in self.calendar.events
        assert self.calendar.events["meeting"] == self.event1
    
    def test_remove_event(self):
        """Test removing events from calendar."""
        self.calendar.add_event(self.event1)
        result = self.calendar.remove_event("meeting")
        
        assert result is True
        assert "meeting" not in self.calendar.events
    
    def test_get_events_in_interval(self):
        """Test getting events in time interval."""
        self.calendar.add_event(self.event1)
        
        start = TimePoint(timestamp=datetime(2024, 1, 1, 8, 0, 0))
        end = TimePoint(timestamp=datetime(2024, 1, 1, 11, 0, 0))
        
        events = self.calendar.get_events_in_interval(start, end)
        
        assert len(events) == 1
        assert events[0] == self.event1
    
    def test_is_working_time(self):
        """Test working time check."""
        # Monday 10 AM (working time)
        working_time = TimePoint(timestamp=datetime(2024, 1, 1, 10, 0, 0))
        assert self.calendar.is_working_time(working_time) is True
        
        # Sunday 10 AM (not working time)
        non_working_time = TimePoint(timestamp=datetime(2024, 1, 7, 10, 0, 0))
        assert self.calendar.is_working_time(non_working_time) is False


class TestTemporalConstraint:
    """Test TemporalConstraint class."""
    
    def test_constraint_creation(self):
        """Test temporal constraint creation."""
        constraint = TemporalConstraint(
            event1="event1",
            event2="event2",
            relation=TemporalRelation.BEFORE,
            confidence=0.9
        )
        
        assert constraint.event1 == "event1"
        assert constraint.event2 == "event2"
        assert constraint.relation == TemporalRelation.BEFORE
        assert constraint.confidence == 0.9
        assert str(constraint) == "event1 before event2"


if __name__ == "__main__":
    pytest.main([__file__]) 
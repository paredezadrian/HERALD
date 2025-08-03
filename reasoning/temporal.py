"""
HERALD Temporal Logic System
Temporal Reasoning and Event Processing

This module implements:
- Event Sequence Modeling
- Duration Estimation from context
- Temporal relationship reasoning ("before", "after", "during")
- Calendar support for dates/schedules
- Temporal reasoning capabilities testing
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
from datetime import datetime, timedelta, date
import calendar
import re
import json


class TemporalError(Exception):
    """Raised when temporal operations fail."""
    pass


class EventError(Exception):
    """Raised when event operations fail."""
    pass


class DurationError(Exception):
    """Raised when duration calculations fail."""
    pass


class TemporalRelation(Enum):
    """Types of temporal relationships."""
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    MEETS = "meets"
    MET_BY = "met_by"
    STARTS = "starts"
    STARTED_BY = "started_by"
    FINISHES = "finishes"
    FINISHED_BY = "finished_by"
    EQUALS = "equals"
    SIMULTANEOUS = "simultaneous"


class EventType(Enum):
    """Types of events."""
    INSTANTANEOUS = "instantaneous"
    DURATIVE = "durative"
    PERIODIC = "periodic"
    CONDITIONAL = "conditional"


class DurationUnit(Enum):
    """Units for duration measurement."""
    MILLISECONDS = "ms"
    SECONDS = "s"
    MINUTES = "min"
    HOURS = "h"
    DAYS = "d"
    WEEKS = "w"
    MONTHS = "mo"
    YEARS = "y"


@dataclass
class TimePoint:
    """Represents a specific point in time."""
    timestamp: datetime
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return self.timestamp.isoformat()
    
    def __hash__(self) -> int:
        return hash(self.timestamp)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, TimePoint):
            return False
        return self.timestamp == other.timestamp
    
    def __lt__(self, other) -> bool:
        if not isinstance(other, TimePoint):
            return NotImplemented
        return self.timestamp < other.timestamp


@dataclass
class Duration:
    """Represents a duration of time."""
    value: float
    unit: DurationUnit
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.value}{self.unit.value}"
    
    def to_seconds(self) -> float:
        """Convert duration to seconds."""
        conversion_factors = {
            DurationUnit.MILLISECONDS: 0.001,
            DurationUnit.SECONDS: 1.0,
            DurationUnit.MINUTES: 60.0,
            DurationUnit.HOURS: 3600.0,
            DurationUnit.DAYS: 86400.0,
            DurationUnit.WEEKS: 604800.0,
            DurationUnit.MONTHS: 2592000.0,  # Approximate
            DurationUnit.YEARS: 31536000.0   # Approximate
        }
        return self.value * conversion_factors[self.unit]
    
    def to_timedelta(self) -> timedelta:
        """Convert to timedelta object."""
        seconds = self.to_seconds()
        return timedelta(seconds=seconds)


@dataclass
class TimeInterval:
    """Represents a time interval with start and end points."""
    start: TimePoint
    end: TimePoint
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.start.timestamp > self.end.timestamp:
            raise TemporalError("Start time must be before end time")
    
    def __str__(self) -> str:
        return f"[{self.start} - {self.end}]"
    
    def duration(self) -> Duration:
        """Get the duration of this interval."""
        delta = self.end.timestamp - self.start.timestamp
        return Duration(value=delta.total_seconds(), unit=DurationUnit.SECONDS)
    
    def contains(self, time_point: TimePoint) -> bool:
        """Check if interval contains the given time point."""
        return self.start.timestamp <= time_point.timestamp <= self.end.timestamp
    
    def overlaps(self, other: 'TimeInterval') -> bool:
        """Check if this interval overlaps with another."""
        return (self.start.timestamp <= other.end.timestamp and 
                other.start.timestamp <= self.end.timestamp)


@dataclass
class Event:
    """Represents a temporal event."""
    name: str
    event_type: EventType
    start_time: Optional[TimePoint] = None
    end_time: Optional[TimePoint] = None
    duration: Optional[Duration] = None
    description: str = ""
    participants: List[str] = field(default_factory=list)
    location: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.event_type == EventType.INSTANTANEOUS:
            if self.start_time is None:
                raise EventError("Instantaneous events must have a start time")
            if self.end_time is not None:
                raise EventError("Instantaneous events cannot have an end time")
        elif self.event_type == EventType.DURATIVE:
            if self.start_time is None or self.end_time is None:
                raise EventError("Durative events must have both start and end times")
        elif self.event_type == EventType.PERIODIC:
            if self.duration is None:
                raise EventError("Periodic events must have a duration")
    
    def __str__(self) -> str:
        if self.event_type == EventType.INSTANTANEOUS:
            return f"{self.name} at {self.start_time}"
        elif self.event_type == EventType.DURATIVE:
            return f"{self.name} from {self.start_time} to {self.end_time}"
        else:
            return f"{self.name} ({self.event_type.value})"
    
    def get_duration(self) -> Optional[Duration]:
        """Get the duration of this event."""
        if self.event_type == EventType.INSTANTANEOUS:
            return Duration(value=0, unit=DurationUnit.SECONDS)
        elif self.event_type == EventType.DURATIVE and self.start_time and self.end_time:
            delta = self.end_time.timestamp - self.start_time.timestamp
            return Duration(value=delta.total_seconds(), unit=DurationUnit.SECONDS)
        else:
            return self.duration


@dataclass
class TemporalConstraint:
    """Represents a temporal constraint between events."""
    event1: str
    event2: str
    relation: TemporalRelation
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.event1} {self.relation.value} {self.event2}"


@dataclass
class Calendar:
    """Represents a calendar with events and schedules."""
    name: str
    events: Dict[str, Event] = field(default_factory=dict)
    constraints: List[TemporalConstraint] = field(default_factory=list)
    working_hours: Tuple[int, int] = (9, 17)  # 9 AM to 5 PM
    working_days: Set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4})  # Monday to Friday
    timezone: str = "UTC"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_event(self, event: Event) -> None:
        """Add an event to the calendar."""
        self.events[event.name] = event
    
    def remove_event(self, event_name: str) -> bool:
        """Remove an event from the calendar."""
        if event_name in self.events:
            del self.events[event_name]
            return True
        return False
    
    def add_constraint(self, constraint: TemporalConstraint) -> None:
        """Add a temporal constraint."""
        self.constraints.append(constraint)
    
    def get_events_in_interval(self, start: TimePoint, end: TimePoint) -> List[Event]:
        """Get all events that occur in the given interval."""
        result = []
        for event in self.events.values():
            if event.start_time and event.end_time:
                if event.start_time.timestamp <= end.timestamp and event.end_time.timestamp >= start.timestamp:
                    result.append(event)
            elif event.start_time:
                if start.timestamp <= event.start_time.timestamp <= end.timestamp:
                    result.append(event)
        return result
    
    def is_working_time(self, time_point: TimePoint) -> bool:
        """Check if the given time is within working hours."""
        hour = time_point.timestamp.hour
        weekday = time_point.timestamp.weekday()
        return (weekday in self.working_days and 
                self.working_hours[0] <= hour < self.working_hours[1])


class EventSequenceModeling:
    """Models and analyzes sequences of events."""
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: Dict[str, Event] = {}
        self.sequences: List[List[str]] = []
        self.logger = logging.getLogger(__name__)
        self.stats = {
            'events_processed': 0,
            'sequences_analyzed': 0,
            'constraints_generated': 0
        }
    
    def add_event(self, event: Event) -> None:
        """Add an event to the model."""
        if len(self.events) >= self.max_events:
            raise EventError(f"Maximum number of events ({self.max_events}) exceeded")
        
        self.events[event.name] = event
        self.stats['events_processed'] += 1
    
    def create_sequence(self, event_names: List[str]) -> List[str]:
        """Create a sequence of events."""
        if not all(name in self.events for name in event_names):
            raise EventError("All event names must exist in the model")
        
        self.sequences.append(event_names)
        self.stats['sequences_analyzed'] += 1
        return event_names
    
    def analyze_sequence(self, sequence: List[str]) -> Dict[str, Any]:
        """Analyze a sequence of events for temporal patterns."""
        if not sequence:
            return {'error': 'Empty sequence'}
        
        analysis = {
            'total_duration': Duration(value=0, unit=DurationUnit.SECONDS),
            'gaps': [],
            'overlaps': [],
            'temporal_order': [],
            'constraints': []
        }
        
        events = [self.events[name] for name in sequence if name in self.events]
        if not events:
            return {'error': 'No valid events found'}
        
        # Sort events by start time
        sorted_events = sorted(events, key=lambda e: e.start_time.timestamp if e.start_time else datetime.min)
        
        # Calculate total duration and gaps
        total_duration = Duration(value=0, unit=DurationUnit.SECONDS)
        for i, event in enumerate(sorted_events):
            if event.get_duration():
                total_duration.value += event.get_duration().to_seconds()
            
            if i > 0:
                prev_event = sorted_events[i-1]
                if prev_event.end_time and event.start_time:
                    gap = event.start_time.timestamp - prev_event.end_time.timestamp
                    gap_seconds = gap.total_seconds()
                    if gap_seconds > 0:
                        analysis['gaps'].append({
                            'between': (prev_event.name, event.name),
                            'duration': Duration(value=gap_seconds, unit=DurationUnit.SECONDS)
                        })
                    elif gap_seconds < 0:
                        analysis['overlaps'].append({
                            'between': (prev_event.name, event.name),
                            'duration': Duration(value=abs(gap_seconds), unit=DurationUnit.SECONDS)
                        })
        
        analysis['total_duration'] = total_duration
        analysis['temporal_order'] = [event.name for event in sorted_events]
        
        # Generate temporal constraints
        for i in range(len(sorted_events) - 1):
            constraint = TemporalConstraint(
                event1=sorted_events[i].name,
                event2=sorted_events[i+1].name,
                relation=TemporalRelation.BEFORE
            )
            analysis['constraints'].append(constraint)
            self.stats['constraints_generated'] += 1
        
        return analysis
    
    def find_patterns(self, min_occurrences: int = 2) -> List[Dict[str, Any]]:
        """Find recurring patterns in event sequences."""
        patterns = []
        
        # Count subsequences
        subsequence_counts = defaultdict(int)
        for sequence in self.sequences:
            for length in range(2, min(len(sequence) + 1, 6)):  # Look for patterns of length 2-5
                for i in range(len(sequence) - length + 1):
                    subsequence = tuple(sequence[i:i+length])
                    subsequence_counts[subsequence] += 1
        
        # Filter patterns by minimum occurrences
        for subsequence, count in subsequence_counts.items():
            if count >= min_occurrences:
                patterns.append({
                    'pattern': list(subsequence),
                    'occurrences': count,
                    'events': [self.events[name] for name in subsequence if name in self.events]
                })
        
        return sorted(patterns, key=lambda p: p['occurrences'], reverse=True)
    
    def predict_next_event(self, current_sequence: List[str], max_predictions: int = 5) -> List[Dict[str, Any]]:
        """Predict the next event based on current sequence."""
        predictions = []
        
        # Find similar sequences
        for sequence in self.sequences:
            if len(sequence) > len(current_sequence):
                # Check if current sequence is a prefix of this sequence
                if sequence[:len(current_sequence)] == current_sequence:
                    next_event_name = sequence[len(current_sequence)]
                    if next_event_name in self.events:
                        predictions.append({
                            'event': self.events[next_event_name],
                            'confidence': 1.0,
                            'based_on': sequence
                        })
        
        # Sort by confidence and return top predictions
        predictions.sort(key=lambda p: p['confidence'], reverse=True)
        return predictions[:max_predictions]


class DurationEstimation:
    """Estimates durations from context and patterns."""
    
    def __init__(self):
        self.duration_patterns: Dict[str, List[Duration]] = defaultdict(list)
        self.context_patterns: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        self.stats = {
            'estimations_made': 0,
            'patterns_learned': 0,
            'accuracy_score': 0.0
        }
    
    def learn_duration_pattern(self, event_type: str, context: Dict[str, Any], 
                             actual_duration: Duration) -> None:
        """Learn a duration pattern from context."""
        pattern_key = f"{event_type}_{hash(frozenset(context.items()))}"
        self.duration_patterns[pattern_key].append(actual_duration)
        self.context_patterns[pattern_key] = context
        self.stats['patterns_learned'] += 1
    
    def estimate_duration(self, event_type: str, context: Dict[str, Any]) -> Optional[Duration]:
        """Estimate duration based on learned patterns."""
        pattern_key = f"{event_type}_{hash(frozenset(context.items()))}"
        
        if pattern_key in self.duration_patterns:
            durations = self.duration_patterns[pattern_key]
            if durations:
                # Calculate average duration
                total_seconds = sum(d.to_seconds() for d in durations)
                avg_seconds = total_seconds / len(durations)
                
                # Choose appropriate unit
                if avg_seconds < 60:
                    unit = DurationUnit.SECONDS
                elif avg_seconds < 3600:
                    unit = DurationUnit.MINUTES
                    avg_seconds /= 60
                elif avg_seconds < 86400:
                    unit = DurationUnit.HOURS
                    avg_seconds /= 3600
                else:
                    unit = DurationUnit.DAYS
                    avg_seconds /= 86400
                
                self.stats['estimations_made'] += 1
                return Duration(value=avg_seconds, unit=unit)
        
        return None
    
    def estimate_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Estimate durations from natural language text."""
        estimations = []
        
        # Common duration patterns
        duration_patterns = [
            (r'(\d+)\s*(?:milliseconds?|ms)', DurationUnit.MILLISECONDS),
            (r'(\d+)\s*(?:seconds?|s)', DurationUnit.SECONDS),
            (r'(\d+)\s*(?:minutes?|min)', DurationUnit.MINUTES),
            (r'(\d+)\s*(?:hours?|h)', DurationUnit.HOURS),
            (r'(\d+)\s*(?:days?|d)', DurationUnit.DAYS),
            (r'(\d+)\s*(?:weeks?|w)', DurationUnit.WEEKS),
            (r'(\d+)\s*(?:months?|mo)', DurationUnit.MONTHS),
            (r'(\d+)\s*(?:years?|y)', DurationUnit.YEARS)
        ]
        
        for pattern, unit in duration_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = float(match.group(1))
                duration = Duration(value=value, unit=unit)
                estimations.append({
                    'duration': duration,
                    'text': match.group(0),
                    'position': match.span(),
                    'confidence': 0.9
                })
        
        # Relative duration patterns
        relative_patterns = [
            (r'(?:a few|several)\s+(?:minutes?|min)', Duration(value=5, unit=DurationUnit.MINUTES)),
            (r'(?:a few|several)\s+(?:hours?|h)', Duration(value=3, unit=DurationUnit.HOURS)),
            (r'(?:a few|several)\s+(?:days?|d)', Duration(value=3, unit=DurationUnit.DAYS)),
            (r'(?:a couple of)\s+(?:hours?|h)', Duration(value=2, unit=DurationUnit.HOURS)),
            (r'(?:a couple of)\s+(?:days?|d)', Duration(value=2, unit=DurationUnit.DAYS))
        ]
        
        for pattern, duration in relative_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                estimations.append({
                    'duration': duration,
                    'text': pattern,
                    'confidence': 0.7
                })
        
        return estimations
    
    def update_accuracy(self, estimated: Duration, actual: Duration) -> None:
        """Update accuracy score based on estimation vs actual."""
        estimated_seconds = estimated.to_seconds()
        actual_seconds = actual.to_seconds()
        
        if actual_seconds > 0:
            error = abs(estimated_seconds - actual_seconds) / actual_seconds
            self.stats['accuracy_score'] = (self.stats['accuracy_score'] * 0.9 + (1 - error) * 0.1)


class TemporalRelationshipReasoning:
    """Reasons about temporal relationships between events."""
    
    def __init__(self):
        self.relationships: Dict[Tuple[str, str], TemporalRelation] = {}
        self.logger = logging.getLogger(__name__)
        self.stats = {
            'relationships_analyzed': 0,
            'inferences_made': 0,
            'consistency_checks': 0
        }
    
    def add_relationship(self, event1: str, event2: str, relation: TemporalRelation) -> None:
        """Add a temporal relationship between two events."""
        self.relationships[(event1, event2)] = relation
        self.stats['relationships_analyzed'] += 1
    
    def infer_relationship(self, event1: Event, event2: Event) -> Optional[TemporalRelation]:
        """Infer the temporal relationship between two events."""
        if not event1.start_time or not event2.start_time:
            return None
        
        start1 = event1.start_time.timestamp
        start2 = event2.start_time.timestamp
        end1 = event1.end_time.timestamp if event1.end_time else start1
        end2 = event2.end_time.timestamp if event2.end_time else start2
        
        # Determine relationship based on time intervals
        if start1 < start2 and end1 < start2:
            return TemporalRelation.BEFORE
        elif start2 < start1 and end2 < start1:
            return TemporalRelation.AFTER
        elif start1 == start2 and end1 == end2:
            return TemporalRelation.EQUALS
        elif start1 == start2:
            return TemporalRelation.STARTS
        elif end1 == end2:
            return TemporalRelation.FINISHES
        elif start1 < start2 < end1:
            return TemporalRelation.OVERLAPS
        elif start2 < start1 < end2:
            return TemporalRelation.OVERLAPS
        elif end1 == start2:
            return TemporalRelation.MEETS
        elif end2 == start1:
            return TemporalRelation.MET_BY
        elif start1 == start2 or end1 == end2:
            return TemporalRelation.OVERLAPS
        else:
            return TemporalRelation.SIMULTANEOUS
    
    def check_consistency(self, events: Dict[str, Event]) -> Dict[str, Any]:
        """Check consistency of temporal relationships."""
        inconsistencies = []
        consistent_relationships = []
        
        for (event1_name, event2_name), relation in self.relationships.items():
            if event1_name in events and event2_name in events:
                event1 = events[event1_name]
                event2 = events[event2_name]
                
                inferred_relation = self.infer_relationship(event1, event2)
                if inferred_relation and inferred_relation != relation:
                    inconsistencies.append({
                        'events': (event1_name, event2_name),
                        'declared': relation,
                        'inferred': inferred_relation
                    })
                else:
                    consistent_relationships.append({
                        'events': (event1_name, event2_name),
                        'relation': relation
                    })
        
        self.stats['consistency_checks'] += 1
        return {
            'inconsistencies': inconsistencies,
            'consistent_relationships': consistent_relationships,
            'total_relationships': len(self.relationships)
        }
    
    def transitive_closure(self) -> Dict[Tuple[str, str], TemporalRelation]:
        """Compute transitive closure of temporal relationships."""
        closure = self.relationships.copy()
        
        # Floyd-Warshall algorithm for transitive closure
        events = set()
        for (e1, e2) in self.relationships.keys():
            events.add(e1)
            events.add(e2)
        
        for event1 in events:
            for event2 in events:
                for event3 in events:
                    if (event1, event2) in closure and (event2, event3) in closure:
                        relation1 = closure[(event1, event2)]
                        relation2 = closure[(event2, event3)]
                        
                        # Apply transitive rules
                        transitive_relation = self._apply_transitive_rule(relation1, relation2)
                        if transitive_relation:
                            closure[(event1, event3)] = transitive_relation
                            self.stats['inferences_made'] += 1
        
        return closure
    
    def _apply_transitive_rule(self, relation1: TemporalRelation, relation2: TemporalRelation) -> Optional[TemporalRelation]:
        """Apply transitive rules for temporal relations."""
        # Allen's interval algebra rules
        rules = {
            (TemporalRelation.BEFORE, TemporalRelation.BEFORE): TemporalRelation.BEFORE,
            (TemporalRelation.BEFORE, TemporalRelation.MEETS): TemporalRelation.BEFORE,
            (TemporalRelation.MEETS, TemporalRelation.BEFORE): TemporalRelation.BEFORE,
            (TemporalRelation.OVERLAPS, TemporalRelation.BEFORE): TemporalRelation.BEFORE,
            (TemporalRelation.STARTS, TemporalRelation.BEFORE): TemporalRelation.BEFORE,
            (TemporalRelation.FINISHES, TemporalRelation.BEFORE): TemporalRelation.BEFORE,
            (TemporalRelation.EQUALS, TemporalRelation.BEFORE): TemporalRelation.BEFORE,
            # Add more rules as needed
        }
        
        return rules.get((relation1, relation2))


class CalendarSupport:
    """Provides calendar functionality for temporal reasoning."""
    
    def __init__(self):
        self.calendars: Dict[str, Calendar] = {}
        self.logger = logging.getLogger(__name__)
        self.stats = {
            'calendars_created': 0,
            'events_scheduled': 0,
            'conflicts_detected': 0
        }
    
    def create_calendar(self, name: str, timezone: str = "UTC") -> Calendar:
        """Create a new calendar."""
        calendar = Calendar(name=name, timezone=timezone)
        self.calendars[name] = calendar
        self.stats['calendars_created'] += 1
        return calendar
    
    def schedule_event(self, calendar_name: str, event: Event) -> Dict[str, Any]:
        """Schedule an event in a calendar."""
        if calendar_name not in self.calendars:
            raise TemporalError(f"Calendar '{calendar_name}' not found")
        
        calendar = self.calendars[calendar_name]
        conflicts = []
        
        # Check for conflicts with existing events
        if event.start_time and event.end_time:
            existing_events = calendar.get_events_in_interval(event.start_time, event.end_time)
            for existing_event in existing_events:
                if existing_event.name != event.name:
                    conflicts.append({
                        'conflicting_event': existing_event,
                        'overlap_type': 'full' if event.start_time.timestamp <= existing_event.start_time.timestamp and event.end_time.timestamp >= existing_event.end_time.timestamp else 'partial'
                    })
        
        if conflicts:
            self.stats['conflicts_detected'] += 1
            return {
                'success': False,
                'conflicts': conflicts,
                'event': event
            }
        else:
            calendar.add_event(event)
            self.stats['events_scheduled'] += 1
            return {
                'success': True,
                'event': event
            }
    
    def find_free_slots(self, calendar_name: str, duration: Duration, 
                       start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Find free time slots in a calendar."""
        if calendar_name not in self.calendars:
            raise TemporalError(f"Calendar '{calendar_name}' not found")
        
        calendar = self.calendars[calendar_name]
        free_slots = []
        
        current_date = start_date
        while current_date <= end_date:
            # Check each hour of the day
            for hour in range(calendar.working_hours[0], calendar.working_hours[1]):
                start_time = TimePoint(timestamp=datetime.combine(current_date, datetime.min.time().replace(hour=hour)))
                end_time = TimePoint(timestamp=start_time.timestamp + duration.to_timedelta())
                
                # Check if this slot is free
                conflicting_events = calendar.get_events_in_interval(start_time, end_time)
                if not conflicting_events:
                    free_slots.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'date': current_date
                    })
            
            current_date += timedelta(days=1)
        
        return free_slots
    
    def export_calendar(self, calendar_name: str) -> Dict[str, Any]:
        """Export calendar data."""
        if calendar_name not in self.calendars:
            raise TemporalError(f"Calendar '{calendar_name}' not found")
        
        calendar = self.calendars[calendar_name]
        return {
            'name': calendar.name,
            'timezone': calendar.timezone,
            'working_hours': calendar.working_hours,
            'working_days': list(calendar.working_days),
            'events': {
                name: {
                    'name': event.name,
                    'event_type': event.event_type.value,
                    'start_time': event.start_time.timestamp.isoformat() if event.start_time else None,
                    'end_time': event.end_time.timestamp.isoformat() if event.end_time else None,
                    'duration': str(event.duration) if event.duration else None,
                    'description': event.description,
                    'participants': event.participants,
                    'location': event.location,
                    'conditions': event.conditions
                }
                for name, event in calendar.events.items()
            },
            'constraints': [
                {
                    'event1': constraint.event1,
                    'event2': constraint.event2,
                    'relation': constraint.relation.value,
                    'confidence': constraint.confidence
                }
                for constraint in calendar.constraints
            ]
        }


class TemporalLogicEngine:
    """Main temporal logic engine that coordinates all temporal reasoning."""
    
    def __init__(self, max_events: int = 10000):
        self.event_sequence_modeling = EventSequenceModeling(max_events)
        self.duration_estimation = DurationEstimation()
        self.temporal_relationship_reasoning = TemporalRelationshipReasoning()
        self.calendar_support = CalendarSupport()
        self.logger = logging.getLogger(__name__)
        self.stats = {
            'total_events_processed': 0,
            'total_estimations_made': 0,
            'total_relationships_analyzed': 0,
            'total_calendar_operations': 0
        }
    
    def add_event(self, event: Event) -> None:
        """Add an event to the temporal logic engine."""
        self.event_sequence_modeling.add_event(event)
        self.stats['total_events_processed'] += 1
    
    def analyze_event_sequence(self, sequence: List[str]) -> Dict[str, Any]:
        """Analyze a sequence of events."""
        return self.event_sequence_modeling.analyze_sequence(sequence)
    
    def estimate_duration_from_context(self, event_type: str, context: Dict[str, Any]) -> Optional[Duration]:
        """Estimate duration from context."""
        estimation = self.duration_estimation.estimate_duration(event_type, context)
        if estimation:
            self.stats['total_estimations_made'] += 1
        return estimation
    
    def extract_durations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract duration information from text."""
        return self.duration_estimation.estimate_from_text(text)
    
    def infer_temporal_relationships(self, events: List[Event]) -> List[Dict[str, Any]]:
        """Infer temporal relationships between events."""
        relationships = []
        
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events):
                if i != j:
                    relation = self.temporal_relationship_reasoning.infer_relationship(event1, event2)
                    if relation:
                        relationships.append({
                            'event1': event1.name,
                            'event2': event2.name,
                            'relation': relation,
                            'confidence': 0.8
                        })
                        self.stats['total_relationships_analyzed'] += 1
        
        return relationships
    
    def create_calendar(self, name: str, timezone: str = "UTC") -> Calendar:
        """Create a new calendar."""
        calendar = self.calendar_support.create_calendar(name, timezone)
        self.stats['total_calendar_operations'] += 1
        return calendar
    
    def schedule_event_in_calendar(self, calendar_name: str, event: Event) -> Dict[str, Any]:
        """Schedule an event in a calendar."""
        result = self.calendar_support.schedule_event(calendar_name, event)
        self.stats['total_calendar_operations'] += 1
        return result
    
    def find_free_time_slots(self, calendar_name: str, duration: Duration, 
                           start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Find free time slots in a calendar."""
        slots = self.calendar_support.find_free_slots(calendar_name, duration, start_date, end_date)
        self.stats['total_calendar_operations'] += 1
        return slots
    
    def check_temporal_consistency(self, events: Dict[str, Event]) -> Dict[str, Any]:
        """Check consistency of temporal relationships."""
        return self.temporal_relationship_reasoning.check_consistency(events)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'total_events_processed': self.stats['total_events_processed'],
            'total_estimations_made': self.stats['total_estimations_made'],
            'total_relationships_analyzed': self.stats['total_relationships_analyzed'],
            'total_calendar_operations': self.stats['total_calendar_operations'],
            'event_sequence_stats': self.event_sequence_modeling.stats,
            'duration_estimation_stats': self.duration_estimation.stats,
            'temporal_relationship_stats': self.temporal_relationship_reasoning.stats,
            'calendar_support_stats': self.calendar_support.stats
        }
    
    def reset_stats(self) -> None:
        """Reset all performance statistics."""
        self.stats = {
            'total_events_processed': 0,
            'total_estimations_made': 0,
            'total_relationships_analyzed': 0,
            'total_calendar_operations': 0
        }
        self.event_sequence_modeling.stats = {
            'events_processed': 0,
            'sequences_analyzed': 0,
            'constraints_generated': 0
        }
        self.duration_estimation.stats = {
            'estimations_made': 0,
            'patterns_learned': 0,
            'accuracy_score': 0.0
        }
        self.temporal_relationship_reasoning.stats = {
            'relationships_analyzed': 0,
            'inferences_made': 0,
            'consistency_checks': 0
        }
        self.calendar_support.stats = {
            'calendars_created': 0,
            'events_scheduled': 0,
            'conflicts_detected': 0
        }


# Optimized functions using Numba
@jit(nopython=True)
def _optimized_duration_calculation(start_times: np.ndarray, end_times: np.ndarray) -> np.ndarray:
    """Optimized duration calculation using Numba."""
    durations = np.zeros(len(start_times))
    for i in prange(len(start_times)):
        if start_times[i] > 0 and end_times[i] > 0:
            durations[i] = end_times[i] - start_times[i]
    return durations


@jit(nopython=True, parallel=True)
def _optimized_temporal_relationship_matrix(start_times: np.ndarray, end_times: np.ndarray) -> np.ndarray:
    """Optimized temporal relationship matrix calculation."""
    n = len(start_times)
    relationship_matrix = np.zeros((n, n), dtype=np.int32)
    
    for i in prange(n):
        for j in prange(n):
            if i != j:
                start_i, end_i = start_times[i], end_times[i]
                start_j, end_j = start_times[j], end_times[j]
                
                if start_i < start_j and end_i < start_j:
                    relationship_matrix[i, j] = 1  # BEFORE
                elif start_j < start_i and end_j < start_i:
                    relationship_matrix[i, j] = 2  # AFTER
                elif start_i == start_j and end_i == end_j:
                    relationship_matrix[i, j] = 3  # EQUALS
                elif start_i < start_j < end_i:
                    relationship_matrix[i, j] = 4  # OVERLAPS
                else:
                    relationship_matrix[i, j] = 5  # SIMULTANEOUS
    
    return relationship_matrix


@jit(nopython=True)
def _optimized_working_time_check(timestamps: np.ndarray, working_hours_start: int, 
                                working_hours_end: int) -> np.ndarray:
    """Optimized working time check."""
    is_working_time = np.zeros(len(timestamps), dtype=np.bool_)
    
    for i in prange(len(timestamps)):
        # Extract hour from timestamp (assuming timestamp is in seconds since epoch)
        # This is a simplified version - in practice, you'd need proper datetime conversion
        hour = int((timestamps[i] % 86400) // 3600)
        is_working_time[i] = working_hours_start <= hour < working_hours_end
    
    return is_working_time.astype(np.bool_) 
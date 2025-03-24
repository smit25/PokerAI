import time

class TimeManager:
    """
    Advanced time management for poker tournaments.
    Optimizes time allocation across a match using dynamic programming concepts.
    """
    
    def __init__(self):
        # Time tracking
        self.decision_start_time = None
        self.total_time_used = 0
        self.decision_times = []
        self.street_decision_times = {
            0: [],  # Preflop
            1: [],  # Flop
            2: [],  # Turn
            3: []   # River
        }
        
        # Tournament settings
        self.tournament_phase = 1
        self.total_time_bank = 500  # 500 seconds default
        
        # Dynamic time allocation
        self.time_allocation = {
            'preflop': 0.2,
            'flop': 0.3,
            'turn': 0.25,
            'river': 0.25
        }
        
        # Importance multipliers
        self.importance_multiplier = {
            'low': 0.5,
            'normal': 1.0,
            'high': 2.0,
            'critical': 3.0
        }
        
        # Hand tracking
        self.hands_played = 0
        self.total_hands = 1000
        self.hands_remaining_prediction = 1000
        
        # Reserved time
        self.reserved_time_fraction = 0.15  # Reduced from 0.2
        self.reserved_time = self.total_time_bank * self.reserved_time_fraction
        
        # Average time tracking
        self.avg_time_per_hand = 0.9 * self.total_time_bank / self.total_hands
    
    def start_decision(self):
        """Mark the start time of a decision."""
        self.decision_start_time = time.time()
    
    def end_decision(self):
        """
        Mark the end of a decision and update time tracking.
        
        Returns:
            Elapsed time for this decision
        """
        if self.decision_start_time is None:
            return 0
            
        elapsed = time.time() - self.decision_start_time
        self.total_time_used += elapsed
        self.decision_times.append(elapsed)
        
        # Keep only recent decisions
        if len(self.decision_times) > 100:
            self.decision_times.pop(0)
        
        return elapsed
    
    def get_time_remaining(self):
        """
        Calculate remaining time in the bank.
        """
        return max(0, self.total_time_bank - self.total_time_used)
    
    def is_time_critical(self):
        """
        Check if we're running critically low on time.
        
        Returns:
            Boolean indicating if time usage should be restricted
        """
        time_remaining = self.get_time_remaining()
        hands_remaining = max(1, self.total_hands - self.hands_played)
        
        # If we have less than 10% of our optimal time remaining
        return time_remaining < 0.1 * self.avg_time_per_hand * hands_remaining
    
    def emergency_time_restriction(self):
        """
        Get an emergency time restriction when nearly out of time.
        
        Returns:
            Maximum allowed time for next decision
        """
        time_remaining = self.get_time_remaining()
        hands_remaining = max(1, self.total_hands - self.hands_played)
        
        # Ultra conservative time allocation
        return max(0.001, time_remaining / (2 * hands_remaining))


# class TimeManager:
#     """
#     Advanced time management system using dynamic programming concepts.
#     Optimizes time allocation across a tournament based on expected value of decisions.
#     """
    
#     def __init__(self):
#         # Time tracking with exponential moving average
#         self.decision_start_time = None
#         self.total_time_used = 0
#         self.decision_times = []
#         self.street_decision_times = {
#             0: [],  # Preflop
#             1: [],  # Flop
#             2: [],  # Turn
#             3: []   # River
#         }
        
#         self.phase_times = {1:500, 2:1000, 3:1500}
#         self.tournament_phase = 1  # Default to Phase 1
#         self.total_time_bank = self.phase_times[self.tournament_phase]  # Default to Phase 1 (500 seconds)
        
#         # Dynamic time allocation using value-based approach
#         self.time_allocation = {
#             'preflop': 0.2,  # Base allocations
#             'flop': 0.3,
#             'turn': 0.25,
#             'river': 0.25
#         }
        
#         # Time allocation adjustment factors
#         self.importance_multiplier = {
#             'low': 0.5,
#             'normal': 1.0,
#             'high': 2.0,
#             'critical': 3.0
#         }
        
#         # Hand tracking with prediction
#         self.hands_played = 0
#         self.total_hands = 1000
#         self.hands_remaining_prediction = 1000
        
#         # Learning rate for adaptation
#         self.learning_rate = 0.1
        
#         # Reserved time for critical decisions
#         self.reserved_time_fraction = 0.2
#         self.reserved_time = 0
        
#         # Decision complexity tracking
#         self.complexity_history = []

#         self.avg_time_per_hand = 0.9*self.total_time_bank/self.total_hands
    
#     def start_decision(self):
#         """Mark the start time of a decision."""
#         self.decision_start_time = time.time()
    
#     def end_decision(self):
#         """
#         Mark the end of a decision and update time tracking.
        
#         Returns:
#             Elapsed time for this decision
#         """
#         if self.decision_start_time is None:
#             return 0
            
#         elapsed = time.time() - self.decision_start_time
#         self.total_time_used += elapsed
#         self.decision_times.append(elapsed)
        
#         # Keep only the last 100 decisions to track recent performance
#         if len(self.decision_times) > 100:
#             self.decision_times.pop(0)
        
#         return elapsed
    
    
#     def get_time_remaining(self):
#         """
#         Calculate remaining time in the bank.
#         """
#         return max(0, self.total_time_bank - self.total_time_used)
    
#     def get_target_decision_time(self, street, hand_importance='normal'):
#         """
#         Calculate the optimal time allocation for a decision using dynamic programming concepts.
#         Balances current decision value with expected future decision value.
        
#         Args:
#             street: Current street (0-3)
#             hand_importance: 'low', 'normal', 'high', or 'critical'
            
#         Returns:
#             Target time in seconds for this decision
#         """
#         # If in emergency mode, use ultra-conservative allocation
#         if self.is_time_critical():
#             return self.emergency_time_restriction()
        
#         # Map street to allocation key
#         street_key = ['preflop', 'flop', 'turn', 'river'][min(street, 3)]
        
#         # Base allocation for this street
#         base_allocation = self.time_allocation[street_key]
        
#         # Adjust for hand importance
#         importance_mult = self.importance_multiplier.get(hand_importance, 1.0)
        
#         # Calculate hands remaining (with uncertainty handling)
#         hands_remaining = max(1, self.total_hands - self.hands_played)
#         predicted_remaining = max(hands_remaining, self.hands_remaining_prediction)
        
#         # Calculate baseline time per hand
#         available_time = self.get_time_remaining() - self.reserved_time
#         baseline_time = available_time / predicted_remaining
        
#         # Calculate time for this decision
#         decision_time = baseline_time * base_allocation * importance_mult
        
#         # Apply adaptive caps based on tournament stage
#         stage_factor = self._calculate_stage_factor()
        
#         # Early game: more exploration
#         if self.hands_played < 100:
#             max_allowed = baseline_time * 1.5 * stage_factor
#         # Mid game: balanced
#         elif self.hands_played < 800:
#             max_allowed = baseline_time * 1.0 * stage_factor
#         # Late game: more critical decisions
#         else:
#             max_allowed = baseline_time * 2.0 * stage_factor
        
#         # Ensure we don't use too much time on a single decision
#         capped_time = min(decision_time, max_allowed)
        
#         # Add street-specific minimum time
#         # More complex streets get higher minimum time
#         min_time = 0.01 * (street + 1)
        
#         return max(min_time, capped_time)
    
#     def _calculate_stage_factor(self):
#         """
#         Calculate stage factor based on tournament progress.
#         Allows more time variance in later stages.
        
#         Returns:
#             Stage factor multiplier
#         """
#         progress = self.hands_played / self.total_hands
        
#         # More variance allowed as tournament progresses
#         if progress < 0.3:
#             return 1.0  # Conservative early
#         elif progress < 0.7:
#             return 1.5  # More flexible mid-game
#         else:
#             return 2.0  # Most flexible late-game
    
    
#     def is_time_critical(self):
#         """
#         Check if we're running critically low on time.
        
#         Returns:
#             Boolean indicating if time usage should be restricted
#         """
#         time_remaining = self.get_time_remaining()
#         hands_remaining = max(1, self.total_hands - self.hands_played)
        
#         # If we have less than 10% of our optimal time remaining
#         return time_remaining < 0.1 * self.avg_time_per_hand * hands_remaining
    
#     def emergency_time_restriction(self):
#         """
#         Get an emergency time restriction when nearly out of time.
        
#         Returns:
#             Maximum allowed time for next decision
#         """
#         time_remaining = self.get_time_remaining()
#         hands_remaining = max(1, self.total_hands - self.hands_played)
        
#         # Ultra conservative time allocation
#         return max(0.001, time_remaining / (2 * hands_remaining))
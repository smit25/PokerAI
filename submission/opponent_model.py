
from gym_env import PokerEnv
from collections import defaultdict

action_types = PokerEnv.ActionType

class OpponentModel:
    """
    Advanced opponent modeling for poker with 27-card deck.
    Uses Bayesian inference and pattern recognition to exploit opponent tendencies.
    """
    
    def __init__(self):
        # Core tracking statistics with optimized priors for this variant
        self.aggression = 0.5  # Scale 0-1, 0.5 is baseline
        self.fold_frequency = 0.5  # How often they fold to bets
        self.bluff_frequency = 0.3  # Estimated bluffing frequency
        
        # Redraw statistics
        self.redraw_count = 0
        self.hands_seen = 0
        self.redraw_history = []  # List of (discarded_card, drawn_card, street)
        
        # Bayesian confidence tracking
        self.aggression_confidence = 1.0
        self.fold_confidence = 1.0
        self.bluff_confidence = 1.0
        
        # Street-specific patterns with recency bias
        self.actions_by_street = {
            0: [],  # Preflop
            1: [],  # Flop
            2: [],  # Turn
            3: []   # River
        }
        
        # Advanced pattern detection
        self.continuation_bet_freq = 0.5  # How often they c-bet
        self.check_raise_freq = 0.1  # How often they check-raise
        self.bet_sizing_by_street = {
            0: [],  # Preflop bet sizes
            1: [],  # Flop bet sizes
            2: [],  # Turn bet sizes
            3: []   # River bet sizes
        }
        
        # Showdown tracking
        self.showdown_history = []
        self.bluffs_detected = 0
        self.value_bets_detected = 0
        
        # Pattern recognition
        self.action_sequences = {}
        
        # Learning rate with decay
        self.base_learning_rate = 0.2
        self.learning_rate = 0.2
        
        # Positional tendencies
        self.position_aggression = {
            'SB': 0.5,
            'BB': 0.5
        }
        
        # Redraw tendencies - specialized for this variant
        self.redraw_patterns = {
            'high_card_discard': 0,
            'low_card_discard': 0,
            'suited_preference': 0,
            'ace_behavior': 'unknown'  # 'keeps', 'discards', or 'unknown'
        }
    
    def update_from_action(self, action, street, bet_size=None, was_all_in=False):
        """
        Update model based on opponent's action using Bayesian updating
        """
        # Record action with recency bias
        self.actions_by_street[street].append(action)
        
        # Keep only recent actions
        max_actions = 50
        if len(self.actions_by_street[street]) > max_actions:
            self.actions_by_street[street] = self.actions_by_street[street][-max_actions:]
        
        # Update aggression metric
        if action == action_types.RAISE.value:
            # Bayesian update
            prior = self.aggression
            likelihood = 0.9  # High likelihood of aggressive player raising
            
            posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
            
            # Apply with learning rate and recency bias
            self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * posterior
            
            # Update position-specific aggression
            position = 'SB' if street % 2 == 0 else 'BB'  # Rough position estimation
            self.position_aggression[position] = (
                0.9 * self.position_aggression[position] + 0.1 * posterior
            )
            
            # Track bet sizing
            if bet_size is not None:
                self.bet_sizing_by_street[street].append(bet_size)
                # Keep only recent bet sizes
                if len(self.bet_sizing_by_street[street]) > max_actions:
                    self.bet_sizing_by_street[street] = self.bet_sizing_by_street[street][-max_actions:]
        
        elif action == action_types.CALL.value:
            self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * 0.5
        
        elif action == action_types.CHECK.value:
            self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * 0.3
        
        elif action == action_types.FOLD.value:
            self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * 0.2
            
            # Update fold frequency
            prior = self.fold_frequency
            likelihood = 0.9
            posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
            self.fold_frequency = (1 - self.learning_rate) * self.fold_frequency + self.learning_rate * posterior
        
        # Update confidence
        self.aggression_confidence = min(10.0, self.aggression_confidence + 0.05)
        self.fold_confidence = min(10.0, self.fold_confidence + 0.05)
        
        # Track action sequences for pattern detection
        all_actions = []
        for s in range(4):
            all_actions.extend(self.actions_by_street[s][-3:])
        
        if len(all_actions) >= 3:
            seq = tuple(all_actions[-3:])
            self.action_sequences[seq] = self.action_sequences.get(seq, 0) + 1
    
    def update_from_redraw(self, discarded_card, drawn_card, street):
        """
        Update model based on opponent's redraw action
        """
        self.redraw_count += 1
        self.redraw_history.append((discarded_card, drawn_card, street))
        
        # Advanced analysis of discarded cards
        discarded_rank = discarded_card // 3
        discarded_suit = discarded_card % 3
        
        # Track tendencies
        if discarded_rank >= 6:  # 8, 9, A
            self.redraw_patterns['high_card_discard'] += 1
            
            # Track Ace discard behavior specifically
            if discarded_rank == 8:  # Ace
                self.redraw_patterns['ace_behavior'] = 'discards'
        else:
            self.redraw_patterns['low_card_discard'] += 1
        
        # Look for suited preferences
        if len(self.redraw_history) >= 2:
            # Check if they consistently keep cards of a particular suit
            kept_suits = []
            for discard, _, _ in self.redraw_history:
                # For each discard, find the kept card's suit
                kept_suit = None
                for suit in range(3):
                    for rank in range(9):
                        card = rank * 3 + suit
                        if card != discard and card != drawn_card:
                            kept_suit = suit
                            break
                if kept_suit is not None:
                    kept_suits.append(kept_suit)
            
            # Check if there's a pattern in kept suits
            if kept_suits:
                suit_counts = {}
                for s in kept_suits:
                    suit_counts[s] = suit_counts.get(s, 0) + 1
                
                # If one suit dominates their keeps
                max_suit = max(suit_counts, key=suit_counts.get)
                if suit_counts[max_suit] / len(kept_suits) > 0.7:
                    self.redraw_patterns['suited_preference'] = max_suit
        
        # Update redraw frequency model
        if hasattr(self, 'redraw_frequency_by_street'):
            self.redraw_frequency_by_street[street] = (
                self.redraw_frequency_by_street[street][0] + 1,
                self.redraw_frequency_by_street[street][1] + 1
            )
        else:
            self.redraw_frequency_by_street = {
                0: (1, 1),  # (redraws, hands seen)
                1: (0, 0)
            }
            if street == 1:
                self.redraw_frequency_by_street[1] = (1, 1)
    
    def update_from_hand(self, hand_info):
        """
        Update overall stats at the end of a hand
        """
        self.hands_seen += 1
        
        # Adjust learning rate as we gather more data
        # Start aggressive, become more stable over time
        if self.hands_seen <= 10:
            self.learning_rate = self.base_learning_rate  # Fast learning at start
        elif self.hands_seen <= 50:
            self.learning_rate = self.base_learning_rate * 0.8  # Moderate
        else:
            self.learning_rate = self.base_learning_rate * 0.6  # Slower, more stable
    
    def get_redraw_frequency(self):
        """
        Get opponent's redraw frequency with confidence weighting
        """
        if self.hands_seen == 0:
            return 0.5  # Default assumption
        
        # Basic frequency
        basic_freq = self.redraw_count / self.hands_seen
        
        # Street-specific data if available
        if hasattr(self, 'redraw_frequency_by_street'):
            preflop_freq = self.redraw_frequency_by_street[0][0] / max(1, self.redraw_frequency_by_street[0][1])
            flop_freq = self.redraw_frequency_by_street[1][0] / max(1, self.redraw_frequency_by_street[1][1])
            
            # Weight based on confidence
            preflop_confidence = min(1.0, self.redraw_frequency_by_street[0][1] / 10.0)
            flop_confidence = min(1.0, self.redraw_frequency_by_street[1][1] / 10.0)
            
            # Combined weighted frequency
            weighted_freq = (
                preflop_freq * preflop_confidence + 
                flop_freq * flop_confidence + 
                0.5 * (2 - preflop_confidence - flop_confidence)
            ) / 2.0
            
            # Blend with basic frequency
            return 0.7 * weighted_freq + 0.3 * basic_freq
            
        return basic_freq
    
    def get_aggression_factor(self):
        """
        Get opponent's aggression factor with positional weighting
        """
        # Base aggression
        base_aggression = self.aggression
        
        # Apply positional weighting
        positional_aggression = 0.7 * self.position_aggression['SB'] + 0.3 * self.position_aggression['BB']
        
        # Calculate confidence-weighted result
        confidence_weight = min(1.0, self.aggression_confidence / 5.0)
        
        return (base_aggression * 0.7 + positional_aggression * 0.3) * confidence_weight + 0.5 * (1 - confidence_weight)
    
    def get_fold_equity(self):
        """
        Calculate fold equity against this opponent
        """
        # Higher fold frequency means more fold equity
        base_fold_equity = self.fold_frequency
        
        # Adjust based on aggression
        aggression_adjustment = -0.2 * (self.aggression - 0.5)
        
        # Adjust based on recent tendencies
        recent_actions = []
        for street in range(4):
            actions = self.actions_by_street[street][-10:] if self.actions_by_street[street] else []
            recent_actions.extend(actions)
        
        if recent_actions:
            recent_folds = recent_actions.count(action_types.FOLD.value)
            recent_fold_frequency = recent_folds / len(recent_actions)
            recency_adjustment = 0.3 * (recent_fold_frequency - self.fold_frequency)
        else:
            recency_adjustment = 0
        
        # Confidence-based blending with default
        confidence_weight = min(1.0, self.fold_confidence / 5.0)
        
        # Calculate final fold equity
        fold_equity = base_fold_equity + aggression_adjustment + recency_adjustment
        
        # Blend with default
        default_equity = 0.5  # Average fold equity
        
        return fold_equity * confidence_weight + default_equity * (1 - confidence_weight)
    
    def get_bluff_frequency(self):
        """
        Get opponent's estimated bluff frequency
        """
        confidence_weight = min(1.0, self.bluff_confidence / 5.0)
        
        # Calculate observed frequency
        total_aggressive_actions = self.bluffs_detected + self.value_bets_detected
        if total_aggressive_actions > 10:
            observed_frequency = self.bluffs_detected / total_aggressive_actions
        else:
            # Not enough data, rely more on prior
            observed_frequency = 0.3  # Average bluff frequency
            confidence_weight *= total_aggressive_actions / 10.0
        
        # Blend with prior based on confidence
        return (observed_frequency * confidence_weight) + (0.3 * (1 - confidence_weight))
    
    def adjust_hand_strength(self, base_strength, street, is_bluff_candidate=False):
        """
        Adjust effective hand strength based on opponent tendencies
        """
        # Start with base strength
        adjusted_strength = base_strength
        
        # Get opponent tendencies
        aggression = self.get_aggression_factor()
        fold_equity = self.get_fold_equity()
        bluff_frequency = self.get_bluff_frequency()
        
        # Against aggressive opponents, we need stronger hands
        if aggression > 0.7:
            adjusted_strength -= 0.05
        
        # Against passive opponents, we can play more marginal hands
        elif aggression < 0.3:
            adjusted_strength += 0.05
        
        # If opponent folds often and we're considering a bluff, increase effective strength
        if is_bluff_candidate and fold_equity > 0.6:
            adjusted_strength += 0.15
        
        # If opponent rarely folds, bluffing is less effective
        elif is_bluff_candidate and fold_equity < 0.3:
            adjusted_strength -= 0.1
        
        # If opponent bluffs often, we should call more with medium-strength hands
        if bluff_frequency > 0.4 and street >= 2 and 0.3 < base_strength < 0.7:
            adjusted_strength += 0.1
        
        return min(1.0, max(0.0, adjusted_strength))

# class OpponentModel:
#     """
#     Advanced opponent modeling using Bayesian inference and pattern recognition.
#     Implements real-time adaptation to exploit opponent tendencies and detect bluffing patterns.
#     Uses efficient data structures for quick retrieval during decision making.
#     """
    
#     def __init__(self):
#         # Core tracking statistics with Bayesian priors
#         self.aggression = 0.5  # Scale 0-1, 0.5 is baseline
#         self.fold_frequency = 0.5  # How often they fold to bets
#         self.bluff_frequency = 0.3  # Estimated bluffing frequency
        
#         # Redraw statistics with uncertainty tracking
#         self.redraw_count = 0
#         self.hands_seen = 0
#         self.redraw_history = []  # List of (discarded_card, drawn_card, street)
        
#         # Bayesian confidence tracking - how certain we are about our model
#         self.aggression_confidence = 1.0  # Increases with more observations
#         self.fold_confidence = 1.0
#         self.bluff_confidence = 1.0
        
#         # Street-specific patterns with temporal weighting
#         self.actions_by_street = {
#             0: [],  # Preflop
#             1: [],  # Flop
#             2: [],  # Turn
#             3: []   # River
#         }
        
#         # Advanced pattern detection
#         self.continuation_bet_freq = 0.5  # How often they c-bet
#         self.check_raise_freq = 0.1  # How often they check-raise
#         self.bet_sizing_by_street = {
#             0: [],  # Preflop bet sizes
#             1: [],  # Flop bet sizes
#             2: [],  # Turn bet sizes
#             3: []   # River bet sizes
#         }
        
#         # Timing patterns - can reveal hand strength
#         self.decision_times = defaultdict(list)
        
#         # Showdown analysis for advanced bluff detection
#         self.showdown_history = []
#         self.bluffs_detected = 0
#         self.value_bets_detected = 0
        
#         # Pattern recognition for advanced tells
#         self.action_sequences = defaultdict(int)  # Track action sequences
#         self.texture_response = defaultdict(list)  # Response to different board textures
        
#         # Adaptive learning rate - gives more weight to recent observations
#         self.learning_rate = 0.2  # Start with moderate learning rate
        
#         # Positional tendencies
#         self.position_aggression = {
#             'SB': 0.5,
#             'BB': 0.5
#         }
    
#     def update_from_action(self, action, street, bet_size=None, was_all_in=False):
#         """
#         Update model based on opponent's betting action using Bayesian updating.
        
#         Args:
#             action: action_types enum value
#             street: Current street (0-3)
#             bet_size: Size of the bet (if applicable)
#             was_all_in: Whether the action was all-in
#         """
#         # Record action for this street with temporal weighting
#         self.actions_by_street[street].append(action)
        
#         # Keep only recent actions (more weight to recent actions)
#         max_actions = 50  # Keep at most 50 actions per street
#         if len(self.actions_by_street[street]) > max_actions:
#             self.actions_by_street[street] = self.actions_by_street[street][-max_actions:]
        
#         # Update aggression metric with Bayesian updating
#         if action == action_types.RAISE.value:
#             # Calculate prior and likelihood
#             prior = self.aggression
#             likelihood = 0.9  # High likelihood of aggressive player raising
            
#             # Bayesian update
#             posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
            
#             # Apply update with learning rate (more weight to recent observations)
#             self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * posterior
            
#             # Update position-specific aggression
#             position = 'SB' if street % 2 == 0 else 'BB'  # Rough position approximation
#             self.position_aggression[position] = (
#                 0.9 * self.position_aggression[position] + 0.1 * posterior
#             )
            
#             # Track bet sizing for pattern detection
#             if bet_size is not None:
#                 self.bet_sizing_by_street[street].append(bet_size)
#                 # Keep only recent bet sizes
#                 if len(self.bet_sizing_by_street[street]) > max_actions:
#                     self.bet_sizing_by_street[street] = self.bet_sizing_by_street[street][-max_actions:]
        
#         elif action == action_types.CALL.value:
#             self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * 0.5
        
#         elif action == action_types.CHECK.value:
#             self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * 0.3
        
#         elif action == action_types.FOLD.value:
#             self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * 0.2
            
#             # Update fold frequency with Bayesian approach
#             prior = self.fold_frequency
#             likelihood = 0.9  # High likelihood of folding player having high fold frequency
#             posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
            
#             self.fold_frequency = (1 - self.learning_rate) * self.fold_frequency + self.learning_rate * posterior
        
#         # Update confidence - more observations increase confidence
#         self.aggression_confidence = min(10.0, self.aggression_confidence + 0.05)
#         self.fold_confidence = min(10.0, self.fold_confidence + 0.05)
        
#         # Track action sequences for pattern detection
#         # Last 3 actions as a sequence
#         all_actions = []
#         for s in range(4):
#             all_actions.extend(self.actions_by_street[s][-3:])
        
#         if len(all_actions) >= 3:
#             seq = tuple(all_actions[-3:])
#             self.action_sequences[seq] += 1
    
#     def update_from_showdown(self, opponent_cards, board, opponent_actions, won_hand):
#         """
#         Update model based on showdown information using advanced pattern recognition.
#         Essential for bluff detection and hand reading.
        
#         Args:
#             opponent_cards: Opponent's hole cards at showdown
#             board: Final board cards
#             opponent_actions: List of opponent actions leading to showdown
#             won_hand: Whether opponent won the hand
#         """
#         # Record detailed showdown information
#         showdown_record = {
#             'cards': opponent_cards,
#             'board': board,
#             'actions': opponent_actions,
#             'won': won_hand,
#             'street_actions': {s: [] for s in range(4)}  # Actions by street
#         }
        
#         # Classify actions by street for deeper analysis
#         current_street = 0
#         for action in opponent_actions:
#             showdown_record['street_actions'][current_street].append(action)
#             # Detect street changes based on action patterns
#             if action in [action_types.CALL.value, action_types.CHECK.value] and len(showdown_record['street_actions'][current_street]) >= 2:
#                 if current_street < 3:
#                     current_street += 1
        
#         self.showdown_history.append(showdown_record)
        
#         # Keep limited history for efficiency
#         max_showdowns = 100
#         if len(self.showdown_history) > max_showdowns:
#             self.showdown_history = self.showdown_history[-max_showdowns:]
        
#         # Detect bluffs by analyzing hand strength vs. betting pattern
#         if opponent_actions and opponent_actions[-1] == action_types.RAISE.value:
#             # Opponent's last action was a bet/raise
            
#             # Calculate hand strength at showdown
#             # This would use HandEvaluator in actual implementation
#             hand_strength = self._estimate_hand_strength(opponent_cards, board)
            
#             # Define bluff threshold - hands below this strength are considered bluffs when bet
#             bluff_threshold = 0.4  # 40th percentile or lower
            
#             was_bluffing = hand_strength < bluff_threshold
            
#             if was_bluffing:
#                 self.bluffs_detected += 1
#             else:
#                 self.value_bets_detected += 1
            
#             # Update bluff frequency with Bayesian updating
#             total_aggressive_actions = self.bluffs_detected + self.value_bets_detected
#             if total_aggressive_actions > 0:
#                 # Prior probability
#                 prior = self.bluff_frequency
                
#                 # New evidence (whether this bet was a bluff)
#                 likelihood = 0.8 if was_bluffing else 0.2
                
#                 # Bayesian update
#                 posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
                
#                 # Apply update with learning rate
#                 self.bluff_frequency = (1 - self.learning_rate) * self.bluff_frequency + self.learning_rate * posterior
                
#                 # Increase confidence in our bluff model
#                 self.bluff_confidence = min(10.0, self.bluff_confidence + 0.1)
        
#         # Analyze board texture vs. betting pattern
#         texture = self._classify_board_texture(board)
#         self.texture_response[texture].append({
#             'actions': opponent_actions,
#             'hand_strength': self._estimate_hand_strength(opponent_cards, board),
#             'won': won_hand
#         })


#     def update_from_redraw(self, discarded_card, drawn_card, street):
#         """
#         Update model based on opponent's redraw action with detailed pattern analysis.
        
#         Args:
#             discarded_card: Card index that was discarded
#             drawn_card: Card index that was drawn
#             street: Current street (0-1)
#         """
#         if street > 1:
#             return
        
#         self.redraw_count += 1
#         self.redraw_history.append((discarded_card, drawn_card, street))
        
#         # Advanced analysis of discarded cards
#         # Track what types of cards are being discarded
#         discarded_rank = discarded_card // 3
#         discarded_suit = discarded_card % 3
        
#         # Track if opponent tends to discard high cards or low cards
#         if discarded_rank >= 7:  # 9 or A
#             self.high_card_discard = getattr(self, 'high_card_discard', 0) + 1
#         else:
#             self.low_card_discard = getattr(self, 'low_card_discard', 0) + 1
        
        
#         # Update redraw frequency model with uncertainty
#         if hasattr(self, 'redraw_frequency_by_street'):
#             self.redraw_frequency_by_street[street] = (
#                 self.redraw_frequency_by_street[street][0] + 1,
#                 self.redraw_frequency_by_street[street][1] + 1
#             )
#         else:
#             self.redraw_frequency_by_street = {
#                 0: (1, 1),  # (redraws, hands seen)
#                 1: (0, 0)
#             }
#             if street == 1:
#                 self.redraw_frequency_by_street[1] = (1, 1)
    
#     def _estimate_hand_strength(self, hole_cards, board):
#         """
#         Estimate the strength of a hand without using the full hand evaluator.
#         Used for efficiency in opponent modeling.
        
#         Args:
#             hole_cards: Opponent's hole cards
#             board: Board cards
            
#         Returns:
#             Estimated hand strength (0-1)
#         """
#         # In a full implementation, this would use a simplified but accurate
#         # version of hand strength calculation
        
#         # Quick made hand checking
#         all_cards = hole_cards + board
        
#         # Count ranks and suits
#         ranks = [card // 3 for card in all_cards]
#         suits = [card % 3 for card in all_cards]
        
#         rank_counts = {}
#         for r in ranks:
#             rank_counts[r] = rank_counts.get(r, 0) + 1
        
#         suit_counts = {}
#         for s in suits:
#             suit_counts[s] = suit_counts.get(s, 0) + 1
        
#         # Check for common hand types
#         has_trips = max(rank_counts.values() if rank_counts else [0]) >= 3
#         has_two_pair = len([r for r, count in rank_counts.items() if count >= 2]) >= 2
#         has_pair = max(rank_counts.values() if rank_counts else [0]) >= 2
#         has_flush = max(suit_counts.values() if suit_counts else [0]) >= 5
        
#         # Approximate hand strength
#         if has_flush:
#             return 0.8  # Flush
#         elif has_trips:
#             return 0.7  # Three of a kind
#         elif has_two_pair:
#             return 0.6  # Two pair
#         elif has_pair:
#             return 0.4  # One pair
#         else:
#             # High card - rough estimation based on highest card
#             high_ranks = [r for r in ranks if r >= 7]  # 9 or A
#             if high_ranks:
#                 return 0.2  # High card with at least one high card
#             else:
#                 return 0.1  # Low high card
    
#     def _classify_board_texture(self, board):
#         """
#         Classify the board texture for pattern recognition.
        
#         Args:
#             board: List of board card indices
            
#         Returns:
#             String representing the board texture
#         """
#         if not board:
#             return "empty"
            
#         ranks = [card // 3 for card in board]
#         suits = [card % 3 for card in board]
        
#         # Count ranks and suits
#         rank_counts = {}
#         for r in ranks:
#             rank_counts[r] = rank_counts.get(r, 0) + 1
        
#         suit_counts = {}
#         for s in suits:
#             suit_counts[s] = suit_counts.get(s, 0) + 1
        
#         # Check for paired board
#         paired = max(rank_counts.values() if rank_counts else [0]) >= 2
        
#         # Check for flush potential
#         flush_draw = max(suit_counts.values() if suit_counts else [0]) >= 3
        
#         # Check for straight potential
#         straight_potential = False
#         if len(set(ranks)) >= 3:
#             sorted_ranks = sorted(set(ranks))
#             for i in range(len(sorted_ranks) - 2):
#                 if sorted_ranks[i+2] - sorted_ranks[i] <= 4:
#                     straight_potential = True
#                     break
        
#         # Check for high cards
#         high_card_count = sum(1 for r in ranks if r >= 7)  # 9 or A
        
#         # Classify texture
#         if paired and flush_draw:
#             return "paired_flush_draw"
#         elif paired:
#             return "paired"
#         elif flush_draw and straight_potential:
#             return "draw_heavy"
#         elif flush_draw:
#             return "flush_draw"
#         elif straight_potential:
#             return "straight_draw"
#         elif high_card_count >= 2:
#             return "high_cards"
#         else:
#             return "dry_low"
            
#     def update_from_hand(self, hand_info):
#         """
#         Update overall stats at the end of a hand with pattern detection.
        
#         Args:
#             hand_info: Dictionary of hand information
#         """
#         self.hands_seen += 1
        
#         # Adjust learning rate as we accumulate more hands
#         # Start aggressive, gradually become more stable
#         if self.hands_seen <= 10:
#             self.learning_rate = 0.3  # Fast learning at start
#         elif self.hands_seen <= 50:
#             self.learning_rate = 0.2  # Moderate learning
#         else:
#             self.learning_rate = 0.1  # Slower, more stable learning
    
#     def get_redraw_frequency(self):
#         """
#         Get opponent's overall redraw frequency with confidence weighting.
        
#         Returns:
#             Float representing frequency (0-1)
#         """
#         if self.hands_seen == 0:
#             return 0.5  # Default assumption
        
#         # Basic frequency
#         basic_freq = self.redraw_count / self.hands_seen
        
#         # If we have street-specific data, use that with weighting
#         if hasattr(self, 'redraw_frequency_by_street'):
#             preflop_freq = self.redraw_frequency_by_street[0][0] / max(1, self.redraw_frequency_by_street[0][1])
#             flop_freq = self.redraw_frequency_by_street[1][0] / max(1, self.redraw_frequency_by_street[1][1])
            
#             # Weight based on confidence
#             preflop_confidence = min(1.0, self.redraw_frequency_by_street[0][1] / 10.0)
#             flop_confidence = min(1.0, self.redraw_frequency_by_street[1][1] / 10.0)
            
#             # Combined weighted frequency
#             weighted_freq = (
#                 preflop_freq * preflop_confidence + 
#                 flop_freq * flop_confidence + 
#                 0.5 * (2 - preflop_confidence - flop_confidence)
#             ) / 2.0
            
#             # Blend with basic frequency
#             return 0.7 * weighted_freq + 0.3 * basic_freq
            
#         return basic_freq
    
#     def get_bluff_frequency(self):
#         """
#         Get opponent's estimated bluff frequency with confidence weighting.
        
#         Returns:
#             Float representing bluff frequency (0-1)
#         """
#         # Blend prior knowledge with observed frequency
#         # As confidence increases, we rely more on observed frequency
#         confidence_weight = min(1.0, self.bluff_confidence / 5.0)
        
#         # Calculate observed frequency
#         total_aggressive_actions = self.bluffs_detected + self.value_bets_detected
#         if total_aggressive_actions > 10:
#             observed_frequency = self.bluffs_detected / total_aggressive_actions
#         else:
#             # Not enough data, rely more on prior
#             observed_frequency = 0.3  # Average bluff frequency
#             confidence_weight *= total_aggressive_actions / 10.0
        
#         # Blend with prior based on confidence
#         return (observed_frequency * confidence_weight) + (0.3 * (1 - confidence_weight))
    
#     def get_aggression_factor(self):
#         """
#         Get opponent's aggression factor with positional weighting.
        
#         Returns:
#             Float representing aggression (0-1)
#         """
#         # Base aggression
#         base_aggression = self.aggression
        
#         # Apply positional weighting
#         positional_aggression = 0.7 * self.position_aggression['SB'] + 0.3 * self.position_aggression['BB']
        
#         # Calculate confidence-weighted result
#         confidence_weight = min(1.0, self.aggression_confidence / 5.0)
        
#         return (base_aggression * 0.7 + positional_aggression * 0.3) * confidence_weight + 0.5 * (1 - confidence_weight)
    
#     def get_fold_equity(self):
#         """
#         Calculate fold equity against this opponent with pattern recognition.
        
#         Returns:
#             Float between 0-1 representing likelihood of successful bluff
#         """
#         # Higher fold frequency means more fold equity
#         base_fold_equity = self.fold_frequency
        
#         # Adjust based on aggression (more aggressive players fold less)
#         aggression_adjustment = -0.2 * (self.aggression - 0.5)
        
#         # Adjust based on recent tendencies (weighted recency)
#         recent_actions = []
#         for street in range(4):
#             actions = self.actions_by_street[street][-10:] if self.actions_by_street[street] else []
#             recent_actions.extend(actions)
        
#         if recent_actions:
#             recent_folds = recent_actions.count(action_types.FOLD.value)
#             recent_fold_frequency = recent_folds / len(recent_actions)
#             recency_adjustment = 0.3 * (recent_fold_frequency - self.fold_frequency)
#         else:
#             recency_adjustment = 0
        
#         # Pattern-based adjustments from observed sequences
#         pattern_adjustment = 0
        
#         # Get most recent action sequence
#         all_actions = []
#         for street in range(4):
#             all_actions.extend(self.actions_by_street[street])
        
#         if len(all_actions) >= 3:
#             # Check if this sequence commonly leads to folds
#             recent_seq = tuple(all_actions[-3:])
#             if recent_seq in self.action_sequences and self.action_sequences[recent_seq] >= 3:
#                 # Check if this sequence is followed by folds
#                 fold_actions = all_actions.count(action_types.FOLD.value)
#                 if fold_actions > 0:
#                     pattern_adjustment = 0.1  # Slight adjustment based on pattern
        
#         # Confidence-based blending with default
#         confidence_weight = min(1.0, self.fold_confidence / 5.0)
        
#         # Calculate final fold equity
#         fold_equity = base_fold_equity + aggression_adjustment + recency_adjustment + pattern_adjustment
        
#         # Blend with default based on confidence
#         default_equity = 0.5  # Average fold equity
        
#         return fold_equity * confidence_weight + default_equity * (1 - confidence_weight)
    
#     def adjust_hand_strength(self, base_strength, street, is_bluff_candidate=False):
#         """
#         Adjust effective hand strength based on opponent tendencies with pattern matching.
        
#         Args:
#             base_strength: Raw hand strength (0-1)
#             street: Current street
#             is_bluff_candidate: Whether we're considering a bluff
            
#         Returns:
#             Adjusted hand strength
#         """
#         # Start with base strength
#         adjusted_strength = base_strength
        
#         # Get opponent tendencies with confidence weighting
#         aggression = self.get_aggression_factor()
#         fold_equity = self.get_fold_equity()
#         bluff_frequency = self.get_bluff_frequency()
        
#         # Against aggressive opponents, we need stronger hands to continue
#         if aggression > 0.7:
#             adjusted_strength -= 0.05
        
#         # Against passive opponents, we can play more marginal hands
#         elif aggression < 0.3:
#             adjusted_strength += 0.05
        
#         # If opponent folds often and we're considering a bluff, increase effective strength
#         if is_bluff_candidate and fold_equity > 0.6:
#             adjusted_strength += 0.15
        
#         # If opponent rarely folds, bluffing is less effective
#         elif is_bluff_candidate and fold_equity < 0.3:
#             adjusted_strength -= 0.1
        
#         # If opponent bluffs often, we should call more often with medium-strength hands
#         if bluff_frequency > 0.4 and street >= 2 and 0.3 < base_strength < 0.7:
#             adjusted_strength += 0.1
        
#         # On later streets, pattern recognition becomes more important
#         if street >= 2 and self.showdown_history:
#             # Pattern-based adjustment from showdown analysis
#             pattern_adj = self._get_pattern_based_adjustment(base_strength, street)
#             adjusted_strength += pattern_adj
        
#         return min(1.0, max(0.0, adjusted_strength))
    
#     def _get_pattern_based_adjustment(self, hand_strength, street):
#         """
#         Get pattern-based adjustment based on showdown history.
        
#         Args:
#             hand_strength: Current hand strength
#             street: Current street
            
#         Returns:
#             Adjustment value (-0.2 to 0.2)
#         """
#         # Not enough data for pattern recognition
#         if len(self.showdown_history) < 5:
#             return 0.0
        
#         # Look for patterns in similar situations
#         similar_situations = []
        
#         for showdown in self.showdown_history:
#             # Similar hand strength situations
#             if 'hand_strength' in showdown and abs(showdown.get('hand_strength', 0) - hand_strength) < 0.2:
#                 similar_situations.append(showdown)
        
#         # Not enough similar situations
#         if len(similar_situations) < 3:
#             return 0.0
        
#         # Analyze betting patterns in similar situations
#         aggression_count = 0
#         passive_count = 0
        
#         for situation in similar_situations:
#             actions = situation.get('actions', [])
#             if not actions:
#                 continue
                
#             street_actions = situation.get('street_actions', {}).get(street, [])
#             if not street_actions:
#                 continue
            
#             # Check if they played aggressively or passively
#             if action_types.RAISE.value in street_actions:
#                 aggression_count += 1
#             elif action_types.CHECK.value in street_actions or action_types.CALL.value in street_actions:
#                 passive_count += 1
        
#         # Calculate adjustment based on tendency
#         total = aggression_count + passive_count
#         if total < 3:
#             return 0.0
            
#         aggression_ratio = aggression_count / total
        
#         # If they're usually aggressive with similar hands, we should be more cautious
#         if aggression_ratio > 0.7:
#             return -0.1
#         # If they're usually passive with similar hands, we can be more aggressive
#         elif aggression_ratio < 0.3:
#             return 0.1
            
#         return 0.0
from gym_env import PokerEnv
from collections import defaultdict

action_types = PokerEnv.ActionType

class OpponentModel:
    """
    Improved opponent modeling using Bayesian inference and pattern recognition.
    Implements real-time adaptation to exploit opponent tendencies and detect bluffing patterns.
    Optimized for single-game rounds with reset after each match and single redraw limitation.
    """
    
    def __init__(self):
        # Core tracking statistics with Bayesian priors
        self.aggression = 0.5  # Scale 0-1, 0.5 is baseline
        self.fold_frequency = 0.5  # How often they fold to bets
        self.bluff_frequency = 0.3  # Estimated bluffing frequency
        
        # Redraw tracking - optimized for single redraw
        self.has_redrawn = False
        self.redraw_street = None  # Which street they redrawn on (0=preflop, 1=flop)
        self.discarded_card = None
        self.drawn_card = None
        
        # Bayesian confidence tracking - how certain we are about our model
        self.aggression_confidence = 1.0  # Increases with more observations
        self.fold_confidence = 1.0
        self.bluff_confidence = 1.0
        
        # Street-specific patterns with temporal weighting
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
        
        # Timing patterns - can reveal hand strength
        self.decision_times = defaultdict(list)
        
        # Showdown analysis for advanced bluff detection
        self.showdown_history = []
        self.bluffs_detected = 0
        self.value_bets_detected = 0
        
        # Pattern recognition for advanced tells
        self.action_sequences = defaultdict(int)  # Track action sequences
        self.texture_response = defaultdict(list)  # Response to different board textures
        
        # Adaptive learning rate - gives more weight to recent observations
        # Higher learning rate for single-game adaptation
        self.learning_rate = 0.3  # Start with high learning rate for faster adaptation
        
        # Positional tendencies
        self.position_aggression = {
            'SB': 0.5,
            'BB': 0.5
        }
    
    def update_from_action(self, action, street, bet_size=None, was_all_in=False):
        """
        Update model based on opponent's betting action using Bayesian updating.
        
        Args:
            action: action_types enum value
            street: Current street (0-3)
            bet_size: Size of the bet (if applicable)
            was_all_in: Whether the action was all-in
        """
        # Record action for this street with temporal weighting
        self.actions_by_street[street].append(action)
        
        # Update aggression metric with Bayesian updating
        if action == action_types.RAISE.value:
            # Calculate prior and likelihood
            prior = self.aggression
            likelihood = 0.9  # High likelihood of aggressive player raising
            
            # Bayesian update
            posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
            
            # Apply update with learning rate (more weight to recent observations)
            self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * posterior
            
            # Update position-specific aggression
            position = 'SB' if street % 2 == 0 else 'BB'  # Rough position approximation
            self.position_aggression[position] = (
                0.9 * self.position_aggression[position] + 0.1 * posterior
            )
            
            # Track bet sizing for pattern detection
            if bet_size is not None:
                self.bet_sizing_by_street[street].append(bet_size)
        
        elif action == action_types.CALL.value:
            self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * 0.5
        
        elif action == action_types.CHECK.value:
            self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * 0.3
        
        elif action == action_types.FOLD.value:
            self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * 0.2
            
            # Update fold frequency with Bayesian approach
            prior = self.fold_frequency
            likelihood = 0.9  # High likelihood of folding player having high fold frequency
            posterior = (likelihood * prior) / (likelihood * prior + (1 - likelihood) * (1 - prior))
            
            self.fold_frequency = (1 - self.learning_rate) * self.fold_frequency + self.learning_rate * posterior
        
        # Update confidence - more observations increase confidence
        self.aggression_confidence = min(5.0, self.aggression_confidence + 0.1)
        self.fold_confidence = min(5.0, self.fold_confidence + 0.1)
        
        # Track action sequences for pattern detection
        # Last 3 actions as a sequence
        all_actions = []
        for s in range(4):
            all_actions.extend(self.actions_by_street[s])
        
        if len(all_actions) >= 3:
            seq = tuple(all_actions[-3:])
            self.action_sequences[seq] += 1
    

    def update_from_redraw(self, discarded_card, drawn_card, street):
        """
        Update model based on opponent's redraw action with detailed analysis.
        Optimized for single redraw per match rule.
        
        Args:
            discarded_card: Card index that was discarded
            drawn_card: Card index that was drawn
            street: Current street (0-1)
        """
        if street > 1 or self.has_redrawn:
            return  # Can only redraw once before turn
        
        self.has_redrawn = True
        self.redraw_street = street
        self.discarded_card = discarded_card
        self.drawn_card = drawn_card
        
        # Analyze what kind of card was discarded
        discarded_rank = discarded_card // 3
        discarded_suit = discarded_card % 3
        
        # Check if they discarded a high card (implies risk-averse play)
        is_high_card = discarded_rank >= 7  # 9 or A
        
        # Check if drawn card is high (could imply chasing high cards)
        drawn_rank = drawn_card // 3
        is_drawn_high = drawn_rank >= 7
        
        # Analyze discard/draw pattern to infer strategy
        if is_high_card and not is_drawn_high:
            # Discarded high card for lower card - likely building a specific hand
            # Adjust bluff detection - this player may be more pattern-focused
            self.bluff_frequency = (1 - self.learning_rate) * self.bluff_frequency + self.learning_rate * 0.2
        
        elif not is_high_card and is_drawn_high:
            # Discarded low card for high card - likely chasing high cards
            # More straightforward play, adjust bluff frequency higher (less sophisticated)
            self.bluff_frequency = (1 - self.learning_rate) * self.bluff_frequency + self.learning_rate * 0.4
            
        # Redrawing on flop is more desperate than preflop
        if street == 1:  # Flop
            # More likely to be drawing to complete a hand - less likely bluffing
            self.bluff_frequency = (1 - self.learning_rate) * self.bluff_frequency + self.learning_rate * 0.2
            # More aggressive play
            self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * 0.6
        else:  # Preflop
            # Standard adjustments based on typical preflop strategy
            self.aggression = (1 - self.learning_rate) * self.aggression + self.learning_rate * 0.5
    
    def _estimate_hand_strength(self, hole_cards, board):
        """
        Estimate the strength of a hand without using the full hand evaluator.
        Used for efficiency in opponent modeling.
        
        Args:
            hole_cards: Opponent's hole cards
            board: Board cards
            
        Returns:
            Estimated hand strength (0-1)
        """
        # In a full implementation, this would use a simplified but accurate
        # version of hand strength calculation
        
        # Quick made hand checking
        all_cards = hole_cards + board
        
        # Count ranks and suits
        ranks = [card // 3 for card in all_cards]
        suits = [card % 3 for card in all_cards]
        
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        
        # Check for common hand types
        has_trips = max(rank_counts.values() if rank_counts else [0]) >= 3
        has_two_pair = len([r for r, count in rank_counts.items() if count >= 2]) >= 2
        has_pair = max(rank_counts.values() if rank_counts else [0]) >= 2
        has_flush = max(suit_counts.values() if suit_counts else [0]) >= 5
        
        # Approximate hand strength
        if has_flush:
            return 0.8  # Flush
        elif has_trips:
            return 0.7  # Three of a kind
        elif has_two_pair:
            return 0.6  # Two pair
        elif has_pair:
            return 0.4  # One pair
        else:
            # High card - rough estimation based on highest card
            high_ranks = [r for r in ranks if r >= 7]  # 9 or A
            if high_ranks:
                return 0.2  # High card with at least one high card
            else:
                return 0.1  # Low high card
    
    def update_from_hand(self, hand_info):
        """
        Update overall stats at the end of a hand.
        
        Args:
            hand_info: Dictionary of hand information
        """
        # Adjust learning rate to be more aggressive for single-game adaptation
        self.learning_rate = 0.3
    
    def get_redraw_insight(self):
        """
        Get insights from opponent's redraw behavior.
        
        Returns:
            Dictionary with redraw insights
        """
        if not self.has_redrawn:
            return {
                'has_redrawn': False,
                'likely_drawing_to': 'unknown',
                'hand_quality': 'unknown'
            }
        
        discarded_rank = self.discarded_card // 3
        discarded_suit = self.discarded_card % 3
        drawn_rank = self.drawn_card // 3
        drawn_suit = self.drawn_card % 3
        
        likely_drawing_to = 'unknown'
        hand_quality = 'unknown'
        
        # If they discarded a high card
        if discarded_rank >= 7:  # 9 or A
            hand_quality = 'medium-strong'  # They likely have something specific
            # They're willing to give up a high card, so likely building a specific hand
            if drawn_suit == discarded_suit:
                likely_drawing_to = 'flush'
            else:
                likely_drawing_to = 'specific_pattern'
        else:
            # If they discarded a low card
            if drawn_rank >= 7:  # Drew a high card
                hand_quality = 'weak-medium'  # They're trying to improve
                likely_drawing_to = 'high_card_strength'
            else:
                # Both discarded and drew low cards
                hand_quality = 'weak'
                likely_drawing_to = 'specific_pattern'
                
        # Adjust based on which street they redrawn
        if self.redraw_street == 1:  # Flop
            # Redrawing on flop implies more desperation
            if hand_quality == 'medium-strong':
                hand_quality = 'medium'
            elif hand_quality == 'weak-medium':
                hand_quality = 'weak'
        
        return {
            'has_redrawn': True,
            'redraw_street': self.redraw_street,
            'likely_drawing_to': likely_drawing_to,
            'hand_quality': hand_quality,
            'discarded_high': discarded_rank >= 7,
            'drew_high': drawn_rank >= 7
        }
    
    def get_bluff_frequency(self):
        """
        Get opponent's estimated bluff frequency based on current game state only.
        
        Returns:
            Float representing bluff frequency (0-1)
        """
        # Base bluff frequency - start with a reasonable default
        base_bluff_freq = 0.3
        
        # If opponent has redrawn, this gives us concrete information to work with
        if self.has_redrawn:
            redraw_insight = self.get_redraw_insight()
            
            if redraw_insight['hand_quality'] == 'weak':
                # Weak hand means more likely to bluff
                bluff_adjustment = 0.2
            elif redraw_insight['hand_quality'] == 'medium':
                bluff_adjustment = 0.1
            else:
                # Strong hand means less likely to bluff
                bluff_adjustment = -0.1
                    
            # Apply the adjustment
            base_bluff_freq = max(0.1, min(0.9, base_bluff_freq + bluff_adjustment))
            
            # Redraw street-specific adjustments
            if self.redraw_street == 1:  # Flop
                # Redrawing on flop is usually targeting a specific draw
                # Less likely to be pure bluffing later if they made this investment
                base_bluff_freq -= 0.05
            
            # Card-specific adjustments
            if hasattr(self, 'discarded_card') and hasattr(self, 'drawn_card'):
                discarded_rank = self.discarded_card // 3
                drawn_rank = self.drawn_card // 3
                
                # If they discarded a high card for a lower card, they're building a pattern
                # Less likely to be purely bluffing
                if discarded_rank >= 7 and drawn_rank < discarded_rank:
                    base_bluff_freq -= 0.1
                
                # If they discarded a low card to draw a high card, classic value play
                # Still might bluff if they miss, but less likely overall
                elif discarded_rank < 7 and drawn_rank >= 7:
                    base_bluff_freq -= 0.05
        
        # Current street aggression patterns
        current_actions = []
        for street, actions in self.actions_by_street.items():
            if actions:  # Only consider streets with actions
                current_actions.extend(actions)
        
        if current_actions:
            # Analyze action patterns in current game
            raise_count = current_actions.count(action_types.RAISE.value)
            check_count = current_actions.count(action_types.CHECK.value)
            
            # Lots of raises often indicates either strong hands or bluffing
            if len(current_actions) >= 3:
                raise_ratio = raise_count / len(current_actions)
                
                # Very high raise frequency usually means more bluffing
                if raise_ratio > 0.7:
                    base_bluff_freq += 0.15
                # Moderate raising with some checking often indicates honest play
                elif 0.3 <= raise_ratio <= 0.5 and check_count > 0:
                    base_bluff_freq -= 0.1
        
        # Confidence weighting - in a single game, we have limited confidence
        confidence = min(0.6, self.aggression_confidence / 6.0)  # Cap at 60%
        
        # Default bluff frequency to blend with
        default_freq = 0.3
        
        # Final weighted result
        return base_bluff_freq * confidence + default_freq * (1.0 - confidence)


    def get_aggression_factor(self):
        """
        Get opponent's aggression factor with positional weighting.
        Factors in redraw insights.
        
        Returns:
            Float representing aggression (0-1)
        """
        # Base aggression
        base_aggression = self.aggression
        
        # Apply positional weighting
        positional_aggression = 0.7 * self.position_aggression['SB'] + 0.3 * self.position_aggression['BB']
        
        # Calculate confidence-weighted result
        confidence_weight = min(1.0, self.aggression_confidence / 5.0)
        
        # If opponent has redrawn, factor that into aggression estimate
        if self.has_redrawn:
            redraw_insight = self.get_redraw_insight()
            
            if redraw_insight['hand_quality'] == 'strong':
                # Strong hand correlates with more aggression
                aggression_adjustment = 0.2
            elif redraw_insight['hand_quality'] == 'medium':
                aggression_adjustment = 0.1
            elif redraw_insight['hand_quality'] == 'weak':
                # Weak hand typically means less aggression
                aggression_adjustment = -0.1
            else:
                aggression_adjustment = 0
                
            # Apply redraw-based adjustment
            base_aggression = max(0.1, min(0.9, base_aggression + aggression_adjustment))
        
        return (base_aggression * 0.7 + positional_aggression * 0.3) * confidence_weight + 0.5 * (1 - confidence_weight)
    

    def get_fold_equity(self):
        """
        Calculate fold equity against this opponent with pattern recognition.
        Takes into account redraw information, optimized for single-match scenario.
        
        Returns:
            Float between 0-1 representing likelihood of successful bluff
        """
        base_fold_equity = 0.4
        
        # Adjust based on aggression (more aggressive players fold less)
        aggression_adjustment = -0.2 * (self.aggression - 0.5)
        
        # If opponent has redrawn, factor that into fold equity
        if self.has_redrawn:
            redraw_insight = self.get_redraw_insight()
            
            if redraw_insight['hand_quality'] == 'strong':
                # Strong hand means less likely to fold
                fold_adjustment = -0.2
            elif redraw_insight['hand_quality'] == 'medium':
                fold_adjustment = -0.1
            elif redraw_insight['hand_quality'] == 'weak':
                # Weak hand means more likely to fold
                fold_adjustment = 0.2
            else:
                fold_adjustment = 0
                
            # Apply redraw-based fold equity adjustment
            base_fold_equity = max(0.1, min(0.9, base_fold_equity + fold_adjustment))
            
            # Additional adjustments based on redraw street
            if self.redraw_street == 1:  # Redrew on flop
                # Redrawing on flop often indicates desperation, so more likely to fold later
                base_fold_equity += 0.1
            
            # Discarded/drawn card analysis
            if hasattr(self, 'discarded_card') and hasattr(self, 'drawn_card'):
                discarded_rank = self.discarded_card // 3
                drawn_rank = self.drawn_card // 3
                
                # If they discarded a high card (9 or A) for a lower card
                if discarded_rank >= 7 and drawn_rank < discarded_rank:
                    # They're playing for a specific pattern - less likely to fold
                    base_fold_equity -= 0.1
                
                # If they discarded a low card for a higher card
                elif discarded_rank < 7 and drawn_rank > discarded_rank:
                    # They're chasing higher cards - likely weaker hand, more likely to fold
                    base_fold_equity += 0.1
        
        # Only look at current street actions
        current_street_actions = self.actions_by_street.get(max(0, min(3, self.redraw_street or 0)), [])
        
        if current_street_actions:
            # Calculate aggression ratio from actual actions
            raise_count = current_street_actions.count(action_types.RAISE.value)
            fold_count = current_street_actions.count(action_types.FOLD.value)
            
            # Only make adjustments if we have enough actions
            if len(current_street_actions) >= 2:
                action_ratio = (raise_count - fold_count) / max(1, len(current_street_actions))
                
                # Aggressive action ratio means less fold equity
                aggression_adjustment = -0.15 * action_ratio
        
        # Since we have low confidence in a single-match scenario, weight towards default
        confidence_weight = min(0.7, self.aggression_confidence / 7.0)  # Max 70% confidence
        
        # Calculate final fold equity
        fold_equity = base_fold_equity + aggression_adjustment
        
        # Blend with default based on confidence
        default_equity = 0.5  # Average fold equity
        
        return fold_equity * confidence_weight + default_equity * (1 - confidence_weight)
    

    def adjust_hand_strength(self, base_strength, street, is_bluff_candidate=False):
        """
        Adjust effective hand strength based on opponent tendencies with pattern matching.
        Takes into account redraw information for more accurate adjustments.
        
        Args:
            base_strength: Raw hand strength (0-1)
            street: Current street
            is_bluff_candidate: Whether we're considering a bluff
            
        Returns:
            Adjusted hand strength
        """
        # Start with base strength
        adjusted_strength = base_strength
        
        # Get opponent tendencies with confidence weighting
        aggression = self.get_aggression_factor()
        fold_equity = self.get_fold_equity()
        bluff_frequency = self.get_bluff_frequency()
        
        # Against aggressive opponents, we need stronger hands to continue
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
        
        # If opponent bluffs often, we should call more often with medium-strength hands
        if bluff_frequency > 0.4 and street >= 2 and 0.3 < base_strength < 0.7:
            adjusted_strength += 0.1
        
        # Factor in redraw information
        if self.has_redrawn:
            redraw_insight = self.get_redraw_insight()
            
            # If we're past the redraw street, we can use this information more confidently
            if street > self.redraw_street:
                if redraw_insight['hand_quality'] == 'strong':
                    # They have a strong hand, we need a stronger hand to continue
                    if is_bluff_candidate:
                        adjusted_strength -= 0.15  # Bluffing is less effective
                    else:
                        adjusted_strength -= 0.1   # Need stronger value hand
                
                elif redraw_insight['hand_quality'] == 'weak':
                    # They have a weak hand, we can be more aggressive
                    if is_bluff_candidate:
                        adjusted_strength += 0.15  # Bluffing is more effective
                    else:
                        adjusted_strength += 0.05  # Weaker value hands are playable
            
            # Redraw on flop is particularly significant
            if self.redraw_street == 1 and street >= 2:
                # They redrawn on flop - they're likely drawing to something specific
                if redraw_insight['likely_drawing_to'] == 'flush' and street == 2:
                    # If they're drawing to a flush and we're on the turn
                    # They might have completed their hand - be more cautious
                    adjusted_strength -= 0.1
        
        return min(1.0, max(0.0, adjusted_strength))
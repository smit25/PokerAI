import random

class Redraw:
    """
    Advanced redraw strategy using Counterfactual Regret Minimization concepts.
    Implements Expected Value (EV) maximization with efficient pruning for performance.
    Uses importance sampling to efficiently explore the redraw decision space.
    """
    
    def __init__(self, hand_evaluator):
        self.hand_evaluator = hand_evaluator
        self.redraw_cache = {}
        
    def should_redraw(self, hole_cards, board, street, position="SB", opponent_tendencies=None):
        """
        Advanced redraw decision-making using EV maximization and importance sampling.
        
        Args:
            hole_cards: List of hole card indices
            board: List of board card indices
            street: Current street (0=preflop, 1=flop)
            position: "SB" or "BB"
            opponent_tendencies: Dict of opponent stats
            
        Returns:
            Tuple of (should_redraw, card_idx_to_discard)
        """
        if street > 1:
            return False, -1
        
        cache_key = (tuple(hole_cards), tuple(board), street, position)
        if cache_key in self.redraw_cache:
            return self.redraw_cache[cache_key]
        
        # Get current hand strength with potential range information
        opponent_range = self._construct_opponent_range(opponent_tendencies, street, position)
        current_strength = self.hand_evaluator.get_hand_strength(hole_cards, board, opponent_range)
        
        # For very strong hands, don't redraw
        if current_strength > 0.8:
            self.redraw_cache[cache_key] = (False, -1)
            return False, -1
        
        # Calculate expected value of redrawing vs not redrawing
        keep_ev = self._calculate_keep_ev(hole_cards, board, street, position, opponent_tendencies)
        
        # Calculate EV of redrawing each card using importance sampling
        redraw_options = []
        
        for card_idx in range(2):
            # Get the card we're keeping
            other_card = hole_cards[1-card_idx]
            
            # Cards that are already in play and unavailable
            unavailable_cards = set(hole_cards) | set(board)
            
            # Determine available cards for replacement
            available_cards = set(range(27)) - unavailable_cards
            
            # Use importance sampling - focus on likely valuable replacements
            # Group cards into buckets by rank for more efficient sampling
            rank_buckets = {}
            for card in available_cards:
                rank = card // 3
                if rank not in rank_buckets:
                    rank_buckets[rank] = []
                rank_buckets[rank].append(card)
            
            # Adaptive sample size based on available time
            sample_size = 15 if street == 0 else 10
            
            # Sample cards, giving preference to high ranks
            sampled_cards = []
            # Sample high cards (A, 9, 8) more frequently
            high_ranks = [8, 7, 6]  # A, 9, 8
            for rank in high_ranks:
                if rank in rank_buckets:
                    sampled_cards.extend(rank_buckets[rank])
            
            # If we need more samples, add middle ranks
            if len(sampled_cards) < sample_size:
                mid_ranks = [5, 4, 3]  # 7, 6, 5
                for rank in mid_ranks:
                    if rank in rank_buckets and len(sampled_cards) < sample_size:
                        sampled_cards.extend(rank_buckets[rank])
            
            # If still need more, add remaining ranks
            remaining_ranks = [2, 1, 0]  # 4, 3, 2
            for rank in remaining_ranks:
                if rank in rank_buckets and len(sampled_cards) < sample_size:
                    sampled_cards.extend(rank_buckets[rank])
            
            # Limit to desired sample size
            if len(sampled_cards) > sample_size:
                sampled_cards = random.sample(sampled_cards, sample_size)
            
            # Calculate EV for each replacement
            total_ev = 0
            weight_sum = 0
            
            for new_card in sampled_cards:
                # Higher weight for higher ranked cards
                rank = new_card // 3
                weight = 1.0 + (rank / 8.0)  # Weight from 1.0 to 2.0
                
                # Create new hand
                new_hand = [other_card, new_card]
                
                # Calculate new hand EV
                new_ev = self._calculate_hand_ev(new_hand, board, street, position, opponent_tendencies)
                
                # Weighted sum
                total_ev += new_ev * weight
                weight_sum += weight
            
            # Calculate weighted average EV
            if weight_sum > 0:
                avg_ev = total_ev / weight_sum
                redraw_options.append((avg_ev, card_idx))
            else:
                redraw_options.append((0, card_idx))
        
        # Find best redraw option
        redraw_options.sort(reverse=True)
        best_redraw_ev, best_card_idx = redraw_options[0]
        
        # Compare to keeping current hand
        if best_redraw_ev > keep_ev + 0.05:  # Threshold for redrawing
            decision = (True, best_card_idx)
        else:
            decision = (False, -1)
        
        # Cache result
        self.redraw_cache[cache_key] = decision
        return decision
    
    def _construct_opponent_range(self, opponent_tendencies, street, position):
        """
        Construct opponent range model from tendencies.
        
        Args:
            opponent_tendencies: Dictionary of opponent statistics
            street: Current street
            position: "SB" or "BB"
            
        Returns:
            Range model dictionary
        """
        # Default range model for unknown opponent
        if not opponent_tendencies:
            return {
                'polarization': 0.5,  # How polarized the range is (0=condensed, 1=polarized)
                'strength_cap': 1.0,   # Upper bound on hand strength
                'strength_floor': 0.0  # Lower bound on hand strength
            }
        
        # Extract relevant tendencies
        aggression = opponent_tendencies.get('aggression', 0.5)
        fold_frequency = opponent_tendencies.get('fold_frequency', 0.5)
        
        # Calculate range polarization
        # More aggressive opponents tend to have more polarized ranges
        polarization = min(1.0, aggression * 1.2)
        
        # Calculate strength bounds
        # Tight opponents have higher strength floors
        strength_floor = min(0.4, (1 - fold_frequency) * 0.5)
        
        # Street-specific adjustments
        if street == 0:  # Preflop
            pass  # Use base values
        elif street == 1:  # Flop
            # Ranges typically get stronger post-flop
            strength_floor += 0.1
        
        return {
            'polarization': polarization,
            'strength_cap': 1.0,
            'strength_floor': strength_floor
        }
    
    def _calculate_keep_ev(self, hole_cards, board, street, position, opponent_tendencies):
        """
        Calculate EV of keeping current hand.
        
        Args:
            hole_cards: List of hole card indices
            board: List of board card indices
            street: Current street
            position: "SB" or "BB"
            opponent_tendencies: Dict of opponent stats
            
        Returns:
            Expected value of keeping current hand
        """
        return self._calculate_hand_ev(hole_cards, board, street, position, opponent_tendencies)
    
    def _calculate_hand_ev(self, hole_cards, board, street, position, opponent_tendencies):
        """
        Calculate expected value of a hand in the current game state.
        
        Args:
            hole_cards: List of hole card indices
            board: List of board card indices
            street: Current street
            position: "SB" or "BB"
            opponent_tendencies: Dict of opponent stats
            
        Returns:
            Expected value estimate
        """
        # Construct opponent range model
        opponent_range = self._construct_opponent_range(opponent_tendencies, street, position)
        
        # Get hand strength against range
        hand_strength = self.hand_evaluator.get_hand_strength(hole_cards, board, opponent_range)
        
        # Position-based adjustments
        position_factor = 1.05 if position == "SB" else 0.95
        
        # Street-based adjustments
        street_factor = 1.0
        if street == 1:  # Flop
            # On the flop, position matters more
            position_factor = 1.1 if position == "SB" else 0.9
        
        # Calculate simplified EV
        return hand_strength * position_factor * street_factor
    
    def strategic_redraw(self, hole_cards, board, street, position, opponent_model=None):
        """
        Advanced redraw strategy that incorporates game theory concepts and opponent exploitation.
        
        Args:
            hole_cards: List of hole card indices
            board: List of board card indices
            street: Current street
            position: "SB" or "BB"
            opponent_model: OpponentModel instance
            
        Returns:
            Tuple of (should_redraw, card_idx_to_discard)
        """
        # Core redraw decision
        should_redraw, card_idx = self.should_redraw(
            hole_cards, board, street, position,
            opponent_model.__dict__ if opponent_model else None
        )
        
        # If opponent model is available, make exploitative adjustments
        if opponent_model and hasattr(opponent_model, 'get_redraw_frequency'):
            # Extract opponent tendencies
            opp_redraw_freq = opponent_model.get_redraw_frequency()
            opp_fold_equity = opponent_model.get_fold_equity()
            opp_aggression = getattr(opponent_model, 'aggression', 0.5)
            
            # Current hand strength
            current_strength = self.hand_evaluator.get_hand_strength(hole_cards, board)
            
            # Exploitative adjustments based on game theory principles
            
            # Case 1: Opponent redraws very often - we should be more selective
            if opp_redraw_freq > 0.7:
                # Only redraw very weak hands or for significant improvements
                if current_strength > 0.4 and should_redraw:
                    # Verify the improvement is truly significant
                    improvements = self._calculate_targeted_improvements(
                        hole_cards, board, street, position, opponent_model.__dict__
                    )
                    
                    # Only redraw for major improvements
                    if improvements[0][0] < 0.15:  # Need 15% improvement
                        return False, -1
            
            # Case 2: Opponent rarely redraws - we can be more aggressive with redraws
            elif opp_redraw_freq < 0.3 and not should_redraw:
                # Check for even marginal improvements
                improvements = self._calculate_targeted_improvements(
                    hole_cards, board, street, position, opponent_model.__dict__
                )
                
                # Redraw for even small improvements
                if improvements[0][0] > 0.05:  # Just 5% improvement needed
                    return True, improvements[0][1]
            
            # Case 3: Deceptive redraw with strong hands
            if not should_redraw and current_strength > 0.75:
                # Calculate the balance between deception and value
                deception_threshold = 0.02  # Reduced from 0.1 to 0.02 (80% reduction)
                
                # Adjust based on opponent type:
                # - Against observant opponents (lower fold equity), deception is more valuable
                # - Against unobservant opponents (high fold equity), keeping value is better
                deception_threshold *= (1.5 - opp_fold_equity)
                
                # Add additional restriction: only consider deception in later hands
                if opponent_model.hands_seen < 10:
                    return should_redraw, card_idx  # No deception in early hands
                
                # Occasionally make a deceptive redraw
                if random.random() < deception_threshold:
                    # Determine which card to discard (typically the lower one)
                    card1_rank = hole_cards[0] // 3
                    card2_rank = hole_cards[1] // 3
                    
                    # Discard lower card
                    discard_idx = 0 if card1_rank < card2_rank else 1
                    return True, discard_idx
            # Case 4: Information gathering early in the match
            if opponent_model.hands_seen < 20 and random.random() < 0.2:
                # Sometimes redraw even marginal hands to gather information
                if 0.3 < current_strength < 0.6:
                    # Calculate which card to discard
                    improvements = self._calculate_targeted_improvements(
                        hole_cards, board, street, position, opponent_model.__dict__
                    )
                    return True, improvements[0][1]
        
        return should_redraw, card_idx
    
    def _calculate_targeted_improvements(self, hole_cards, board, street, position, opponent_tendencies):
        """
        Calculate improvements with targeted card sampling for efficiency.
        
        Args:
            hole_cards: List of hole card indices
            board: List of board card indices
            street: Current street
            position: "SB" or "BB"
            opponent_tendencies: Dict of opponent stats
            
        Returns:
            List of (improvement, card_idx) tuples sorted by improvement
        """
        current_ev = self._calculate_hand_ev(hole_cards, board, street, position, opponent_tendencies)
        improvements = []
        
        for card_idx in range(2):
            other_card = hole_cards[1-card_idx]
            
            # Get unavailable cards
            unavailable = set(hole_cards) | set(board)
            
            # Advanced importance sampling focused on highest potential improvement
            available_cards = []
            
            # First, check high cards (always worth sampling)
            high_ranks = [8, 7, 6]  # A, 9, 8
            for rank in high_ranks:
                for suit in range(3):
                    card = rank * 3 + suit
                    if card not in unavailable:
                        available_cards.append(card)
            
            # Then add potential straight/flush completers if on the flop
            if board and len(board) >= 3:
                # Cards that might complete straights or flushes
                all_ranks = [c // 3 for c in [other_card] + board]
                all_suits = [c % 3 for c in [other_card] + board]
                
                # Possible straight completers
                for rank in range(9):  # 2-A
                    if rank not in all_ranks:
                        # Skip if not potentially completing a straight
                        continue
                    
                    # Check if this rank could complete a straight
                    straight_potential = False
                    for i in range(max(0, rank-4), min(9, rank+1)):
                        count = sum(1 for r in all_ranks if i <= r < i+5)
                        if count >= 3:  # Could complete a straight
                            straight_potential = True
                            break
                    
                    if straight_potential:
                        for suit in range(3):
                            card = rank * 3 + suit
                            if card not in unavailable and card not in available_cards:
                                available_cards.append(card)
                
                # Possible flush completers
                suit_counts = {}
                for s in all_suits:
                    suit_counts[s] = suit_counts.get(s, 0) + 1
                
                for suit, count in suit_counts.items():
                    if count >= 3:  # Potential flush
                        for rank in range(9):
                            card = rank * 3 + suit
                            if card not in unavailable and card not in available_cards:
                                available_cards.append(card)
            
            # Add a few random cards to ensure diversity
            all_remaining = [c for c in range(27) 
                            if c not in unavailable and c not in available_cards]
            
            if all_remaining:
                random_sample = random.sample(
                    all_remaining, 
                    min(5, len(all_remaining))
                )
                available_cards.extend(random_sample)
            
            # Calculate improvement for each card
            card_improvements = []
            
            for new_card in available_cards:
                new_hand = [other_card, new_card]
                new_ev = self._calculate_hand_ev(
                    new_hand, board, street, position, opponent_tendencies
                )
                improvement = new_ev - current_ev
                card_improvements.append(improvement)
            
            # Average improvement
            if card_improvements:
                avg_improvement = sum(card_improvements) / len(card_improvements)
                # Find max improvement (optimistic estimate)
                max_improvement = max(card_improvements) if card_improvements else 0
                # Use weighted combination of average and max
                combined_improvement = 0.7 * avg_improvement + 0.3 * max_improvement
                improvements.append((combined_improvement, card_idx))
            else:
                improvements.append((0, card_idx))
        
        # Sort by improvement (descending)
        return sorted(improvements, reverse=True)
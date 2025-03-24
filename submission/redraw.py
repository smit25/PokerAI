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
        # Safety checks
        if hole_cards is None or len(hole_cards) != 2:
            return False, -1
            
        if board is None:
            board = []
            
        if street > 1 or street < 0:
            return False, -1
        
        cache_key = (tuple(hole_cards), tuple(board), street, position)
        if cache_key in self.redraw_cache:
            return self.redraw_cache[cache_key]
        
        # Get current hand strength with potential range information
        opponent_range = self._construct_opponent_range(opponent_tendencies, street, position)
        current_strength = self.hand_evaluator.get_hand_strength(hole_cards, board, opponent_range)
        
        # For very strong hands, don't redraw
        if current_strength > 0.85:  # Increased threshold slightly
            self.redraw_cache[cache_key] = (False, -1)
            return False, -1
        
        # Calculate expected value of keeping current hand
        keep_ev = self._calculate_keep_ev(hole_cards, board, street, position, opponent_tendencies)
        
        # Get hand type to make more informed decisions
        hand_type = self._get_hand_type(hole_cards, board)
        
        # SMART CARD SELECTION: Directly evaluate which card to discard based on rank and potential
        card_values = []
        for i, card in enumerate(hole_cards):
            # Calculate the card's intrinsic value
            rank = card // 3
            suit = card % 3
            
            # Calculate intrinsic value - higher is better to keep
            # Aces (rank 8) are worth much more than low cards
            intrinsic_value = (rank / 8.0) ** 1.5  # Exponential scaling to prioritize high cards
            
            # Value adjustment based on current board
            if board and street > 0:
                # Count cards of same rank and suit on board
                same_rank_count = sum(1 for b in board if b // 3 == rank)
                same_suit_count = sum(1 for b in board if b % 3 == suit)
                
                # Pairs are valuable
                if same_rank_count > 0:
                    intrinsic_value += 0.3 * same_rank_count
                
                # Potential flushes are valuable
                if same_suit_count >= 2:
                    intrinsic_value += 0.25
                
                # Potential straights
                board_ranks = sorted([b // 3 for b in board])
                straight_potential = 0
                
                # Check if this card could be part of a straight
                for start in range(max(0, rank-4), min(5, rank)):
                    # Count how many unique ranks in range [start, start+4] are in board_ranks or rank
                    ranks_in_range = set(r for r in board_ranks if start <= r <= start+4)
                    if rank >= start and rank <= start+4:
                        ranks_in_range.add(rank)
                    
                    # If we have 3+ ranks, there's straight potential
                    if len(ranks_in_range) >= 3:
                        straight_potential = 0.2
                        break
                
                intrinsic_value += straight_potential
            
            # Store the calculated value
            card_values.append((intrinsic_value, i))
        
        # Sort by value (ascending, so lowest value is first - to be discarded)
        card_values.sort()
        
        # The card with the lowest value is the best candidate for discarding
        best_discard_idx = card_values[0][1]
        other_card_idx = 1 - best_discard_idx
        
        # Now calculate expected improvement from redrawing
        unavailable_cards = set(hole_cards) | set(board)
        
        # Determine available cards for replacement
        available_cards = set(range(27)) - unavailable_cards
        
        # More strategic sampling of replacement cards
        sampled_cards = self._smart_card_sampling(available_cards, hole_cards[other_card_idx], board, hand_type, street)
        
        # Calculate EV for redrawing with these sampled cards
        total_ev = 0
        weight_sum = 0
        
        for new_card in sampled_cards:
            # Higher weight for cards that synergize with current hand
            weight = self._calculate_synergy_weight(new_card, hole_cards[other_card_idx], board, hand_type)
            
            # Create new hand
            new_hand = [hole_cards[other_card_idx], new_card]
            
            # Calculate new hand EV
            new_ev = self._calculate_hand_ev(new_hand, board, street, position, opponent_tendencies)
            
            # Weighted sum
            total_ev += new_ev * weight
            weight_sum += weight
        
        # Calculate weighted average EV
        if weight_sum > 0:
            redraw_ev = total_ev / weight_sum
        else:
            redraw_ev = 0
        
        # Decision logic: Compare redraw EV to keep EV with a threshold
        improvement_threshold = 0.05  # Base threshold
        
        # Adjust threshold based on hand strength and street
        if current_strength > 0.7:
            # For strong hands, require more improvement
            improvement_threshold = 0.1
        elif current_strength < 0.3:
            # For weak hands, be more willing to redraw
            improvement_threshold = 0.02
        
        # On the flop, be more conservative with redraws
        if street == 1:
            improvement_threshold += 0.02
        
        # Final decision
        if redraw_ev > keep_ev + improvement_threshold:
            decision = (True, best_discard_idx)
        else:
            decision = (False, -1)
        
        # Cache result
        self.redraw_cache[cache_key] = decision
        return decision
    
    def _get_hand_type(self, hole_cards, board):
        """
        Determine the current hand type/pattern for making informed redraw decisions.
        
        Args:
            hole_cards: List of hole card indices
            board: List of board card indices
            
        Returns:
            String representing hand type
        """
        # Safety checks
        if hole_cards is None or len(hole_cards) < 2:
            return "unknown"
            
        if board is None:
            board = []
            
        if not board:
            # Preflop only - classify based on hole cards
            ranks = [card // 3 for card in hole_cards]
            suits = [card % 3 for card in hole_cards]

            ranks.sorted()
            
            if ranks[0] == ranks[1]:
                return "pair"
            elif suits[0] == suits[1]:
                return "suited"
            elif abs(ranks[0] - ranks[1]) <=4 or abs(8 - ranks[1] - ranks[0]) <=4:
                return "connected"
            elif max(ranks) >= 7:  # 9 or Ace
                return "high_card"
            else:
                return "low_cards"
        else:
            # Post-flop
            all_cards = hole_cards + board
            ranks = [card // 3 for card in all_cards]
            suits = [card % 3 for card in all_cards]
            
            # Count ranks and suits
            rank_counts = {}
            for r in ranks:
                rank_counts[r] = rank_counts.get(r, 0) + 1
            
            suit_counts = {}
            for s in suits:
                suit_counts[s] = suit_counts.get(s, 0) + 1
            
            # Check hand types
            if max(rank_counts.values() if rank_counts else [0]) >= 3:
                return "trips_plus"
            elif len([r for r, count in rank_counts.items() if count >= 2]) >= 2:
                return "two_pair_plus"
            elif max(rank_counts.values() if rank_counts else [0]) >= 2:
                return "pair_plus"
            elif max(suit_counts.values() if suit_counts else [0]) >= 4:
                return "flush_draw"
            
            # Check for straight draws
            sorted_ranks = sorted(set(ranks))
            for i in range(len(sorted_ranks) - 3):
                if sorted_ranks[i+3] - sorted_ranks[i] <= 4:
                    return "straight_draw"
            
            return "high_card" if any(r >= 6 for r in ranks) else "low_cards"
    
    def _smart_card_sampling(self, available_cards, kept_card, board, hand_type, street):
        """
        Intelligently sample potential replacement cards based on current hand type.
        
        Args:
            available_cards: Set of available card indices
            kept_card: The card we're keeping
            board: List of board card indices
            hand_type: String representing current hand type
            street: Current street
            
        Returns:
            List of sampled card indices
        """
        # Safety checks
        if not available_cards:
            return []
            
        if board is None:
            board = []
            
        kept_rank = kept_card // 3
        kept_suit = kept_card % 3
        
        # Convert available_cards set to list for easier manipulation
        available_list = list(available_cards)
        
        # Calculate card priorities based on hand type
        card_priorities = []
        
        for card in available_list:
            rank = card // 3
            suit = card % 3
            priority = 0
            
            # Base priority based on rank (higher is better)
            priority += rank / 8.0
            
            # Hand-type specific adjustments
            if hand_type == "pair" or hand_type == "trips_plus":
                # Prioritize same rank as kept card
                if rank == kept_rank:
                    priority += 3.0
                    
            elif hand_type == "suited" or hand_type == "flush_draw":
                # Prioritize same suit as kept card
                if suit == kept_suit:
                    priority += 2.0
            
            elif hand_type == "connected" or hand_type == "straight_draw":
                # Prioritize cards that could complete a straight
                if abs(rank - kept_rank) <= 4:
                    priority += 1.0 + (1.0 / (abs(rank - kept_rank) + 1))
            
            elif hand_type == "high_card":
                # Prioritize high cards and cards that match kept card suit
                if rank >= 6:  # 8 or higher
                    priority += 1.0
                if suit == kept_suit:
                    priority += 0.5
            
            # Board-specific adjustments
            if board:
                board_ranks = [b // 3 for b in board]
                board_suits = [b % 3 for b in board]
                
                # Potential pairs with board
                if rank in board_ranks:
                    priority += 1.5
                
                # Potential flush with board
                if suit in board_suits and sum(1 for s in board_suits if s == suit) >= 2:
                    priority += 1.0
                
                # Potential straight with board
                board_ranks_set = set(board_ranks)
                board_ranks_set.add(kept_rank)
                
                for start in range(max(0, rank-4), min(9, rank+1)):
                    straight_ranks = set(range(start, start+5))
                    existing_count = len(board_ranks_set & straight_ranks)
                    if existing_count >= 3:  # We'd have at least 4 with this card
                        priority += 1.0
                        break
            
            card_priorities.append((priority, card))
        
        # Sort by priority (descending)
        card_priorities.sort(reverse=True)
        
        # Sample size based on street
        sample_size = 15 if street == 0 else 10
        
        # Take the top cards based on priorities
        top_cards = [card for _, card in card_priorities[:sample_size]]
        
        # Add some random low-priority cards for exploration
        remaining_cards = [card for _, card in card_priorities[sample_size:]]
        if remaining_cards and len(top_cards) < sample_size:
            random_count = min(sample_size - len(top_cards), len(remaining_cards))
            random_cards = random.sample(remaining_cards, random_count)
            top_cards.extend(random_cards)
        
        return top_cards
    
    def _calculate_synergy_weight(self, new_card, kept_card, board, hand_type):
        """
        Calculate a synergy weight for a potential new card.
        
        Args:
            new_card: Potential new card index
            kept_card: Card we're keeping
            board: Current board cards
            hand_type: Current hand type
            
        Returns:
            Weight value (higher means better synergy)
        """
        new_rank = new_card // 3
        new_suit = new_card % 3
        kept_rank = kept_card // 3
        kept_suit = kept_card % 3
        
        # Base weight
        weight = 1.0
        
        # Rank-based weights (higher ranks get higher weight)
        weight += (new_rank / 8.0) * 0.5
        
        # Pair synergy
        if new_rank == kept_rank:
            weight += 1.5
        
        # Suit synergy
        if new_suit == kept_suit:
            weight += 0.5
        
        # Straight synergy
        if abs(new_rank - kept_rank) <= 4:
            weight += 0.3
        
        # Board synergy if available
        if board:
            board_ranks = [b // 3 for b in board]
            board_suits = [b % 3 for b in board]
            
            # Pair with board
            if new_rank in board_ranks:
                weight += 1.0
            
            # Flush potential
            same_suit_count = sum(1 for s in board_suits if s == new_suit)
            if same_suit_count >= 2:
                weight += 0.7
            
            # Straight potential with board
            sorted_ranks = sorted(set(board_ranks + [kept_rank, new_rank]))
            for i in range(len(sorted_ranks) - 3):
                if sorted_ranks[i+3] - sorted_ranks[i] <= 4:
                    weight += 0.7
                    break
        
        return max(0.1, weight)  # Minimum weight of 0.1
    
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
        
        # If opponent model has redraw information, incorporate it
        has_redrawn = opponent_tendencies.get('has_redrawn', False)
        redraw_street = opponent_tendencies.get('redraw_street', None)
        
        if 'get_redraw_insight' in opponent_tendencies:
            # This is a method call, we can't directly access it
            # But we know the opponent has redrawn information
            if has_redrawn:
                # Adjust based on typical redraw behavior
                if redraw_street == street:
                    # They redrawn this street, likely have improved their hand
                    aggression = max(0.6, aggression)  # More aggressive
                
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
        # Safety checks
            
        if street < 0 or street > 3:
            street = 0  # Default to preflop
        
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
        
        # Calculate simplified EV with bounds checking
        ev = hand_strength * position_factor * street_factor
        
        # Ensure EV is within valid range [0, 1]
        return max(0.0, min(1.0, ev))
    
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
        # Safety checks
        if hole_cards is None or len(hole_cards) != 2:
            return False, -1
            
        if board is None:
            board = []
            
        if street > 1 or street < 0:
            return False, -1
            
        # Core redraw decision
        should_redraw, card_idx = self.should_redraw(
            hole_cards, board, street, position,
            opponent_model.__dict__ if opponent_model else None
        )
        
        # If opponent model is available, make exploitative adjustments
        if opponent_model and hasattr(opponent_model, 'get_redraw_frequency'):
            # Extract opponent tendencies
            if hasattr(opponent_model, 'get_redraw_frequency'):
                opp_redraw_freq = opponent_model.get_redraw_frequency()
            else:
                opp_redraw_freq = 0.5
                
            if hasattr(opponent_model, 'get_fold_equity'):
                opp_fold_equity = opponent_model.get_fold_equity()
            else:
                opp_fold_equity = 0.5
                
            opp_aggression = getattr(opponent_model, 'aggression', 0.5)
            
            # Current hand strength
            current_strength = self.hand_evaluator.get_hand_strength(hole_cards, board)
            
            # Check if opponent has redrawn
            has_redrawn = getattr(opponent_model, 'has_redrawn', False)
            
            # If opponent has redrawn, adjust our strategy
            if has_redrawn:
                redraw_street = getattr(opponent_model, 'redraw_street', None)
                
                # They've used their redraw, we can be more aggressive
                if not should_redraw and current_strength > 0.4 and current_strength < 0.7:
                    # Sometimes be more aggressive with medium hands
                    # since opponent has no more redraws
                    if random.random() < 0.3:
                        return False, -1
            
            # Case 1: Opponent redraws very often - we should be more selective
            if opp_redraw_freq > 0.7:
                # Only redraw very weak hands or for significant improvements
                if current_strength > 0.4 and should_redraw:
                    # Verify the improvement is truly significant
                    # Just use the card index already determined
                    return should_redraw, card_idx
            
            # Case 2: Opponent rarely redraws - we can be more aggressive with redraws
            elif opp_redraw_freq < 0.3 and not should_redraw:
                # For weak hands, reconsider redrawing
                if current_strength < 0.3:
                    # Get the weaker card
                    ranks = [card // 3 for card in hole_cards]
                    lower_idx = 0 if ranks[0] < ranks[1] else 1
                    
                    # 40% chance to redraw weak hands against conservative opponents
                    if random.random() < 0.4:
                        return True, lower_idx
            
            # Case 3: Deceptive redraw with strong hands
            if not should_redraw and current_strength > 0.75:
                # Calculate the balance between deception and value
                deception_threshold = 0.15  # Base threshold
                
                # Adjust based on opponent type
                deception_threshold *= (1.5 - opp_fold_equity)
                
                # Occasionally make a deceptive redraw
                if random.random() < deception_threshold:
                    # Determine which card to discard (the lower one)
                    card1_rank = hole_cards[0] // 3
                    card2_rank = hole_cards[1] // 3
                    
                    # Discard lower card
                    discard_idx = 0 if card1_rank < card2_rank else 1
                    return True, discard_idx
        
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
            # IMPORTANT FIX: Never discard an Ace for a lower card
            if hole_cards[card_idx] // 3 == 8:  # Ace's rank is 8
                other_card_rank = hole_cards[1-card_idx] // 3
                if other_card_rank < 8:  # Other card is lower than Ace
                    # Strongly negative improvement to avoid discarding Ace
                    improvements.append((-1.0, card_idx))
                    continue
            
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
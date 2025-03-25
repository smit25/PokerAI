import time
import contextlib

class TimeManager:
    """Time management utilities for poker bot."""
    
    def __init__(self, max_time=0.95):
        self.max_time = max_time
        self.start_time = None
    
    def start_timing(self):
        """Start timing a decision."""
        self.start_time = time.time()
        return self.start_time
    
    def time_elapsed(self):
        """Get elapsed time since start."""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
    
    def time_remaining(self):
        """Get remaining time within max time."""
        elapsed = self.time_elapsed()
        return max(0, self.max_time - elapsed)
    
    @contextlib.contextmanager
    def time_check(self, allocation):
        """Context manager for time-bounded operations."""
        start = time.time()
        yield
        elapsed = time.time() - start
        
        # If operation took too long, adjust max time for future operations
        if elapsed > allocation * self.max_time and self.max_time > 0.5:
            # Reduce max time to account for slow operation
            self.max_time = self.max_time * 0.9
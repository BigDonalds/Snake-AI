from typing import Optional, List, Dict, Any
from .run_data import RunData

class ReplayManager:
    """
    Manages replay data - a data provider for the visualizer
    """
    
    def __init__(self, config: Dict[str, Any], game_class, snake_class):
        self.config = config
        self.game_class = game_class
        self.snake_class = snake_class
    
    # These methods just return the run data - the visualizer handles display
    def get_best_run(self, stats) -> Optional[RunData]:
        """Get the best run from stats"""
        return stats.best_run if stats else None
    
    def get_worst_run(self, stats) -> Optional[RunData]:
        """Get the worst run from stats"""
        return stats.worst_run if stats else None
    
    def get_median_run(self, stats) -> Optional[RunData]:
        """Get the median run from stats"""
        return stats.median_run if stats else None
    
    def get_random_run(self, stats) -> Optional[RunData]:
        """Get a random run from stats"""
        if stats and stats.runs:
            import random
            return random.choice(stats.runs)
        return None

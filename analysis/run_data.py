import numpy as np
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

class DeathReason(Enum):
    """Enum for tracking cause of death"""
    WALL = "wall"
    SELF = "self"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    
    @property
    def WALL_COLLISION(self):
        return self.WALL
    
    @property
    def SELF_COLLISION(self):
        return self.SELF
    
    @property
    def STARVATION(self):
        return self.TIMEOUT
    
    @property
    def TRAPPED(self):
        return self.TIMEOUT

class MoveRecord:
    """Records a single move in the snake's life"""
    
    def __init__(self, step: int, direction: str, head_position: Tuple[int, int], 
                 food_position: Tuple[int, int], score: int, snake_length: int,
                 action_scores: Optional[List[float]] = None):
        self.step = step
        self.direction = direction
        self.head_position = head_position
        self.food_position = food_position
        self.score = score
        self.snake_length = snake_length
        self.action_scores = action_scores
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'step': self.step,
            'direction': self.direction,
            'head_x': self.head_position[0],
            'head_y': self.head_position[1],
            'food_x': self.food_position[0],
            'food_y': self.food_position[1],
            'score': self.score,
            'snake_length': self.snake_length,
            'action_scores': self.action_scores,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MoveRecord':
        """Create from dictionary"""
        return cls(
            step=data['step'],
            direction=data['direction'],
            head_position=(data['head_x'], data['head_y']),
            food_position=(data['food_x'], data['food_y']),
            score=data['score'],
            snake_length=data['snake_length'],
            action_scores=data.get('action_scores')
        )

class RunData:
    """
    Complete data for a single agent's run
    """
    
    def __init__(self, agent_id: str, generation: int, genome: np.ndarray = None):
        self.agent_id = agent_id
        self.generation = generation
        self.genome = genome.copy() if genome is not None else None
        
        self.final_score = 0
        self.final_fitness = 0.0
        self.total_steps = 0
        self.death_reason = DeathReason.UNKNOWN
        self.death_position = None
        self.path_efficiency = 0.0
        self.exploration_rate = 0.0
        self.cycle_detected = False
        
        self.death_step = 0
        self.foods_eaten = []
        
        self.moves: List[MoveRecord] = []
        self.start_time = datetime.now()
        self.end_time = None
        
        self.avg_action_confidence = 0.0
        self.cycle_length = 0
        
        self.strategies_used: List[str] = []
        self.milestones_achieved: List[int] = []
    
    def add_move(self, move: MoveRecord) -> None:
        """Record a move"""
        self.moves.append(move)
        self.total_steps = len(self.moves)
    
    def end_run(self, score: int, fitness: float, death_reason: DeathReason, 
                death_position: Optional[Tuple[int, int]] = None) -> None:
        """Mark the run as completed"""
        self.final_score = score
        self.final_fitness = fitness
        self.death_reason = death_reason
        self.death_step = len(self.moves)
        self.death_position = death_position
        self.end_time = datetime.now()
        
        self._calculate_metrics()
    
    def _calculate_metrics(self) -> None:
        """Calculate post-run metrics"""
        if not self.moves:
            return
        
        # Path efficiency (how direct was path to food)
        if self.foods_eaten:
            total_distance = 0
            for i in range(1, len(self.foods_eaten)):
                prev_food = self.foods_eaten[i-1]
                curr_food = self.foods_eaten[i]
                distance = abs(curr_food[0] - prev_food[0]) + abs(curr_food[1] - prev_food[1])
                total_distance += distance
            
            avg_steps_between_food = self.total_steps / len(self.foods_eaten) if self.foods_eaten else 0
            
            if total_distance > 0:
                self.path_efficiency = avg_steps_between_food / (total_distance / len(self.foods_eaten))
            else:
                self.path_efficiency = 1.0
        
        # Exploration rate (unique positions visited)
        unique_positions = set((m.head_position[0], m.head_position[1]) for m in self.moves)
        self.exploration_rate = len(unique_positions) / self.total_steps if self.total_steps > 0 else 0
        
        # Average action confidence
        if self.moves and self.moves[0].action_scores:
            confidences = []
            for move in self.moves:
                if move.action_scores:
                    scores = np.array(move.action_scores)
                    max_score = np.max(scores)
                    others_avg = (np.sum(scores) - max_score) / (len(scores) - 1) if len(scores) > 1 else 0
                    confidence = max_score - others_avg
                    confidences.append(confidence)
            self.avg_action_confidence = np.mean(confidences) if confidences else 0
        
        # Detect cycles
        self._detect_cycles()
    
    def _detect_cycles(self, window_size: int = 20) -> None:
        """Detect if snake got stuck in a cycle"""
        if len(self.moves) < window_size:
            return
        
        recent_positions = [(m.head_position[0], m.head_position[1]) 
                           for m in self.moves[-window_size:]]
        
        position_str = ','.join([f"{x},{y}" for x, y in recent_positions])
        
        for length in range(4, window_size // 2):
            pattern = position_str[:length * 10]
            if position_str.count(pattern) > 2:
                self.cycle_detected = True
                self.cycle_length = length
                break
    
    def add_strategy(self, strategy_name: str) -> None:
        """Record a strategy used"""
        if strategy_name not in self.strategies_used:
            self.strategies_used.append(strategy_name)
    
    def add_milestone(self, score: int) -> None:
        """Record a score milestone achieved"""
        if score not in self.milestones_achieved:
            self.milestones_achieved.append(score)
    
    def add_food_eaten(self, position: Tuple[int, int]) -> None:
        """Record a food eaten"""
        self.foods_eaten.append(position)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'agent_id': self.agent_id,
            'generation': self.generation,
            'genome': self.genome.tolist() if self.genome is not None else None,
            'final_score': self.final_score,
            'final_fitness': self.final_fitness,
            'total_steps': self.total_steps,
            'death_reason': self.death_reason.value,
            'death_position': self.death_position,
            'path_efficiency': self.path_efficiency,
            'exploration_rate': self.exploration_rate,
            'cycle_detected': self.cycle_detected,
            'death_step': self.death_step,
            'moves': [m.to_dict() for m in self.moves],
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'avg_action_confidence': self.avg_action_confidence,
            'cycle_length': self.cycle_length,
            'strategies_used': self.strategies_used,
            'milestones_achieved': self.milestones_achieved,
            'foods_eaten': self.foods_eaten
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunData':
        """Create from dictionary"""
        genome = np.array(data['genome']) if data.get('genome') else None
        run_data = cls(
            agent_id=data['agent_id'],
            generation=data['generation'],
            genome=genome
        )
        
        run_data.final_score = data['final_score']
        run_data.final_fitness = data['final_fitness']
        run_data.total_steps = data['total_steps']
        run_data.death_reason = DeathReason(data['death_reason'])
        run_data.death_position = data['death_position']
        run_data.path_efficiency = data['path_efficiency']
        run_data.exploration_rate = data['exploration_rate']
        run_data.cycle_detected = data['cycle_detected']
        run_data.death_step = data.get('death_step', 0)
        run_data.moves = [MoveRecord.from_dict(m) for m in data.get('moves', [])]
        if 'start_time' in data:
            run_data.start_time = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            run_data.end_time = datetime.fromisoformat(data['end_time'])
        run_data.avg_action_confidence = data.get('avg_action_confidence', 0.0)
        run_data.cycle_length = data.get('cycle_length', 0)
        run_data.strategies_used = data.get('strategies_used', [])
        run_data.milestones_achieved = data.get('milestones_achieved', [])
        run_data.foods_eaten = data.get('foods_eaten', [])
        
        return run_data
    
    def get_summary(self) -> str:
        """Get a human-readable summary"""
        summary = f"""
╔══════════════════════════════════════════╗
║  Agent: {self.agent_id}
║  Generation: {self.generation}
╠══════════════════════════════════════════╣
║  Final Score: {self.final_score}
║  Final Fitness: {self.final_fitness:.2f}
║  Total Steps: {self.total_steps}
║  Death Reason: {self.death_reason.value}
╠══════════════════════════════════════════╣
║  Metrics:
║    Path Efficiency: {self.path_efficiency:.2f}
║    Exploration Rate: {self.exploration_rate:.2f}
║    Cycle Detected: {self.cycle_detected}
╚══════════════════════════════════════════╝
"""
        return summary

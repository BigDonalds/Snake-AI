import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
from .run_data import RunData, DeathReason

class GenerationStats:
    """
    Statistical analysis of a complete generation
    """
    
    def __init__(self, generation: int, population_size: int):
        self.generation = generation
        self.population_size = population_size
        self.runs: List[RunData] = []
        
        # Summary statistics - these are what the visualizer needs
        self.best_run: Optional[RunData] = None
        self.worst_run: Optional[RunData] = None
        self.median_run: Optional[RunData] = None
        
        # Score statistics
        self.score_values: List[int] = []
        self.avg_score = 0.0
        self.max_score = 0
        self.min_score = 0
        
        # Fitness statistics
        self.fitness_values: List[float] = []
        self.avg_fitness = 0.0
        self.max_fitness = 0.0
        self.min_fitness = 0.0
    
    def add_run(self, run_data: RunData) -> None:
        """Add a run to the generation"""
        self.runs.append(run_data)
    
    def finalize(self) -> None:
        """Calculate all statistics after all runs are added"""
        if not self.runs:
            return
        
        # Extract values
        self.score_values = [r.final_score for r in self.runs]
        self.fitness_values = [r.final_fitness for r in self.runs]
        
        # Basic statistics
        self.avg_score = np.mean(self.score_values)
        self.max_score = int(np.max(self.score_values))
        self.min_score = int(np.min(self.score_values))
        
        self.avg_fitness = np.mean(self.fitness_values)
        self.max_fitness = np.max(self.fitness_values)
        self.min_fitness = np.min(self.fitness_values)
        
        # Sort runs by fitness
        sorted_by_fitness = sorted(self.runs, key=lambda x: x.final_fitness, reverse=True)
        
        self.best_run = sorted_by_fitness[0]
        self.worst_run = sorted_by_fitness[-1]
        self.median_run = sorted_by_fitness[len(sorted_by_fitness) // 2]
    
    def get_summary(self) -> str:
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    GENERATION {self.generation:3d} STATISTICS                    ║
╠══════════════════════════════════════════════════════════════╣
║  Population: {self.population_size} individuals
║  Runs Recorded: {len(self.runs)}
╠══════════════════════════════════════════════════════════════╣
║  📊 PERFORMANCE
║     Best Score:   {self.max_score:3d} (Fitness: {self.max_fitness:.2f})
║     Avg Score:    {self.avg_score:.2f}
║     Best Fitness: {self.max_fitness:.2f}
║     Avg Fitness:  {self.avg_fitness:.2f}
╠══════════════════════════════════════════════════════════════╣
║  🏆 BEST AGENT
║     Score: {self.best_run.final_score} | Fitness: {self.best_run.final_fitness:.2f}
║     Agent: {self.best_run.agent_id}
║     Death: {self.best_run.death_reason}
╠══════════════════════════════════════════════════════════════╣
║  💀 WORST AGENT
║     Score: {self.worst_run.final_score} | Fitness: {self.worst_run.final_fitness:.2f}
║     Agent: {self.worst_run.agent_id}
║     Death: {self.worst_run.death_reason}
╚══════════════════════════════════════════════════════════════╝
"""
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'generation': self.generation,
            'population_size': self.population_size,
            'runs_count': len(self.runs),
            'avg_fitness': self.avg_fitness,
            'max_fitness': self.max_fitness,
            'min_fitness': self.min_fitness,
            'avg_score': self.avg_score,
            'max_score': self.max_score,
            'min_score': self.min_score,
            'best_agent_id': self.best_run.agent_id if self.best_run else None,
            'best_agent_score': self.best_run.final_score if self.best_run else None,
            'best_agent_fitness': self.best_run.final_fitness if self.best_run else None,
            'worst_agent_id': self.worst_run.agent_id if self.worst_run else None,
            'worst_agent_score': self.worst_run.final_score if self.worst_run else None,
        }

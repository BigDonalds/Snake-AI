import numpy as np
import random
from collections import deque
from typing import List, Optional, Tuple, Dict, Any, Set
from dataclasses import dataclass, field
import json
import os
import time
from enum import Enum
import sys

from genetic_ai.neural_network import ModularNeuralNetwork, Activation, EvolutionaryOptimizer
from analysis.run_data import RunData, MoveRecord, DeathReason
from analysis.generation_stats import GenerationStats
from memory.ancestors_memory import AncestorsMemory

DEBUG_GA = False

def debug_ga(msg, level="INFO"):
    """Debug print with [GA] prefix"""
    if DEBUG_GA:
        print(f"[GA] {msg}", file=sys.stderr, flush=True)

class BehaviorProfiler:
    """Track and analyze individual behaviors in detail"""
    
    def __init__(self):
        self.behavior_logs = []
        self.death_patterns = {}
        self.movement_patterns = {}
        self.food_approaches = []
        self.oscillation_history = []
        self.wall_hugging_history = []
        
    def log_episode(self, individual, game, moves):
        """Log detailed episode data"""
        # Analyze movement patterns
        positions = game.position_history if hasattr(game, 'position_history') else []
        if len(positions) > 10:
            # Calculate direction changes
            directions = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                if dx > 0: directions.append('right')
                elif dx < 0: directions.append('left')
                elif dy > 0: directions.append('down')
                elif dy < 0: directions.append('up')
            
            # Detect oscillation (back-and-forth movement)
            oscillations = 0
            for i in range(2, len(directions)):
                if directions[i] == directions[i-2] and directions[i] != directions[i-1]:
                    oscillations += 1
            
            # Detect wall hugging
            wall_hugging = 0
            for pos in positions[-20:]:
                x, y = pos
                if x < 100 or x > 650 or y < 100 or y > 650:
                    wall_hugging += 1
            wall_hugging /= max(len(positions[-20:]), 1)
            
            # Detect loops
            recent = positions[-30:]
            loop_detected = len(set(recent)) < len(recent) * 0.7
            
            self.oscillation_history.append(oscillations)
            self.wall_hugging_history.append(wall_hugging)
            
            self.movement_patterns[individual.id] = {
                'oscillations': oscillations,
                'wall_hugging': wall_hugging,
                'loop_detected': loop_detected,
                'unique_ratio': len(set(positions)) / len(positions),
                'direction_frequency': self._direction_freq(directions),
                'total_steps': len(positions)
            }
        
        # Analyze death patterns
        if individual.death_reason:
            death_type = str(individual.death_reason)
            snake_length = len(game.snake.get_body_positions()) if hasattr(game.snake, 'get_body_positions') else 1
            
            if death_type not in self.death_patterns:
                self.death_patterns[death_type] = {'count': 0, 'lengths': []}
            self.death_patterns[death_type]['count'] += 1
            self.death_patterns[death_type]['lengths'].append(snake_length)
        
        # Analyze food approach behavior
        if hasattr(game, 'move_history'):
            food_approaches = []
            for move in game.move_history[-20:]:
                if move.get('food_eaten', False):
                    # Check what led to eating food
                    prev_moves = game.move_history[max(0, move['step']-5):move['step']]
                    food_approaches.append({
                        'steps_before': len(prev_moves),
                        'directions': [m.get('direction') for m in prev_moves]
                    })
            if food_approaches:
                self.food_approaches.append({
                    'individual': individual.id,
                    'score': individual.score,
                    'approaches': food_approaches
                })
    
    def _direction_freq(self, directions):
        freq = {'up':0, 'down':0, 'left':0, 'right':0}
        for d in directions:
            freq[d] += 1
        total = len(directions)
        return {k: v/total for k,v in freq.items()}
    
    def print_summary(self):
        """Print behavioral summary"""
        print("\n" + "="*80)
        print("🔍 BEHAVIORAL ANALYSIS")
        print("="*80)
        
        # Movement patterns
        if self.movement_patterns:
            avg_oscillations = np.mean([p['oscillations'] for p in self.movement_patterns.values()])
            avg_wall_hugging = np.mean([p['wall_hugging'] for p in self.movement_patterns.values()])
            loop_rate = sum(1 for p in self.movement_patterns.values() if p['loop_detected']) / len(self.movement_patterns)
            
            print(f"\n📊 Movement Patterns:")
            print(f"   Average oscillations: {avg_oscillations:.1f} (high = indecisive)")
            print(f"   Wall hugging: {avg_wall_hugging:.2%} (high = afraid of center)")
            print(f"   Loop detection: {loop_rate:.2%} (high = stuck in cycles)")
            print(f"   Avg unique positions: {np.mean([p['unique_ratio'] for p in self.movement_patterns.values()]):.2%}")
            
            # Recent trends
            if len(self.oscillation_history) > 10:
                recent_osc = np.mean(self.oscillation_history[-10:])
                recent_wall = np.mean(self.wall_hugging_history[-10:])
                print(f"   Recent oscillations (last 10): {recent_osc:.1f}")
                print(f"   Recent wall hugging (last 10): {recent_wall:.2%}")
        
        # Death patterns
        if self.death_patterns:
            print(f"\n💀 Death Analysis:")
            total_deaths = sum(d['count'] for d in self.death_patterns.values())
            for death_type, data in sorted(self.death_patterns.items(), key=lambda x: x[1]['count'], reverse=True):
                avg_length = np.mean(data['lengths'])
                percentage = data['count'] / total_deaths * 100
                print(f"   {death_type}: {data['count']} ({percentage:.1f}%), avg length {avg_length:.1f}")
        
        # Food approach analysis
        if self.food_approaches:
            all_approaches = []
            scores = []
            for entry in self.food_approaches:
                all_approaches.extend(entry['approaches'])
                scores.append(entry['score'])
            
            if all_approaches:
                avg_steps_before = np.mean([a['steps_before'] for a in all_approaches])
                print(f"\n🍎 Food Approach:")
                print(f"   Average steps before eating: {avg_steps_before:.1f}")
                print(f"   Total food events: {len(all_approaches)}")
                print(f"   Avg score of food-finders: {np.mean(scores):.1f}")
        
        print("="*80)
    
    def save_analysis(self, filename="behavior_analysis.json"):
        """Save behavioral analysis to file"""
        data = {
            'timestamp': time.time(),
            'movement_patterns': self.movement_patterns,
            'death_patterns': self.death_patterns,
            'food_approaches': self.food_approaches[-100:],
            'oscillation_history': self.oscillation_history[-100:],
            'wall_hugging_history': self.wall_hugging_history[-100:]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"📊 Saved behavior analysis to {filename}")


class SelectionMethod(Enum):
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    BOLTZMANN = "boltzmann"
    TRUNCATION = "truncation"

class ReplacementStrategy(Enum):
    GENERATIONAL = "generational"
    STEADY_STATE = "steady_state"
    ELITIST = "elitist"
    MU_PLUS_LAMBDA = "mu_plus_lambda"
    MU_COMMA_LAMBDA = "mu_comma_lambda"

@dataclass
class Individual:
    genome: np.ndarray
    fitness: float = 0.0
    score: int = 0
    age: int = 0
    generation: int = 0
    species_id: int = 0
    network: Optional['ModularNeuralNetwork'] = None
    id: str = field(default_factory=lambda: f"ind_{random.randint(10000, 99999)}")
    total_steps: int = 0
    best_score: int = 0
    parent_ids: List[str] = field(default_factory=list)
    novelty: float = 0.0
    complexity: float = 0.0
    birth_time: float = field(default_factory=time.time)
    
    # Behavioral characteristics
    behavior_vector: Optional[np.ndarray] = None
    exploration_rate: float = 0.0
    food_efficiency: float = 0.0
    survival_time: float = 0.0
    
    # Fitness components for multi-objective
    fitness_components: Dict[str, float] = field(default_factory=dict)
    
    # Visualizer data
    move_history: List[Dict] = field(default_factory=list)
    death_reason: Optional[Any] = None
    death_position: Optional[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.network is not None:
            try:
                self.network.set_genome(self.genome)
                self.complexity = self._calculate_complexity()
            except Exception as e:
                print(f"⚠️ Genome size mismatch for {self.id}: {e}, creating new genome")
                self.genome = self.network.get_genome()
    
    def _calculate_complexity(self) -> float:
        """Calculate network complexity (number of non-zero weights)"""
        if self.network is None:
            return 0.0
        non_zero = sum(np.sum(np.abs(w) > 1e-6) for w in self.network.weights)
        total = sum(w.size for w in self.network.weights)
        return non_zero / max(total, 1)
    
    def clone(self) -> 'Individual':
        return Individual(
            genome=self.genome.copy(),
            fitness=self.fitness,
            score=self.score,
            age=self.age,
            generation=self.generation,
            species_id=self.species_id,
            network=self.network,
            id=f"{self.id}_clone_{random.randint(1000, 9999)}",
            total_steps=self.total_steps,
            best_score=self.best_score,
            parent_ids=self.parent_ids.copy(),
            novelty=self.novelty,
            complexity=self.complexity,
            behavior_vector=self.behavior_vector.copy() if self.behavior_vector is not None else None,
            exploration_rate=self.exploration_rate,
            food_efficiency=self.food_efficiency,
            survival_time=self.survival_time,
            fitness_components=self.fitness_components.copy(),
            move_history=self.move_history.copy() if self.move_history else [],
            death_reason=self.death_reason,
            death_position=self.death_position
        )

class Species:
    def __init__(self, species_id: int, representative: np.ndarray):
        self.id = species_id
        self.representative = representative
        self.members: List[Individual] = []
        self.best_fitness = 0.0
        self.best_score = 0
        self.stagnation_counter = 0
        self.generation_born = 0
        self.fitness_history: List[float] = []
        self.score_history: List[int] = []
        self.age = 0
        
    def add_member(self, individual: Individual):
        self.members.append(individual)
        if individual.fitness > self.best_fitness:
            self.best_fitness = individual.fitness
            self.fitness_history.append(individual.fitness)
            self.stagnation_counter = 0
        if individual.score > self.best_score:
            self.best_score = individual.score
            self.score_history.append(individual.score)
    
    def get_adjusted_fitness(self, individual: Individual) -> float:
        """Calculate fitness adjusted for species sharing"""
        if len(self.members) == 0:
            return individual.fitness
        
        # Distance to representative
        dist = np.linalg.norm(individual.genome - self.representative)
        
        # Sharing function
        sigma_share = 5.0
        if dist < sigma_share:
            sharing = 1 - (dist / sigma_share)
        else:
            sharing = 0
        
        return individual.fitness / (len(self.members) * (1 - sharing) + 1)


class NeuralGeneticAlgorithm:
    def __init__(self, 
                population_size: int = 100,
                input_size: int = 37,
                elite_size: int = 5,
                mutation_rate: float = 0.15,
                crossover_rate: float = 0.7,
                tournament_size: int = 3,
                compatibility_threshold: float = 3.0,
                use_speciation: bool = True,
                selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
                replacement_strategy: ReplacementStrategy = ReplacementStrategy.ELITIST,
                archive_size: int = 500,
                novelty_weight: float = 0.1):
        
        debug_ga(f"Initializing Genetic Algorithm:")
        debug_ga(f"  population_size={population_size}, input_size={input_size}")
        debug_ga(f"  elite_size={elite_size}, mutation_rate={mutation_rate}, crossover_rate={crossover_rate}")
        
        self.population_size = population_size
        self.input_size = input_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.compatibility_threshold = compatibility_threshold
        self.use_speciation = use_speciation
        self.selection_method = selection_method
        self.replacement_strategy = replacement_strategy
        self.archive_size = archive_size
        self.novelty_weight = novelty_weight
        
        self.target_species = max(5, population_size // 15)
        
        self.population: List[Individual] = []
        self.species: List[Species] = []
        self.generation = 0
        self.total_evaluations = 0
        self.current_individual_idx = 0
        
        # History tracking
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.best_score_history: List[int] = []
        self.diversity_history: List[float] = []
        self.species_count_history: List[int] = []
        
        # Memory systems
        self.hall_of_fame: deque = deque(maxlen=50)
        self.novelty_archive: List[np.ndarray] = []
        self.elite_archive: List[Individual] = []
        self.behavior_archive: List[Tuple[np.ndarray, float]] = []
        
        # Best ever tracking
        self.best_ever_score = 0
        self.best_ever_fitness = 0.0
        self.best_ever_genome: Optional[np.ndarray] = None
        self.best_ever_behavior: Optional[np.ndarray] = None
        
        # Stagnation tracking
        self.stagnation_counter = 0
        self.last_best_score = 0
        self.generations_since_improvement = 0
        self.mutation_rate_adaptation = 1.0
        
        # Champion lineage
        self.champion_lineage: List[Dict] = []
        
        # Dynamic parameter adjustment
        self.adaptation_rate = 0.05
        self.min_mutation_rate = 0.05
        self.max_mutation_rate = 0.3
        
        # Visualizer data
        self.generation_stats: List[GenerationStats] = []
        
        # Store expected genome size
        self.expected_genome_size = None
        
        # Diagnostic tools
        self.behavior_profiler = BehaviorProfiler()
        self.start_time = time.time()
        self.last_analysis_gen = 0
        self.debug_mode = True
        
        # Create initial population
        self.create_initial_population()
        
        # Try to load best ever genome
        self._try_load_best_ever()
        
        # Initialize ancestral memory with current state
        try:
            memory = AncestorsMemory("memory/snake_ancestry.json")
            memory.save_current_state(
                generation=self.generation,
                best_ever_score=self.best_ever_score,
                best_ever_fitness=self.best_ever_fitness,
                best_ever_genome=self.best_ever_genome,
                best_fitness_history=self.best_fitness_history,
                avg_fitness_history=self.avg_fitness_history,
                best_score_history=self.best_score_history,
                diversity_history=self.diversity_history
            )
            print(f"📀 Ancestral memory initialized - Generation {self.generation}, Best Score: {self.best_ever_score}")
        except Exception as e:
            print(f"Note: Could not initialize ancestral memory: {e}")

    def create_initial_population(self):
        """Create diverse initial population with multiple initialization strategies"""
        debug_ga("Creating initial population...")
        strategies = [
            'he', 'xavier', 'small', 'large', 'orthogonal',
            'sparse', 'uniform', 'normal', 'truncated_normal'
        ]
        
        # Create template network to get genome size
        template_network = self._create_network_architecture()
        self.expected_genome_size = len(template_network.get_genome())
        debug_ga(f"  Expected genome size: {self.expected_genome_size}")
        
        # Validate champion genome if exists
        champion_valid = self.best_ever_genome is not None and len(self.best_ever_genome) == self.expected_genome_size
        
        for i in range(self.population_size):
            # First individual uses champion if available
            if champion_valid and i == 0:
                debug_ga(f"  Individual {i}: Using champion")
                individual = self._create_champion_individual({'genome': self.best_ever_genome})
                if individual:
                    # Ensure champion genome has correct size
                    if len(individual.genome) != self.expected_genome_size:
                        debug_ga(f"  Fixing champion genome size: {len(individual.genome)} -> {self.expected_genome_size}")
                        individual.genome = self._fix_genome_size(individual.genome, self.expected_genome_size)
                        individual.network.set_genome(individual.genome)
                    self.population.append(individual)
                    continue
            
            # Create new individual with strategy
            strategy = strategies[i % len(strategies)]
            debug_ga(f"  Individual {i}: Strategy={strategy}")
            individual = self._create_individual_with_strategy(strategy, i)
            self.population.append(individual)
        
        print(f"✅ Created initial population of {len(self.population)} individuals")
    
    def _fix_genome_size(self, genome: np.ndarray, target_size: int) -> np.ndarray:
        """Fix genome size to match target"""
        if genome is None:
            # Create new genome if None
            return np.random.randn(target_size).astype(np.float32) * 0.1
        
        if len(genome) == target_size:
            return genome
        elif len(genome) < target_size:
            # Pad with small random values instead of zeros to maintain gradient flow
            padding = np.random.randn(target_size - len(genome)).astype(np.float32) * 0.01
            return np.concatenate([genome, padding])
        else:
            # Truncate
            return genome[:target_size]
    
    def _create_network_architecture(self) -> ModularNeuralNetwork:
        """Create optimal network architecture for snake"""
        debug_ga(f"Creating network architecture with input_size={self.input_size}")
        network = ModularNeuralNetwork(self.input_size, 4)
        
        network.add_dense_layer(32, Activation.LEAKY_RELU, dropout_rate=0.0, use_batch_norm=False)
        network.add_dense_layer(16, Activation.RELU, dropout_rate=0.0, use_batch_norm=False)
        
        network.build()
        
        # Calculate total parameters
        total_params = 0
        for w in network.weights:
            total_params += w.size
        for b in network.biases:
            total_params += b.size
        
        # Store the expected genome size
        self.expected_genome_size = total_params
        
        # Log architecture details
        debug_ga(f"  Network built with {len(network.weights)} weight layers")
        debug_ga(f"  Total parameters: {total_params}")
        
        return network
    
    def _create_individual_with_strategy(self, strategy: str, index: int) -> Individual:
        """Create individual using specific initialization strategy"""
        network = self._create_network_architecture()
        
        # Get the base genome first
        genome = network.get_genome()
        
        if strategy == 'he':
            # He initialization already in build
            pass
        elif strategy == 'xavier':
            for j in range(len(network.weights)):
                fan_in = network.weights[j].shape[0]
                fan_out = network.weights[j].shape[1]
                scale = np.sqrt(2.0 / (fan_in + fan_out))
                network.weights[j] = np.random.randn(fan_in, fan_out) * scale
        elif strategy == 'small':
            for j in range(len(network.weights)):
                network.weights[j] = network.weights[j] * 0.1
        elif strategy == 'large':
            for j in range(len(network.weights)):
                network.weights[j] = network.weights[j] * 2.0
        elif strategy == 'orthogonal':
            for j in range(len(network.weights)):
                rows, cols = network.weights[j].shape
                if rows >= cols:
                    u, _, v = np.linalg.svd(np.random.randn(rows, cols), full_matrices=False)
                    network.weights[j] = u @ v
                else:
                    u, _, v = np.linalg.svd(np.random.randn(cols, rows), full_matrices=False)
                    network.weights[j] = (u @ v).T
        elif strategy == 'sparse':
            for j in range(len(network.weights)):
                mask = np.random.random(network.weights[j].shape) < 0.1
                network.weights[j] = network.weights[j] * mask * 10
        elif strategy == 'uniform':
            for j in range(len(network.weights)):
                limit = np.sqrt(6 / (network.weights[j].shape[0] + network.weights[j].shape[1]))
                network.weights[j] = np.random.uniform(-limit, limit, network.weights[j].shape)
        elif strategy == 'normal':
            for j in range(len(network.weights)):
                network.weights[j] = np.random.randn(*network.weights[j].shape) * 0.1
        elif strategy == 'truncated_normal':
            for j in range(len(network.weights)):
                values = np.random.randn(*network.weights[j].shape) * 0.1
                values = np.clip(values, -0.2, 0.2)
                network.weights[j] = values
        
        # Get updated genome
        genome = network.get_genome()
        
        # Ensure genome size matches expected
        if self.expected_genome_size and len(genome) != self.expected_genome_size:
            genome = self._fix_genome_size(genome, self.expected_genome_size)
            network.set_genome(genome)
        
        return Individual(
            genome=genome,
            generation=0,
            species_id=index % 10,
            network=network,
            id=f"init_{strategy}_{index}"
        )
    
    def _validate_champion_genome(self, expected_size: int) -> bool:
        """Validate champion genome size"""
        if self.best_ever_genome is None:
            return False
        
        if len(self.best_ever_genome) != expected_size:
            debug_ga(f"  ⚠️ Champion genome size mismatch: {len(self.best_ever_genome)} vs {expected_size}")
            self.best_ever_genome = None
            return False
        
        return True
    
    def _try_load_best_ever(self):
        """Load best ever genome from memory"""
        try:
            if os.path.exists("memory/snake_ancestry.json"):
                with open("memory/snake_ancestry.json", 'r') as f:
                    data = json.load(f)
                    if 'hall_of_fame' in data and data['hall_of_fame']:
                        best_champ = max(data['hall_of_fame'], key=lambda x: x.get('score', 0))
                        if best_champ['score'] > self.best_ever_score:
                            self.best_ever_score = best_champ['score']
                            self.best_ever_fitness = best_champ.get('fitness', 0)
                            if best_champ.get('genome'):
                                self.best_ever_genome = np.array(best_champ['genome'])
                                debug_ga(f"🏆 Loaded champion with score {self.best_ever_score}")
        except Exception as e:
            print(f"Note: Could not load champion: {e}")
    
    def _create_champion_individual(self, champ_data: Dict) -> Optional[Individual]:
        """Create individual from champion data"""
        try:
            network = self._create_network_architecture()
            genome = champ_data['genome']
            
            # Fix genome size if needed
            if self.expected_genome_size and len(genome) != self.expected_genome_size:
                debug_ga(f"  Champion genome size mismatch: {len(genome)} vs {self.expected_genome_size}")
                genome = self._fix_genome_size(genome, self.expected_genome_size)
            
            network.set_genome(genome)
            
            individual = Individual(
                genome=genome,
                generation=self.generation,
                species_id=0,
                network=network,
                id=f"champion_gen{champ_data.get('generation', 0)}"
            )
            individual.fitness = champ_data.get('fitness', 0)
            individual.score = champ_data.get('score', 0)
            individual.best_score = individual.score
            
            debug_ga(f"  Created champion individual with score={individual.score}")
            return individual
        except Exception as e:
            debug_ga(f"  Error creating champion individual: {e}")
            return None

    def get_state(self, game) -> np.ndarray:
        debug_ga(f"get_state called for individual {self.current_individual_idx}")
        
        state = game.get_state()
        head_x, head_y = state['head']
        food_x, food_y = state['food']
        width, height = state['game_width'], state['game_height']
        square_size = state['square_size']
        
        grid_width = width // square_size
        grid_height = height // square_size
        
        # Define grid coordinates
        head_grid_x = head_x // square_size
        head_grid_y = head_y // square_size
        food_grid_x = food_x // square_size
        food_grid_y = food_y // square_size
        
        debug_ga(f"  Game state: head=({head_x},{head_y}), food=({food_x},{food_y})")
        debug_ga(f"  Grid: {grid_width}x{grid_height}, square_size={square_size}")
        
        features = []
        max_dist = max(grid_width, grid_height)
        
        # ========== 1. WALL DISTANCES (4 features) ==========
        features.extend([
            head_grid_y / max_dist,                     # distance to top
            (grid_height - 1 - head_grid_y) / max_dist, # distance to bottom
            head_grid_x / max_dist,                     # distance to left
            (grid_width - 1 - head_grid_x) / max_dist   # distance to right
        ])
        
        # ========== 2. DANGER AHEAD (1 feature) ==========
        current_dir = state['direction']
        dir_vectors = {
            'up': (0, -square_size),
            'down': (0, square_size),
            'left': (-square_size, 0),
            'right': (square_size, 0)
        }
        
        dx, dy = dir_vectors.get(current_dir, (0, 0))
        
        steps_to_wall = 0
        check_x, check_y = head_x + dx, head_y + dy
        while 0 <= check_x < width and 0 <= check_y < height:
            if [check_x, check_y] in state['snake'][1:]:
                break
            steps_to_wall += 1
            check_x += dx
            check_y += dy
        
        if steps_to_wall == 0:
            steps_to_wall = max_dist
        
        features.append(min(steps_to_wall / max_dist, 1.0))
        
        # ========== 3. FREE CELLS AHEAD (3 features) ==========
        for step in range(1, 4):
            check_x = head_x + dx * step
            check_y = head_y + dy * step
            if 0 <= check_x < width and 0 <= check_y < height:
                is_free = [check_x, check_y] not in state['snake'][1:]
                features.append(1.0 if is_free else 0.0)
            else:
                features.append(0.0)
        
        # ========== 4. CARDINAL WALL DISTANCES (4 features) ==========
        for ddx, ddy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            dist = 0
            check_x = head_x + ddx * square_size
            check_y = head_y + ddy * square_size
            while 0 <= check_x < width and 0 <= check_y < height:
                dist += 1
                check_x += ddx * square_size
                check_y += ddy * square_size
            features.append(min(dist / max_dist, 1.0))
        
        # ========== 5. DIRECTIONAL DANGER (4 features - only cardinal) ==========
        cardinal_dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        for dx_dir, dy_dir in cardinal_dirs:
            distance = 0
            x, y = head_x, head_y
            danger_found = False
            
            while True:
                x += dx_dir * square_size
                y += dy_dir * square_size
                distance += 1
                
                if x < 0 or x >= width or y < 0 or y >= height:
                    danger_found = True
                    break
                if [x, y] in state['snake'][1:]:
                    danger_found = True
                    break
                if distance > max(grid_width, grid_height):
                    break
            
            if danger_found:
                features.append(1.0 - (distance / max(grid_width, grid_height)))
            else:
                features.append(0.0)
        
        # ========== 6. FOOD INFORMATION (4 features) ==========
        food_dx = (food_x - head_x) / width
        food_dy = (food_y - head_y) / height
        food_dist = np.sqrt(food_dx**2 + food_dy**2)
        food_angle = np.arctan2(food_dy, food_dx) / np.pi
        
        features.extend([food_dx, food_dy, food_dist, food_angle])
        
        # ========== 7. BODY HEAT MAP - 3x3 grid (9 features) ==========
        for dy_grid in range(-1, 2):
            for dx_grid in range(-1, 2):
                check_x = head_x + dx_grid * square_size
                check_y = head_y + dy_grid * square_size
                
                if 0 <= check_x < width and 0 <= check_y < height:
                    if [check_x, check_y] in state['snake'][1:]:
                        dist = np.sqrt(dx_grid**2 + dy_grid**2)
                        features.append(1.0 / (dist + 1))
                    elif [check_x, check_y] == [head_x, head_y]:
                        features.append(0.0)
                    else:
                        features.append(0.0)
                else:
                    features.append(1.0)  # Wall counts as danger
        
        # ========== 8. SNAKE PROPERTIES (2 features) ==========
        snake_length = len(state['snake'])
        max_possible_length = grid_width * grid_height
        length_ratio = snake_length / max_possible_length
        
        free_space_ratio = (max_possible_length - snake_length) / max_possible_length
        
        features.extend([length_ratio, free_space_ratio])
        
        # ========== 9. CURRENT DIRECTION (4 features) ==========
        dir_map = {'up': [1,0,0,0], 'down': [0,1,0,0], 
                'left': [0,0,1,0], 'right': [0,0,0,1]}
        features.extend(dir_map.get(state['direction'], [0,0,0,0]))
        
        # ========== 10. MANHATTAN DISTANCE (1 feature) ==========
        manhattan_dist = abs(head_grid_x - food_grid_x) + abs(head_grid_y - food_grid_y)
        max_manhattan = grid_width + grid_height
        features.append(manhattan_dist / max_manhattan if max_manhattan > 0 else 1.0)
        
        # ========== 11. STEPS WITHOUT FOOD (1 feature) ==========
        steps_without_food = getattr(game, 'steps_without_food', 0)
        if isinstance(steps_without_food, list):
            steps_without_food = steps_without_food[-1] if steps_without_food else 0
        elif steps_without_food is None:
            steps_without_food = 0
        
        max_steps = max(snake_length * 10, 1)
        features.append(min(steps_without_food / max_steps, 1.0))
                
        # Feature count breakdown:
        # 1. Wall distances: 4
        # 2. Danger ahead: 1 = 5
        # 3. Free cells ahead: 3 = 8
        # 4. Cardinal wall distances: 4 = 12
        # 5. Directional danger: 4 = 16
        # 6. Food info: 4 = 20
        # 7. Body heat map: 9 = 29
        # 8. Snake properties: 2 = 31
        # 9. Current direction: 4 = 35
        # 10. Manhattan: 1 = 36
        # 11. Steps without food: 1 = 37
        
        debug_ga(f"  Features generated: count={len(features)}")
        
        # Final verification
        if len(features) != 37:
            debug_ga(f"  ⚠️ WARNING: Expected 37 features, got {len(features)}")
            # Pad or truncate to fix
            if len(features) < 37:
                debug_ga(f"  Padding to 37 features")
                features.extend([0.0] * (37 - len(features)))
            else:
                debug_ga(f"  Truncating to 37 features")
                features = features[:37]
        
        features_array = np.array(features, dtype=np.float32)
        
        # Print first few features for debugging
        debug_ga(f"  First 5 features: {features_array[:5]}")
        debug_ga(f"  Final features: shape={features_array.shape}, len={len(features_array)}")
        
        return features_array
    
    def get_action(self, game) -> str:
        """Get action with epsilon-greedy exploration"""
        debug_ga(f"=== get_action called for individual {self.current_individual_idx} ===")
        
        if self.current_individual_idx >= len(self.population):
            debug_ga(f"  ⚠️ Index out of range: {self.current_individual_idx} >= {len(self.population)}, resetting to 0")
            self.current_individual_idx = 0
        
        individual = self.population[self.current_individual_idx]
        debug_ga(f"  Individual ID: {individual.id}, Age: {individual.age}, Score: {individual.score}")
        
        state = self.get_state(game)
        
        # Epsilon-greedy with decay
        epsilon = max(0.05, 1.0 - (self.generation / 100))
        
        if random.random() < epsilon:
            action = random.choice(['up', 'down', 'left', 'right'])
            debug_ga(f"  Random exploration action: {action}")
            return action
        
        try:
            probs = individual.network.predict(state, temperature=0.3)
            action_idx = np.argmax(probs)
            action = ['up', 'down', 'left', 'right'][action_idx]
            debug_ga(f"  Network action: {action} (idx={action_idx}, probs={probs})")
            return action
        except Exception as e:
            debug_ga(f"  ❌ Error getting action from network: {e}")
            return random.choice(['up', 'down', 'left', 'right'])
    
    def _store_move_history(self, individual: Individual, game):
        """Store move history from game to individual"""
        if hasattr(game, 'move_history') and game.move_history:
            individual.move_history = []
            for move in game.move_history:
                move_copy = {
                    'step': move.get('step', 0),
                    'direction': move.get('direction', 'right'),
                    'head_position': move.get('head_position', (0, 0)),
                    'food_position': move.get('food_position', (0, 0)),
                    'score': move.get('score', 0),
                    'snake_length': move.get('snake_length', 1),
                    'action_scores': move.get('action_scores'),
                    'food_eaten': move.get('food_eaten', False)
                }
                individual.move_history.append(move_copy)
            debug_ga(f"  Stored {len(individual.move_history)} moves from game history")
        elif hasattr(game, 'steps') and hasattr(game, 'snake') and game.steps > 0:
            individual.move_history = []
            try:
                head_pos = game.snake.get_head_position() if hasattr(game.snake, 'get_head_position') else (0, 0)
                food_pos = getattr(game, 'food_position', (0, 0))
                snake_length = len(game.snake.get_body_positions()) if hasattr(game.snake, 'get_body_positions') else 1
                
                move_copy = {
                    'step': 0,
                    'direction': getattr(game, 'last_direction', 'right'),
                    'head_position': head_pos,
                    'food_position': food_pos,
                    'score': getattr(game, 'score', 0),
                    'snake_length': snake_length,
                    'action_scores': None,
                    'food_eaten': False
                }
                individual.move_history.append(move_copy)
                debug_ga(f"  Created minimal move history (1 move)")
            except Exception as e:
                debug_ga(f"  ⚠️ Could not store minimal move history: {e}")

    def end_generation(self):
        """End the current generation and collect statistics"""
        debug_ga(f"=== ENDING GENERATION {self.generation} ===")
        stats = GenerationStats(self.generation, self.population_size)
        
        for individual in self.population:
            run_data = RunData(
                agent_id=individual.id,
                generation=self.generation,
                genome=individual.genome
            )
            
            run_data.final_score = individual.score
            run_data.final_fitness = individual.fitness
            run_data.total_steps = individual.total_steps
            
            if individual.death_reason is not None:
                if hasattr(individual.death_reason, 'value'):
                    run_data.death_reason = DeathReason(individual.death_reason.value)
                elif isinstance(individual.death_reason, str):
                    run_data.death_reason = DeathReason(individual.death_reason)
                else:
                    run_data.death_reason = DeathReason.UNKNOWN
            else:
                run_data.death_reason = DeathReason.UNKNOWN
            
            run_data.death_position = individual.death_position
            run_data.path_efficiency = individual.food_efficiency
            run_data.exploration_rate = individual.exploration_rate
            run_data.cycle_detected = False
            
            if individual.move_history:
                for i, move in enumerate(individual.move_history):
                    move_record = MoveRecord(
                        step=i,
                        direction=move.get('direction', 'right'),
                        head_position=move.get('head_position', (0, 0)),
                        food_position=move.get('food_position', (0, 0)),
                        score=move.get('score', 0),
                        snake_length=move.get('snake_length', 1)
                    )
                    run_data.add_move(move_record)
            
            stats.add_run(run_data)
        
        stats.finalize()
        self.generation_stats.append(stats)
        print(stats.get_summary())
        
        return stats
    
    def end_episode(self, game, total_reward: float, score: int, 
                    death_reason=None, death_position: tuple = None):
        """End episode with balanced fitness"""
        debug_ga(f"=== end_episode called for individual {self.current_individual_idx} ===")
        debug_ga(f"  Score: {score}, Total reward: {total_reward:.2f}, Death reason: {death_reason}")
        
        if self.current_individual_idx >= len(self.population):
            debug_ga(f"  ⚠️ Index out of range, resetting")
            self.current_individual_idx = 0
            return
        
        individual = self.population[self.current_individual_idx]
        debug_ga(f"  Individual ID: {individual.id}")
        
        individual.death_reason = death_reason
        individual.death_position = death_position
        self._store_move_history(individual, game)
        
        if hasattr(game, 'current_run_data') and game.current_run_data:
            individual.food_efficiency = game.current_run_data.path_efficiency
            individual.exploration_rate = game.current_run_data.exploration_rate
            debug_ga(f"  Food efficiency: {individual.food_efficiency:.2f}, Exploration: {individual.exploration_rate:.2f}")
        
        if isinstance(score, (list, np.ndarray)):
            score = int(score[0]) if len(score) > 0 else 0
        elif score is None:
            score = 0
        else:
            score = int(score)
        
        debug_ga(f"  Final score: {score}")
        
        if score > individual.best_score:
            individual.best_score = score
            debug_ga(f"  🎉 New personal best: {score}")
        
        total_steps = getattr(game, 'steps', 1)
        debug_ga(f"  Total steps in episode: {total_steps}")
        
        behavior = self._extract_behavior(game, individual, score, death_reason)
        individual.behavior_vector = behavior
        individual.survival_time = total_steps
        individual.total_steps = total_steps
        
        debug_ga(f"  Total steps stored: {individual.total_steps}")
        
        if self.debug_mode:
            self.behavior_profiler.log_episode(individual, game, individual.move_history)
        
        # Calculate balanced fitness
        fitness, components = self._calculate_fitness(
            individual, game, score, death_reason, behavior
        )
        
        individual.fitness = fitness
        individual.fitness_components = components
        individual.score = score
        individual.age += 1
        
        debug_ga(f"  Calculated fitness: {fitness:.2f}")
        
        novelty = self._calculate_novelty(behavior)
        individual.novelty = novelty
        debug_ga(f"  Novelty score: {novelty:.4f}")
        
        if novelty > 0.5 and len(self.novelty_archive) < self.archive_size:
            self.novelty_archive.append(behavior)
            debug_ga(f"  Added to novelty archive (size={len(self.novelty_archive)})")
        
        self._check_for_records(individual, behavior)
        
        self.total_evaluations += 1
        debug_ga(f"  Total evaluations: {self.total_evaluations}")
        
        self.current_individual_idx += 1
        debug_ga(f"  Moving to next individual: {self.current_individual_idx}/{len(self.population)}")
        
        if self.current_individual_idx >= len(self.population):
            debug_ga(f"  Generation complete, evolving...")
            self.end_generation()
            self.evolve()
            self.current_individual_idx = 0
    
    def _extract_behavior(self, game, individual: Individual, score: int, death_reason=None) -> np.ndarray:
        """Extract behavioral characteristics"""
        debug_ga(f"  Extracting behavior vector...")
        behavior = []
        
        behavior.append(score)
        
        if hasattr(game, 'position_history'):
            positions = game.position_history[-50:] if game.position_history else []
            if len(positions) > 10:
                direction_changes = 0
                for i in range(1, len(positions)):
                    if positions[i] != positions[i-1]:
                        direction_changes += 1
                behavior.append(direction_changes / len(positions))
                
                unique_positions = len(set(positions))
                behavior.append(unique_positions / len(positions))
            else:
                behavior.extend([0.5, 0.5])
        else:
            behavior.extend([0.5, 0.5])
        
        if score > 0 and individual.total_steps > 0:
            steps_per_food = individual.total_steps / score
            behavior.append(min(steps_per_food / 100, 1.0))
        else:
            behavior.append(1.0)
        
        behavior.append(min(individual.total_steps / 1000, 1.0))
        
        death_reason_str = None
        if death_reason is not None:
            if hasattr(death_reason, 'value'):
                death_reason_str = death_reason.value
            elif isinstance(death_reason, str):
                death_reason_str = death_reason
            else:
                death_reason_str = str(death_reason)
        
        death_map = {
            'wall': [1,0,0],
            'self': [0,1,0],
            'timeout': [0,0,1],
            None: [0,0,0]
        }
        behavior.extend(death_map.get(death_reason_str, [0,0,0]))
        
        behavior_array = np.array(behavior, dtype=np.float32)
        return behavior_array
    
    def _calculate_fitness(self, individual: Individual, game, 
                                        score: int, death_reason,
                                        behavior: np.ndarray) -> Tuple[float, Dict]:
        """Calculate fitness with starvation penalty and exploration bonus"""
        debug_ga(f"  Calculating fitness for individual with score={score}, steps={individual.total_steps}")
        components = {}
        
        # Get grid dimensions
        grid_width = game.config['GAME_WIDTH'] // game.config['SQUARE_SIZE']
        grid_height = game.config['GAME_HEIGHT'] // game.config['SQUARE_SIZE']
        total_cells = grid_width * grid_height
        
        # ========== 1. SURVIVAL ==========
        max_possible_steps = 500
        survival_ratio = min(individual.total_steps / max_possible_steps, 1.0)
        survival_fitness = 400 * survival_ratio
        
        components['survival'] = survival_fitness
        
        # ========== 2. SCORE (exponential reward) ==========
        # Exponential encourages more food seeking
        score_fitness = (score ** 1.5) * 100  # 1=100, 2=283, 3=520, 4=800, 5=1118
        
        components['score'] = score_fitness
        
        # ========== 3. WALL AVOIDANCE ==========
        wall_bonus = 0
        if hasattr(game, 'position_history') and game.position_history and hasattr(game, 'config'):
            recent_positions = game.position_history[-30:]
            
            for pos in recent_positions:
                x, y = pos
                x_cell = x // game.config['SQUARE_SIZE']
                y_cell = y // game.config['SQUARE_SIZE']
                
                dist_to_wall = min(x_cell, grid_width - 1 - x_cell, 
                                y_cell, grid_height - 1 - y_cell)
                
                if dist_to_wall >= 4:
                    wall_bonus += 30
                elif dist_to_wall >= 2:
                    wall_bonus += 10
            
            if recent_positions:
                wall_bonus = wall_bonus / len(recent_positions)
        
        components['wall_bonus'] = wall_bonus
        
        # ========== 4. EFFICIENCY BONUS ==========
        if score > 0:
            steps_per_food = individual.total_steps / score
            if steps_per_food < 20:
                efficiency_bonus = 300
            elif steps_per_food < 40:
                efficiency_bonus = 150
            elif steps_per_food < 80:
                efficiency_bonus = 75
            else:
                efficiency_bonus = 25
        else:
            efficiency_bonus = 0
        components['efficiency'] = efficiency_bonus
        
        # ========== 5. EXPLORATION BONUS ==========
        exploration_bonus = 0
        if hasattr(game, 'position_history'):
            positions = game.position_history
            visited_cells = set()
            for pos in positions:
                x_cell = pos[0] // game.config['SQUARE_SIZE']
                y_cell = pos[1] // game.config['SQUARE_SIZE']
                visited_cells.add((x_cell, y_cell))
            
            coverage_ratio = len(visited_cells) / total_cells
            exploration_bonus = coverage_ratio * 200
        components['exploration'] = exploration_bonus
        
        # ========== 6. STARVATION PENALTY ==========
        steps_without_food = getattr(game, 'steps_without_food', 0)
        if isinstance(steps_without_food, list):
            steps_without_food = steps_without_food[-1] if steps_without_food else 0
        
        starvation_penalty = 0
        if steps_without_food > 50:
            starvation_penalty = -100 * min(steps_without_food / 100, 3)
        elif steps_without_food > 20:
            starvation_penalty = -20
        components['starvation'] = starvation_penalty
        
        # ========== 7. LOOP PENALTY ==========
        loop_penalty = 0
        if hasattr(game, 'position_history'):
            recent = game.position_history[-50:]
            if len(recent) > 30:
                unique_positions = len(set(recent))
                if unique_positions < 10:
                    loop_penalty = -50
        components['loop_penalty'] = loop_penalty
        
        # ========== 8. LENGTH BONUS (small) ==========
        length_bonus = 0
        if hasattr(game, 'snake') and game.snake is not None:
            if hasattr(game.snake, 'get_body_positions'):
                snake_len = len(game.snake.get_body_positions())
            else:
                snake_len = 1
            length_bonus = snake_len * 2
        components['length'] = length_bonus
        
        # ========== 9. DEATH PENALTY ==========
        death_penalty = 0
        death_reason_str = None
        if death_reason is not None:
            if hasattr(death_reason, 'value'):
                death_reason_str = death_reason.value
            elif isinstance(death_reason, str):
                death_reason_str = death_reason
        
        if death_reason_str == 'wall' and individual.total_steps < 10:
            death_penalty = -100
        elif death_reason_str == 'self' and individual.total_steps < 10:
            death_penalty = -75
        elif death_reason_str == 'wall' and individual.total_steps < 30:
            death_penalty = -30
        components['death'] = death_penalty
        
        # Calculate total fitness with new weights
        weights = {
            'survival': 0.6,    
            'score': 1.2,   
            'wall_bonus': 0.5,
            'efficiency': 0.3,
            'exploration': 0.8,   
            'starvation': 1.0,     
            'loop_penalty': 1.0,  
            'length': 0.1,
            'death': 1.0
        }
        
        total_fitness = sum(components.get(k, 0) * weights.get(k, 0) for k in components)
        
        # Base survival bonus
        base_bonus = min(individual.total_steps, 50)
        total_fitness += base_bonus
        
        total_fitness = max(total_fitness, 1.0)
        
        debug_ga(f"  Total fitness: {total_fitness:.2f}")
        
        return total_fitness, components
    
    def _calculate_novelty(self, behavior: np.ndarray) -> float:
        """Calculate novelty of behavior compared to archive"""
        if len(self.novelty_archive) == 0:
            return 1.0
        
        k = min(5, len(self.novelty_archive))
        distances = []
        
        for archived in self.novelty_archive[-50:]:
            min_len = min(len(behavior), len(archived))
            dist = np.linalg.norm(behavior[:min_len] - archived[:min_len])
            distances.append(dist)
        
        distances.sort()
        avg_distance = np.mean(distances[:k]) if distances else 1.0
        
        return avg_distance
    
    def _check_for_records(self, individual: Individual, behavior: np.ndarray):
        """Check if individual broke records"""
        if individual.score > self.best_ever_score:
            self.best_ever_score = individual.score
            self.best_ever_fitness = individual.fitness
            self.best_ever_genome = individual.genome.copy()
            self.best_ever_behavior = behavior.copy()
            self.generations_since_improvement = 0
            self.stagnation_counter = 0
            
            def convert_to_python(obj):
                if obj is None:
                    return None
                if isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {key: convert_to_python(value) for key, value in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [convert_to_python(item) for item in obj]
                if hasattr(obj, 'item'):
                    return obj.item()
                return obj
            
            self.champion_lineage.append({
                'generation': self.generation,
                'score': int(individual.score),
                'fitness': float(individual.fitness),
                'id': str(individual.id),
                'behavior': behavior.tolist() if behavior is not None else None
            })
            
            converted_components = {}
            if individual.fitness_components:
                for key, value in individual.fitness_components.items():
                    converted_components[key] = convert_to_python(value)
            
            self.hall_of_fame.append({
                'generation': self.generation,
                'score': int(individual.score),
                'fitness': float(individual.fitness),
                'genome': individual.genome.tolist() if individual.genome is not None else None,
                'id': str(individual.id),
                'behavior': behavior.tolist() if behavior is not None else None,
                'components': converted_components
            })
            
            print(f"\n🎉🌟 NEW ALL-TIME RECORD! Score: {individual.score} at generation {self.generation} 🌟🎉")
            print(f"   Fitness: {individual.fitness:,.0f}")
            debug_ga(f"  🎉 New all-time record! Score={individual.score}")
        
        elif individual.score > 2 and individual.score >= self.best_ever_score * 0.5:
            existing = any(
                champ.get('score') == individual.score and 
                abs(champ.get('fitness', 0) - individual.fitness) < 100 
                for champ in self.hall_of_fame
            )
            
            if not existing:
                def convert_to_python(obj):
                    if obj is None:
                        return None
                    if isinstance(obj, (np.floating, np.float32, np.float64)):
                        return float(obj)
                    if isinstance(obj, (np.integer, np.int32, np.int64)):
                        return int(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, dict):
                        return {key: convert_to_python(value) for key, value in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [convert_to_python(item) for item in obj]
                    if hasattr(obj, 'item'):
                        return obj.item()
                    return obj
                
                converted_components = {}
                if individual.fitness_components:
                    for key, value in individual.fitness_components.items():
                        converted_components[key] = convert_to_python(value)
                
                self.hall_of_fame.append({
                    'generation': self.generation,
                    'score': int(individual.score),
                    'fitness': float(individual.fitness),
                    'genome': individual.genome.tolist() if individual.genome is not None else None,
                    'id': str(individual.id),
                    'behavior': behavior.tolist() if behavior is not None else None,
                    'components': converted_components
                })
                print(f"   📀 Added to Hall of Fame: Score {individual.score}")
    
    def analyze_genetic_diversity(self):
        """Detailed genetic diversity analysis"""
        if len(self.population) < 2:
            return
        
        genomes = np.array([ind.genome for ind in self.population[:50]])
        
        n = min(50, len(genomes))
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(genomes[i] - genomes[j])
                distances.append(dist)
        
        layer_variances = []
        sample_ind = self.population[0]
        if hasattr(sample_ind.network, 'weights'):
            for layer_idx in range(len(sample_ind.network.weights)):
                layer_weights = []
                for ind in self.population[:50]:
                    if hasattr(ind.network, 'weights') and layer_idx < len(ind.network.weights):
                        layer_weights.append(ind.network.weights[layer_idx].flatten())
                
                if layer_weights:
                    layer_weights = np.array(layer_weights)
                    layer_variances.append(np.var(layer_weights, axis=0).mean())
        
        print(f"\n🧬 GENETIC DIVERSITY ANALYSIS:")
        print(f"   Avg pairwise distance: {np.mean(distances):.4f}")
        print(f"   Distance std dev: {np.std(distances):.4f}")
        print(f"   Layer variances: {[f'{v:.4f}' for v in layer_variances]}")
        
        unique_genomes = set()
        duplicates = 0
        for ind in self.population[:100]:
            genome_hash = hash(ind.genome.tobytes())
            if genome_hash in unique_genomes:
                duplicates += 1
            else:
                unique_genomes.add(genome_hash)
        
        if duplicates > 0:
            print(f"   ⚠️ {duplicates} duplicate individuals detected!")
        
        return np.mean(distances)
    
    def analyze_fitness_landscape(self):
        """Analyze how fitness is distributed"""
        scores = [ind.score for ind in self.population]
        fitnesses = [ind.fitness for ind in self.population]
        
        unique_scores = set(scores)
        score_counts = {s: scores.count(s) for s in unique_scores}
        
        print(f"\n📈 FITNESS LANDSCAPE:")
        print(f"   Score distribution: {score_counts}")
        
        for score_val in sorted(unique_scores):
            score_fitnesses = [f for s, f in zip(scores, fitnesses) if s == score_val]
            if score_fitnesses:
                print(f"   Score {score_val}: fitness range [{min(score_fitnesses):.0f}, {max(score_fitnesses):.0f}], "
                      f"mean {np.mean(score_fitnesses):.0f}")
        
        # Check for long-survival zero-score individuals
        zero_score_survival = [(ind.total_steps, ind.fitness, ind.id) for ind in self.population 
                                if ind.score == 0 and ind.total_steps > 50]
        if zero_score_survival:
            print(f"   📊 {len(zero_score_survival)} zero-score individuals survived >50 steps!")
            for steps, f, i in zero_score_survival[:3]:
                print(f"      {i}: steps={steps}, fitness={f:.0f}")
    
    def diagnose_stagnation(self):
        """Deep dive into why stagnation is occurring"""
        print("\n" + "="*80)
        print("🔬 STAGNATION DIAGNOSIS")
        print("="*80)
        
        if self.stagnation_counter > 50:
            best = self.population[0]
            best_score = best.score
            best_fitness = best.fitness
            
            print(f"\n🎯 LOCAL OPTIMUM ANALYSIS:")
            print(f"   Best score: {best_score} (stuck for {self.stagnation_counter} gens)")
            print(f"   Best fitness: {best_fitness:.0f}")
            print(f"   Best individual: {best.id}")
            print(f"   Components: {best.fitness_components}")
            
            best_lineage = best.id.split('_clone_')[-1] if '_clone_' in best.id else best.id
            older_best = [ind for ind in self.population if best_lineage in ind.id]
            if len(older_best) > 5:
                print(f"   ⚠️ Same lineage dominates for {len(older_best)} generations!")
        
        mutation_rate = self.mutation_rate * self.mutation_rate_adaptation
        avg_score = np.mean([ind.score for ind in self.population])
        scoring_individuals = sum(1 for ind in self.population if ind.score > 0)
        
        print(f"\n⚖️ EXPLORATION VS EXPLOITATION:")
        print(f"   Mutation rate: {mutation_rate:.3f}")
        print(f"   Avg score: {avg_score:.2f}")
        print(f"   Scoring individuals: {scoring_individuals}/100")
        
        wall_deaths = sum(1 for ind in self.population if ind.death_reason and 'wall' in str(ind.death_reason).lower())
        self_deaths = sum(1 for ind in self.population if ind.death_reason and 'self' in str(ind.death_reason).lower())
        
        print(f"\n🚫 DEATH ANALYSIS:")
        print(f"   Wall deaths: {wall_deaths}")
        print(f"   Self deaths: {self_deaths}")
        print(f"   Survival steps (avg): {np.mean([ind.total_steps for ind in self.population]):.1f}")
        
        print(f"\n🧬 SPECIES ANALYSIS:")
        print(f"   Species count: {len(self.species)}")
        for i, species in enumerate(self.species[:3]):
            species_scores = [ind.score for ind in species.members]
            species_fitness = [ind.fitness for ind in species.members]
            if species_scores:
                print(f"   Species {i}: {len(species.members)} members, "
                      f"avg score {np.mean(species_scores):.2f}, "
                      f"avg fitness {np.mean(species_fitness):.0f}")
        
        print("="*80)
    
    def performance_dashboard(self):
        """Real-time dashboard of key metrics"""
        window = 10
        recent_best = self.best_score_history[-window:] if len(self.best_score_history) >= window else self.best_score_history
        recent_avg = self.avg_fitness_history[-window:] if len(self.avg_fitness_history) >= window else self.avg_fitness_history
        
        improvement_rate = (recent_best[-1] - recent_best[0]) / max(len(recent_best), 1) if recent_best else 0
        
        print("\n" + "="*80)
        print("📊 PERFORMANCE DASHBOARD")
        print("="*80)
        print(f"\n⏱️  TIMELINE:")
        print(f"   Generation: {self.generation}")
        print(f"   Runtime: {time.time() - self.start_time:.0f}s")
        
        print(f"\n📈 PROGRESS:")
        print(f"   Best score: {self.best_ever_score}")
        print(f"   Improvement rate: {improvement_rate:.2f} score/generation")
        print(f"   Stagnation: {self.stagnation_counter} generations")
        
        avg_survival = np.mean([ind.total_steps for ind in self.population])
        wall_deaths = sum(1 for ind in self.population if ind.death_reason and 'wall' in str(ind.death_reason).lower())
        
        print(f"\n🎯 GOAL ANALYSIS:")
        if wall_deaths > 50:
            print(f"   ⚠️ {wall_deaths}/100 individuals died from wall collisions")
        if avg_survival < 20:
            print(f"   ⚠️ Average survival only {avg_survival:.1f} steps")
        
        print("="*80)
    
    def _add_stagnation_breakout(self, population: List[Individual]):
        """Add special individuals to break stagnation"""
        if self.stagnation_counter > 100:
            for i in range(5):
                network = self._create_network_architecture()
                for j in range(len(network.weights)):
                    network.weights[j] = np.random.randn(*network.weights[j].shape) * 0.1
                for j in range(len(network.biases)):
                    network.biases[j] = np.random.randn(*network.biases[j].shape) * 0.01
                
                genome = network.get_genome()
                if len(genome) != self.expected_genome_size:
                    genome = self._fix_genome_size(genome, self.expected_genome_size)
                
                individual = Individual(
                    genome=genome,
                    generation=self.generation + 1,
                    network=network,
                    id=f"breakout_{i}_{random.randint(1000, 9999)}"
                )
                population.append(individual)
            
            print(f"   🚀 Added {5} breakout individuals to fight stagnation")
            self.stagnation_counter = 0
    
    def evolve(self):
        debug_ga(f"=== EVOLVING GENERATION {self.generation} ===")
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        best_fitness = self.population[0].fitness
        avg_fitness = np.mean([ind.fitness for ind in self.population])
        best_score = self.population[0].score
        diversity = self.calculate_diversity()
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.best_score_history.append(best_score)
        self.diversity_history.append(diversity)
        
        debug_ga(f"  Best fitness: {best_fitness:.2f}, Avg fitness: {avg_fitness:.2f}")
        debug_ga(f"  Best score: {best_score}, Diversity: {diversity:.4f}")
        
        if best_score > self.last_best_score:
            self.stagnation_counter = 0
            self.last_best_score = best_score
            self.generations_since_improvement = 0
            self.mutation_rate_adaptation = 1.0
            debug_ga(f"  Improvement detected! New best score: {best_score}")
        else:
            self.stagnation_counter += 1
            self.generations_since_improvement += 1
            self.mutation_rate_adaptation = min(2.0, 1.0 + self.stagnation_counter * 0.05)
            debug_ga(f"  Stagnation: {self.stagnation_counter} gens, adaptation: {self.mutation_rate_adaptation:.2f}")
        
        if self.stagnation_counter > 50:
            adaptive_mutation_rate = min(0.3, self.mutation_rate * 1.2)
        else:
            adaptive_mutation_rate = self.mutation_rate * self.mutation_rate_adaptation
            adaptive_mutation_rate = np.clip(adaptive_mutation_rate, 
                                            self.min_mutation_rate, 
                                            self.max_mutation_rate)
        debug_ga(f"  Adaptive mutation rate: {adaptive_mutation_rate:.3f}")
        
        if self.use_speciation:
            self.speciate()
            self.species_count_history.append(len(self.species))
            self.adjust_compatibility_threshold()
            debug_ga(f"  Species count: {len(self.species)}")
        
        new_population = self._create_new_population(adaptive_mutation_rate)
        debug_ga(f"  Created {len(new_population)} offspring")
        
        self._add_elites(new_population)
        debug_ga(f"  After adding elites: {len(new_population)}")
        
        self._add_immigrants(new_population)
        debug_ga(f"  After adding immigrants: {len(new_population)}")
        
        if self.stagnation_counter > 100:
            self._add_stagnation_breakout(new_population)
            debug_ga(f"  After adding breakout: {len(new_population)}")
        
        while len(new_population) < self.population_size:
            network = self._create_network_architecture()
            genome = network.get_genome()
            if len(genome) != self.expected_genome_size:
                genome = self._fix_genome_size(genome, self.expected_genome_size)
                network.set_genome(genome)
            
            individual = Individual(
                genome=genome,
                generation=self.generation + 1,
                species_id=random.randint(0, 9),
                network=network,
                id=f"random_{random.randint(1000, 9999)}"
            )
            new_population.append(individual)
            debug_ga(f"  Added random individual (size now {len(new_population)})")
        
        if len(new_population) > self.population_size:
            new_population.sort(key=lambda x: x.fitness, reverse=True)
            new_population = new_population[:self.population_size]
            debug_ga(f"  Trimmed to {len(new_population)}")
        
        for ind in new_population:
            if len(ind.genome) != self.expected_genome_size:
                ind.genome = self._fix_genome_size(ind.genome, self.expected_genome_size)
                if ind.network:
                    ind.network.set_genome(ind.genome)
        
        self.population = new_population
        self.generation += 1
        
        self._update_elite_archive()
        
        if self.generation % 20 == 0 and self.debug_mode:
            print(f"\n🔬 GENERATION {self.generation} DEEP ANALYSIS")
            self.analyze_genetic_diversity()
            self.analyze_fitness_landscape()
            self.behavior_profiler.print_summary()
            self.performance_dashboard()
        
        if self.stagnation_counter > 50 and self.stagnation_counter % 25 == 0:
            print(f"\n⚠️ STAGNATION ALERT! (Gen {self.generation}, {self.stagnation_counter} gens stuck)")
            self.diagnose_stagnation()
        
        self.print_stats()
        
        if self.generation % 10 == 0:
            self.save_checkpoint()
            if self.debug_mode:
                self.behavior_profiler.save_analysis(f"analysis/behavior_gen_{self.generation}.json")
    
    def _create_new_population(self, mutation_rate: float) -> List[Individual]:
        """Create new population through selection and variation"""
        new_population = []
        
        if self.use_speciation and self.species:
            new_population = self._reproduce_through_species(mutation_rate)
        else:
            new_population = self._reproduce_standard(mutation_rate)
        
        return new_population
    
    def _reproduce_through_species(self, mutation_rate: float) -> List[Individual]:
        """Reproduce using species information"""
        new_population = []
        
        total_adjusted_fitness = 0
        species_fitness = []
        
        for species in self.species:
            if not species.members:
                continue
            
            # Use survival and score for species fitness
            species_score = species.best_score
            species_avg_survival = np.mean([m.total_steps for m in species.members]) if species.members else 0
            species_fitness_val = (species_score + 1) * (1 + species_avg_survival / 100)
            species_fitness.append((species, species_fitness_val))
            total_adjusted_fitness += species_fitness_val
        
        for species, score in species_fitness:
            if total_adjusted_fitness == 0:
                offspring_count = max(1, self.population_size // max(len(self.species), 1))
            else:
                offspring_count = max(1, int((score / total_adjusted_fitness) * 
                                           self.population_size * 0.8))
            
            species.members.sort(key=lambda x: x.fitness, reverse=True)
            
            for _ in range(offspring_count):
                if len(new_population) >= self.population_size - self.elite_size - 5:
                    break
                
                if len(species.members) >= 2:
                    parent1 = self._select_parent(species.members)
                    parent2 = self._select_parent(species.members)
                    child = self._create_offspring(parent1, parent2, mutation_rate)
                    child.species_id = species.id
                else:
                    parent = species.members[0]
                    child = self._create_offspring(parent, None, mutation_rate)
                    child.species_id = species.id
                
                if len(child.genome) != self.expected_genome_size:
                    child.genome = self._fix_genome_size(child.genome, self.expected_genome_size)
                    if child.network:
                        child.network.set_genome(child.genome)
                
                new_population.append(child)
        
        return new_population
    
    def _reproduce_standard(self, mutation_rate: float) -> List[Individual]:
        """Standard reproduction without speciation"""
        new_population = []
        
        while len(new_population) < self.population_size - self.elite_size - 5:
            parent1 = self._select_parent(self.population)
            parent2 = self._select_parent(self.population)
            child = self._create_offspring(parent1, parent2, mutation_rate)
            
            if len(child.genome) != self.expected_genome_size:
                child.genome = self._fix_genome_size(child.genome, self.expected_genome_size)
                if child.network:
                    child.network.set_genome(child.genome)
            
            new_population.append(child)
        
        return new_population
    
    def _select_parent(self, population: List[Individual]) -> Individual:
        """Select parent using current selection method"""
        if self.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(population)
        elif self.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection(population)
        elif self.selection_method == SelectionMethod.RANK:
            return self._rank_selection(population)
        elif self.selection_method == SelectionMethod.BOLTZMANN:
            return self._boltzmann_selection(population)
        else:
            return self._tournament_selection(population)
    
    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """Tournament selection"""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _roulette_selection(self, population: List[Individual]) -> Individual:
        """Roulette wheel selection"""
        fitnesses = np.array([ind.fitness for ind in population])
        if np.sum(fitnesses) == 0:
            return random.choice(population)
        
        probs = fitnesses / np.sum(fitnesses)
        return np.random.choice(population, p=probs)
    
    def _rank_selection(self, population: List[Individual]) -> Individual:
        """Rank-based selection"""
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        ranks = np.arange(1, len(sorted_pop) + 1)
        probs = ranks / np.sum(ranks)
        return np.random.choice(sorted_pop, p=probs)
    
    def _boltzmann_selection(self, population: List[Individual]) -> Individual:
        """Boltzmann selection (temperature-based)"""
        temperature = max(0.1, 1.0 - self.generation / 100)
        fitnesses = np.array([ind.fitness for ind in population])
        exp_fitness = np.exp(fitnesses / temperature)
        probs = exp_fitness / np.sum(exp_fitness)
        return np.random.choice(population, p=probs)
    
    def _create_offspring(self, parent1: Individual, parent2: Optional[Individual], 
                          mutation_rate: float) -> Individual:
        """Create offspring through crossover and mutation"""
        
        if parent2 is not None and random.random() < self.crossover_rate:
            if self.stagnation_counter > 15:
                method = random.choice(['simulated_binary', 'blend_alpha', 'uniform'])
            else:
                method = random.choice(['simulated_binary', 'weighted_average'])
            
            child_genome = EvolutionaryOptimizer.crossover(
                parent1.genome, parent2.genome, method
            )
            parent_ids = [parent1.id, parent2.id]
            debug_ga(f"    Crossover: {method} between {parent1.id} and {parent2.id}")
        else:
            child_genome = parent1.genome.copy()
            parent_ids = [parent1.id]
            debug_ga(f"    No crossover, cloning {parent1.id}")
        
        if self.stagnation_counter > 20:
            mutation_method = random.choice(['cauchy', 'adaptive_gaussian', 'swap', 'inversion'])
        else:
            mutation_method = random.choice(['adaptive_gaussian', 'polynomial'])
        
        child_genome = EvolutionaryOptimizer.mutate(
            child_genome, 
            rate=mutation_rate,
            strength=0.1 * self.mutation_rate_adaptation,
            method=mutation_method,
            generation=self.generation
        )
        debug_ga(f"    Mutation: {mutation_method}, rate={mutation_rate:.3f}")
        
        network = self._create_network_architecture()
        network.set_genome(child_genome)
        
        child = Individual(
            genome=child_genome,
            generation=self.generation + 1,
            network=network,
            parent_ids=parent_ids,
            id=f"offspring_{random.randint(10000, 99999)}"
        )
        
        return child
    
    def _add_elites(self, population: List[Individual]):
        """Add elite individuals to new population"""
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        if self.stagnation_counter > 50:
            elite_count = min(2, self.elite_size)
        else:
            elite_count = min(self.elite_size, len(self.population))
        
        for i in range(elite_count):
            elite = self.population[i].clone()
            
            if self.stagnation_counter > 30:
                elite.genome = EvolutionaryOptimizer.mutate(
                    elite.genome, 
                    rate=0.05,
                    strength=0.05,
                    method='adaptive_gaussian',
                    generation=self.generation
                )
                if elite.network:
                    elite.network.set_genome(elite.genome)
            
            if len(elite.genome) != self.expected_genome_size:
                elite.genome = self._fix_genome_size(elite.genome, self.expected_genome_size)
                if elite.network:
                    elite.network.set_genome(elite.genome)
            
            elite.age += 1
            elite.generation = self.generation + 1
            population.insert(0, elite)
            debug_ga(f"    Added elite {i}: score={elite.score}, fitness={elite.fitness:.2f}")
        
        if (self.stagnation_counter > 30 and 
            self.elite_archive and 
            self.generations_since_improvement % 20 == 0):
            
            archive_best = max(self.elite_archive, key=lambda x: x.score)
            if archive_best.score > self.population[0].score:
                champion = archive_best.clone()
                champion.genome = EvolutionaryOptimizer.mutate(
                    champion.genome,
                    rate=0.1,
                    strength=0.1,
                    method='adaptive_gaussian',
                    generation=self.generation
                )
                if champion.network:
                    champion.network.set_genome(champion.genome)
                
                if len(champion.genome) != self.expected_genome_size:
                    champion.genome = self._fix_genome_size(champion.genome, self.expected_genome_size)
                    if champion.network:
                        champion.network.set_genome(champion.genome)
                
                champion.generation = self.generation + 1
                population.insert(1, champion)
                print(f"   🏆 Injected mutated archived elite with score {archive_best.score}")
                debug_ga(f"    Injected mutated archived elite: score={archive_best.score}")
    
    def _add_immigrants(self, population: List[Individual]):
        """Add random immigrants for diversity"""
        if self.diversity_history and len(self.diversity_history) > 0 and self.diversity_history[-1] < 2.0:
            immigrant_count = 5
        else:
            immigrant_count = 3
        
        debug_ga(f"    Adding {immigrant_count} immigrants")
        for i in range(immigrant_count):
            if len(population) >= self.population_size:
                break
            
            network = self._create_network_architecture()
            genome = network.get_genome()
            if len(genome) != self.expected_genome_size:
                genome = self._fix_genome_size(genome, self.expected_genome_size)
                network.set_genome(genome)
            
            immigrant = Individual(
                genome=genome,
                generation=self.generation + 1,
                species_id=random.randint(0, 9),
                network=network,
                id=f"immigrant_{self.generation}_{i}"
            )
            population.append(immigrant)
    
    def _update_elite_archive(self):
        """Update elite archive with best individuals"""
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        top_individuals = self.population[:10]
        
        for ind in top_individuals:
            existing = any(archive_ind.score >= ind.score and 
                          abs(archive_ind.fitness - ind.fitness) < 100 
                          for archive_ind in self.elite_archive)
            
            if not existing and ind.score > 2:
                self.elite_archive.append(ind.clone())
                debug_ga(f"  Added to elite archive: score={ind.score}, fitness={ind.fitness:.2f}")
        
        if len(self.elite_archive) > 50:
            self.elite_archive.sort(key=lambda x: x.score, reverse=True)
            self.elite_archive = self.elite_archive[:50]
            debug_ga(f"  Trimmed elite archive to 50")
    
    def speciate(self):
        """Assign individuals to species"""
        if not self.use_speciation:
            return
        
        debug_ga(f"  Speciating {len(self.population)} individuals...")
        
        for species in self.species:
            species.members = []
        
        for individual in self.population:
            best_species = None
            best_distance = float('inf')
            
            for species in self.species:
                distance = np.linalg.norm(individual.genome - species.representative)
                if distance < best_distance:
                    best_distance = distance
                    best_species = species
            
            if best_species and best_distance < self.compatibility_threshold:
                best_species.add_member(individual)
            else:
                new_species = Species(len(self.species), individual.genome.copy())
                new_species.add_member(individual)
                new_species.generation_born = self.generation
                self.species.append(new_species)
                debug_ga(f"    Created new species {len(self.species)-1}")
        
        self.species = [s for s in self.species if s.members]
        
        for species in self.species:
            if species.members:
                best_member = max(species.members, key=lambda x: x.fitness)
                species.representative = best_member.genome.copy()
                species.age += 1
                
                if best_member.fitness > species.best_fitness:
                    species.best_fitness = best_member.fitness
                    species.stagnation_counter = 0
                else:
                    species.stagnation_counter += 1
        
        if len(self.species) > 1:
            self.species = [s for s in self.species if s.stagnation_counter <= 20]
            debug_ga(f"  Removed stagnant species, now {len(self.species)} species")
    
    def adjust_compatibility_threshold(self):
        """Dynamically adjust threshold to maintain target species count"""
        target_min = max(3, self.target_species // 2)
        target_max = self.target_species * 2
        
        if len(self.species) > target_max:
            self.compatibility_threshold *= 1.1
            print(f"   📊 Increasing threshold to {self.compatibility_threshold:.2f} (species: {len(self.species)})")
            debug_ga(f"  Increased threshold to {self.compatibility_threshold:.2f}")
        elif len(self.species) < target_min and self.compatibility_threshold > 2.0:
            self.compatibility_threshold *= 0.9
            print(f"   📊 Decreasing threshold to {self.compatibility_threshold:.2f} (species: {len(self.species)})")
            debug_ga(f"  Decreased threshold to {self.compatibility_threshold:.2f}")
        
    def calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.population) < 2:
            return 1.0
        
        samples = min(100, len(self.population) * 2)
        distances = []
        
        for _ in range(samples):
            i, j = random.sample(range(len(self.population)), 2)
            dist = np.linalg.norm(self.population[i].genome - self.population[j].genome)
            distances.append(dist)
        
        diversity = np.mean(distances) / 10.0
        debug_ga(f"  Diversity: {diversity:.4f}")
        return diversity
    
    def print_stats(self):
        """Print detailed statistics"""
        print("\n" + "="*80)
        print(f"🧬 GENERATION {self.generation} 🧬")
        print("="*80)
        
        best = self.population[0]
        
        print(f"Best Individual: {best.id}")
        print(f"  Score: {best.score} | Fitness: {best.fitness:,.0f}")
        print(f"  Age: {best.age} | Novelty: {best.novelty:.3f}")
        print(f"  Components: {best.fitness_components}")
        
        scores = [ind.score for ind in self.population]
        fitnesses = [ind.fitness for ind in self.population]
        
        print(f"\nPopulation Stats:")
        print(f"  Avg Score: {np.mean(scores):.2f} | Max: {max(scores)}")
        print(f"  Avg Fitness: {np.mean(fitnesses):,.0f} | Max: {max(fitnesses):,.0f}")
        print(f"  Scoring Individuals: {sum(1 for s in scores if s > 0)}/{len(scores)}")
        
        survival_steps = [ind.total_steps for ind in self.population]
        wall_deaths = sum(1 for ind in self.population if ind.death_reason and 'wall' in str(ind.death_reason).lower())
        
        print(f"  Avg Survival: {np.mean(survival_steps):.1f} steps")
        print(f"  Wall Deaths: {wall_deaths}")
        
        diversity = self.calculate_diversity()
        print(f"\nDiversity: {diversity:.3f}")
        print(f"Species: {len(self.species)} (target: {self.target_species})")
        print(f"Stagnation: {self.stagnation_counter} gens")
        print(f"Mutation Rate: {self.mutation_rate * self.mutation_rate_adaptation:.3f}")
        
        if self.hall_of_fame:
            top_champs = sorted(list(self.hall_of_fame), 
                               key=lambda x: x.get('score', 0), reverse=True)[:3]
            champ_str = ", ".join([f"Gen {c.get('generation', 0)}:{c.get('score', 0)}" for c in top_champs])
            print(f"\nHall of Fame: {champ_str}")
            print(f"All-Time Best: {self.best_ever_score}")
        
        print("="*80)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            'generation': self.generation,
            'best_score': self.best_ever_score,
            'best_fitness': self.best_ever_fitness,
            'population_size': len(self.population),
            'diversity': self.calculate_diversity(),
            'species_count': len(self.species),
            'stagnation': self.stagnation_counter,
            'scoring_count': sum(1 for ind in self.population if ind.score > 0),
            'avg_score': np.mean([ind.score for ind in self.population]) if self.population else 0,
            'avg_fitness': np.mean([ind.fitness for ind in self.population]) if self.population else 0,
            'avg_survival': np.mean([ind.total_steps for ind in self.population]) if self.population else 0,
            'wall_deaths': sum(1 for ind in self.population if ind.death_reason and 'wall' in str(ind.death_reason).lower())
        }
    
    def save_checkpoint(self, path: str = "memory/snake_ancestry.json"):
        """Save checkpoint to file"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            def convert_to_serializable(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_to_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_to_serializable(item) for item in obj)
                elif hasattr(obj, 'item'):
                    return obj.item()
                return obj
            
            for ind in self.population[:min(50, len(self.population))]:
                if len(ind.genome) != self.expected_genome_size:
                    ind.genome = self._fix_genome_size(ind.genome, self.expected_genome_size)
                    if ind.network:
                        ind.network.set_genome(ind.genome)
            
            population_genomes = []
            population_fitness = []
            population_scores = []
            
            save_count = min(50, len(self.population))
            for i in range(save_count):
                ind = self.population[i]
                if ind.genome is not None:
                    population_genomes.append(ind.genome.tolist())
                    population_fitness.append(float(ind.fitness))
                    population_scores.append(int(ind.score))
            
            checkpoint = {
                'generation': self.generation,
                'best_ever_score': self.best_ever_score,
                'best_ever_fitness': convert_to_serializable(self.best_ever_fitness),
                'best_ever_genome': self.best_ever_genome.tolist() if self.best_ever_genome is not None else None,
                'population_genomes': population_genomes,
                'population_fitness': population_fitness,
                'population_scores': population_scores,
                'best_fitness_history': [convert_to_serializable(x) for x in self.best_fitness_history],
                'avg_fitness_history': [convert_to_serializable(x) for x in self.avg_fitness_history],
                'best_score_history': [convert_to_serializable(x) for x in self.best_score_history],
                'diversity_history': [convert_to_serializable(x) for x in self.diversity_history],
                'species_count_history': [convert_to_serializable(x) for x in self.species_count_history],
                'hall_of_fame': convert_to_serializable(list(self.hall_of_fame)),
                'champion_lineage': convert_to_serializable(self.champion_lineage),
                'mutation_rate_adaptation': convert_to_serializable(self.mutation_rate_adaptation),
                'compatibility_threshold': convert_to_serializable(self.compatibility_threshold),
                'expected_genome_size': self.expected_genome_size
            }
            
            with open(path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            try:
                memory = AncestorsMemory(path)
                memory.save_current_state(
                    generation=self.generation,
                    best_ever_score=self.best_ever_score,
                    best_ever_fitness=self.best_ever_fitness,
                    best_ever_genome=self.best_ever_genome,
                    best_fitness_history=self.best_fitness_history,
                    avg_fitness_history=self.avg_fitness_history,
                    best_score_history=self.best_score_history,
                    diversity_history=self.diversity_history
                )
            except Exception as e:
                print(f"Note: Could not update ancestral memory: {e}")
            
            print(f"💾 Saved checkpoint - Generation {self.generation} with {len(population_genomes)} individuals")
            debug_ga(f"Checkpoint saved: {path}")
        except Exception as e:
            print(f"Note: Could not save checkpoint: {e}")
    
    def load_checkpoint(self, path: str = "memory/snake_ancestry.json"):
        """Load checkpoint from file"""
        try:
            if not os.path.exists(path):
                print("⚠️ No checkpoint file found, starting fresh")
                return
            
            with open(path, 'r') as f:
                checkpoint = json.load(f)
            
            self.generation = checkpoint.get('generation', 0)
            self.best_ever_score = checkpoint.get('best_ever_score', 0)
            self.best_ever_fitness = checkpoint.get('best_ever_fitness', 0.0)
            self.expected_genome_size = checkpoint.get('expected_genome_size', None)
            
            if self.expected_genome_size is None:
                template_network = self._create_network_architecture()
                self.expected_genome_size = len(template_network.get_genome())
                print(f"📏 Set expected genome size to {self.expected_genome_size}")
            
            if checkpoint.get('best_ever_genome'):
                self.best_ever_genome = np.array(checkpoint['best_ever_genome'])
                if self.expected_genome_size and len(self.best_ever_genome) != self.expected_genome_size:
                    self.best_ever_genome = self._fix_genome_size(self.best_ever_genome, self.expected_genome_size)
            
            self.best_fitness_history = checkpoint.get('best_fitness_history', [])
            self.avg_fitness_history = checkpoint.get('avg_fitness_history', [])
            self.best_score_history = checkpoint.get('best_score_history', [])
            self.diversity_history = checkpoint.get('diversity_history', [])
            self.species_count_history = checkpoint.get('species_count_history', [])
            self.hall_of_fame = deque(checkpoint.get('hall_of_fame', []), maxlen=50)
            self.champion_lineage = checkpoint.get('champion_lineage', [])
            self.mutation_rate_adaptation = checkpoint.get('mutation_rate_adaptation', 1.0)
            self.compatibility_threshold = checkpoint.get('compatibility_threshold', 10.0)
            
            if 'population_genomes' in checkpoint:
                print(f"🔄 Loading {len(checkpoint['population_genomes'])} individuals from checkpoint...")
                self.population = []
                
                for i, genome_data in enumerate(checkpoint['population_genomes']):
                    try:
                        genome = np.array(genome_data, dtype=np.float32)
                        if self.expected_genome_size and len(genome) != self.expected_genome_size:
                            genome = self._fix_genome_size(genome, self.expected_genome_size)
                        
                        network = self._create_network_architecture()
                        network.set_genome(genome)
                        
                        individual = Individual(
                            genome=genome,
                            generation=self.generation,
                            species_id=i % 10,
                            network=network,
                            id=f"loaded_{i}_{random.randint(1000, 9999)}"
                        )
                        
                        if i < len(checkpoint.get('population_fitness', [])):
                            individual.fitness = checkpoint['population_fitness'][i]
                        
                        if i < len(checkpoint.get('population_scores', [])):
                            individual.score = checkpoint['population_scores'][i]
                            individual.best_score = individual.score
                        
                        self.population.append(individual)
                        
                    except Exception as e:
                        print(f"⚠️ Error loading individual {i}: {e}")
                        continue
                
                print(f"✅ Successfully loaded {len(self.population)} individuals")
            else:
                print("⚠️ No population genomes found in checkpoint, creating new population")
                self.create_initial_population()
            
            print(f"📀 Loaded checkpoint - Generation {self.generation}, Best Score: {self.best_ever_score}")
            print(f"   Population size: {len(self.population)}")
            debug_ga(f"Checkpoint loaded: {path}")
            
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}")
            print("🆕 Creating new population...")
            self.create_initial_population()

import json
import os
import numpy as np
import traceback
import time
from datetime import datetime

class AncestorsMemory:
    """
    Manages the ancestral memory of evolved snakes.
    Stores all genetic history in a JSON file.
    """

    def __init__(self, filename="memory/snake_ancestry.json"):
        self.filename = filename
        self.memory_dir = os.path.dirname(filename)
        self._ensure_memory_directory()
        self.data = self._load_or_create()
        
        # Track save attempts to prevent infinite loops
        self._saving = False
    
    def _ensure_memory_directory(self):
        """Create memory directory if it doesn't exist"""
        if not os.path.exists(self.memory_dir):
            try:
                os.makedirs(self.memory_dir)
                print(f"📁 Created memory directory: {self.memory_dir}")
            except Exception as e:
                print(f"⚠️ Could not create memory directory: {e}")
    
    def _load_or_create(self):
        """Load existing ancestry or create new structure"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        print("⚠️ Memory file is empty, creating new one...")
                        return self._create_new_ancestry()
                    
                    data = json.loads(content)
                    
                    # Validate that we have the expected structure
                    if isinstance(data, dict):
                        self._ensure_data_structure(data)
                        return data
                    else:
                        print("⚠️ Memory file has invalid structure, creating new one...")
                        return self._create_new_ancestry()
                        
            except json.JSONDecodeError as e:
                print(f"⚠️ Memory file corrupted (JSON error), creating new one...")
                # Backup corrupted file
                if os.path.exists(self.filename):
                    backup_name = f"{self.filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    try:
                        os.rename(self.filename, backup_name)
                        print(f"   Backed up corrupted file to: {backup_name}")
                    except Exception as backup_error:
                        print(f"   Could not backup: {backup_error}")
                return self._create_new_ancestry()
                
            except Exception as e:
                print(f"⚠️ Error loading memory: {e}, creating new one...")
                return self._create_new_ancestry()
        else:
            return self._create_new_ancestry()
    
    def _ensure_data_structure(self, data):
        """Ensure all required fields exist in data structure"""
        if 'metadata' not in data:
            data['metadata'] = {}
        
        default_metadata = {
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'version': '1.0',
            'total_generations': 0,
            'total_champions': 0
        }
        for key, value in default_metadata.items():
            if key not in data['metadata']:
                data['metadata'][key] = value
        
        if 'current_state' not in data:
            data['current_state'] = {
                'generation': 0,
                'best_ever_score': 0,
                'best_ever_fitness': -1e9,
                'best_ever_genome': None
            }
        
        if 'history' not in data:
            data['history'] = {
                'generations': [],
                'best_fitness': [],
                'avg_fitness': [],
                'best_scores': [],
                'diversity': []
            }
        
        # Add behavioral stats to track zigzag
        if 'behavior_stats' not in data:
            data['behavior_stats'] = {
                'zigzag_tendency': [],
                'self_collision_rate': [],
                'avg_turn_frequency': []
            }
        
        if 'hall_of_fame' not in data:
            data['hall_of_fame'] = []
        
        if 'strategies' not in data:
            data['strategies'] = []
        
        if 'lineage' not in data:
            data['lineage'] = {}
    
    def _create_new_ancestry(self):
        """Create a new ancestry record"""
        return {
            'metadata': {
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'version': '1.0',
                'total_generations': 0,
                'total_champions': 0
            },
            'current_state': {
                'generation': 0,
                'best_ever_score': 0,
                'best_ever_fitness': -1e9,
                'best_ever_genome': None
            },
            'history': {
                'generations': [],
                'best_fitness': [],
                'avg_fitness': [],
                'best_scores': [],
                'diversity': []
            },
            'behavior_stats': {
                'zigzag_tendency': [],
                'self_collision_rate': [],
                'avg_turn_frequency': []
            },
            'hall_of_fame': [],  # Champions across generations
            'strategies': [],     # Discovered strategies
            'lineage': {}         # Family tree
        }
    
    def _serialize_genome(self, genome):
        """Convert numpy array to list for JSON safely"""
        if genome is None:
            return None
        
        try:
            if isinstance(genome, np.ndarray):
                # Handle NaN and Inf values
                if np.any(np.isnan(genome)) or np.any(np.isinf(genome)):
                    # Replace problematic values
                    genome = np.nan_to_num(genome, nan=0.0, posinf=5.0, neginf=-5.0)
                return genome.tolist()
            
            if isinstance(genome, list):
                # Check for numpy types in list
                clean_list = []
                for item in genome:
                    if isinstance(item, (np.floating, np.integer)):
                        clean_list.append(float(item) if isinstance(item, np.floating) else int(item))
                    elif isinstance(item, (float, int)):
                        clean_list.append(item)
                    else:
                        clean_list.append(0.0)
                return clean_list
            
            if isinstance(genome, (np.floating, np.integer)):
                return float(genome) if isinstance(genome, np.floating) else int(genome)
            
        except Exception as e:
            print(f"Note: Error serializing genome: {e}")
            return None
        
        return None
    
    def _deserialize_genome(self, genome_list):
        """Convert list back to numpy array safely"""
        if genome_list is None:
            return None
        
        try:
            if isinstance(genome_list, list):
                # Convert any non-numeric values
                clean_list = []
                for item in genome_list:
                    if isinstance(item, (int, float)):
                        clean_list.append(float(item))
                    elif isinstance(item, (np.floating, np.integer)):
                        clean_list.append(float(item))
                    else:
                        clean_list.append(0.0)
                return np.array(clean_list, dtype=np.float32)
        except Exception as e:
            print(f"Note: Error deserializing genome: {e}")
            return None
        
        return None
    
    def _convert_to_python_types(self, obj):
        """Recursively convert numpy types to Python native types"""
        if obj is None:
            return None
        if isinstance(obj, (np.floating, float)):
            # Handle inf and nan
            if np.isnan(obj) or np.isinf(obj):
                return 0.0
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return self._serialize_genome(obj)
        if isinstance(obj, list):
            return [self._convert_to_python_types(item) for item in obj]
        if isinstance(obj, dict):
            return {str(key): self._convert_to_python_types(value) for key, value in obj.items()}
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    
    def save_current_state(self,
                          generation,
                          best_ever_score,
                          best_ever_fitness,
                          best_ever_genome,
                          best_fitness_history,
                          avg_fitness_history,
                          best_score_history,
                          diversity_history):
        """Save the current state of the algorithm"""
        
        # Convert any numpy values to Python native types
        try:
            def clean_history_list(hist):
                if not isinstance(hist, list):
                    return []
                cleaned = []
                for item in hist:
                    if isinstance(item, (list, np.ndarray)):
                        # If it's a list of lists, take the last value
                        if len(item) > 0:
                            val = item[-1]
                            if isinstance(val, (int, float, np.number)):
                                cleaned.append(float(val))
                            else:
                                cleaned.append(0.0)
                        else:
                            cleaned.append(0.0)
                    elif isinstance(item, (int, float, np.number)):
                        cleaned.append(float(item))
                    else:
                        cleaned.append(0.0)
                return cleaned
            
            best_fitness_history = clean_history_list(best_fitness_history)
            avg_fitness_history = clean_history_list(avg_fitness_history)
            best_score_history = clean_history_list(best_score_history)
            diversity_history = clean_history_list(diversity_history)
            
            fitness_value = best_ever_fitness
            if fitness_value == float('-inf') or fitness_value == -np.inf:
                fitness_value = -1e9
            else:
                try:
                    fitness_value = float(fitness_value)
                except:
                    fitness_value = -1e9
            
            self.data['current_state'] = {
                'generation': int(generation),
                'best_ever_score': int(best_ever_score),
                'best_ever_fitness': fitness_value,
                'best_ever_genome': self._serialize_genome(best_ever_genome)
            }
            
            # Generate generations list
            generations = list(range(len(best_fitness_history))) if best_fitness_history else []
            
            self.data['history'] = {
                'generations': generations,
                'best_fitness': best_fitness_history,
                'avg_fitness': avg_fitness_history,
                'best_scores': best_score_history,
                'diversity': diversity_history
            }
            
            self.data['metadata']['last_updated'] = datetime.now().isoformat()
            self.data['metadata']['total_generations'] = int(generation)
            
            self._save()
            
        except Exception as e:
            print(f"Note: Could not save current state: {e}")
            traceback.print_exc()
    
    def add_champion(self, generation, score, fitness, genome, metadata=None):
        """Add a champion to the hall of fame"""
        try:
            if isinstance(score, (list, np.ndarray)):
                score_val = float(score[0]) if len(score) > 0 else 0
            else:
                score_val = float(score) if score is not None else 0
            
            if isinstance(fitness, (list, np.ndarray)):
                fitness_val = float(fitness[0]) if len(fitness) > 0 else -1e9
            else:
                fitness_val = float(fitness) if fitness is not None and fitness != float('-inf') else -1e9
            
            champion = {
                'id': f"champ_gen_{generation}_{len(self.data['hall_of_fame'])}",
                'generation': int(generation),
                'score': int(score_val),
                'fitness': fitness_val,
                'genome': self._serialize_genome(genome),
                'timestamp': datetime.now().isoformat(),
                'metadata': self._convert_to_python_types(metadata or {})
            }
            
            self.data['hall_of_fame'].append(champion)
            
            # Keep hall of fame sorted by score
            self.data['hall_of_fame'].sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Keep only top 50 champions
            if len(self.data['hall_of_fame']) > 50:
                self.data['hall_of_fame'] = self.data['hall_of_fame'][:50]
            
            self.data['metadata']['total_champions'] = len(self.data['hall_of_fame'])
            self._save()
            
            return champion['id']
        except Exception as e:
            print(f"Note: Could not add champion: {e}")
            return None
    
    def add_behavior_stats(self, generation, zigzag_rate, self_collision_rate, turn_frequency):
        """Track behavioral statistics to analyze zigzag problem"""
        try:
            if 'behavior_stats' not in self.data:
                self.data['behavior_stats'] = {
                    'zigzag_tendency': [],
                    'self_collision_rate': [],
                    'avg_turn_frequency': []
                }
            
            zigzag_val = float(zigzag_rate) if zigzag_rate is not None else 0.0
            collision_val = float(self_collision_rate) if self_collision_rate is not None else 0.0
            turn_val = float(turn_frequency) if turn_frequency is not None else 0.0
            
            self.data['behavior_stats']['zigzag_tendency'].append({
                'generation': int(generation),
                'value': zigzag_val
            })
            
            self.data['behavior_stats']['self_collision_rate'].append({
                'generation': int(generation),
                'value': collision_val
            })
            
            self.data['behavior_stats']['avg_turn_frequency'].append({
                'generation': int(generation),
                'value': turn_val
            })
            
            for key in self.data['behavior_stats']:
                if len(self.data['behavior_stats'][key]) > 100:
                    self.data['behavior_stats'][key] = self.data['behavior_stats'][key][-100:]
            
            self._save()
        except Exception as e:
            print(f"Note: Could not add behavior stats: {e}")
    
    def add_strategy(self, generation, strategy_name, description, score_milestone=None):
        """Record a new strategy discovered by the AI"""
        try:
            milestone_val = None
            if score_milestone is not None:
                if isinstance(score_milestone, (list, np.ndarray)):
                    milestone_val = int(score_milestone[0]) if len(score_milestone) > 0 else None
                else:
                    milestone_val = int(score_milestone)
            
            strategy = {
                'id': f"strat_gen_{generation}_{len(self.data['strategies'])}",
                'generation': int(generation),
                'name': str(strategy_name),
                'description': str(description),
                'score_milestone': milestone_val,
                'timestamp': datetime.now().isoformat()
            }
            
            self.data['strategies'].append(strategy)
            self._save()
            
            return strategy['id']
        except Exception as e:
            print(f"Note: Could not add strategy: {e}")
            return None
    
    def add_lineage(self, child_id, parent1_id, parent2_id, generation):
        """Record parent-child relationships"""
        try:
            gen_key = f"gen_{generation}"
            if gen_key not in self.data['lineage']:
                self.data['lineage'][gen_key] = []
            
            self.data['lineage'][gen_key].append({
                'child': str(child_id),
                'parents': [str(parent1_id), str(parent2_id)],
                'timestamp': datetime.now().isoformat()
            })
            
            self._save()
        except Exception as e:
            print(f"Note: Could not add lineage: {e}")
    
    def load_state(self):
        """Load the current state for the genetic algorithm"""
        try:
            self.data = self._load_or_create()
            
            current_state = self.data.get('current_state', {})
            
            # Deserialize genome
            best_genome = None
            if current_state.get('best_ever_genome'):
                best_genome = self._deserialize_genome(
                    current_state['best_ever_genome']
                )
            
            best_fitness = current_state.get('best_ever_fitness', -1e9)
            
            # Ensure all history lists are flat
            def flatten_history(hist):
                if not isinstance(hist, list):
                    return []
                flattened = []
                for item in hist:
                    if isinstance(item, list):
                        flattened.extend(item)
                    else:
                        flattened.append(item)
                return flattened
            
            history = self.data.get('history', {})
            
            return {
                'generation': current_state.get('generation', 0),
                'best_ever_score': current_state.get('best_ever_score', 0),
                'best_ever_fitness': best_fitness,
                'best_ever_genome': best_genome,
                'best_fitness_history': flatten_history(history.get('best_fitness', [])),
                'avg_fitness_history': flatten_history(history.get('avg_fitness', [])),
                'best_score_history': flatten_history(history.get('best_scores', [])),
                'diversity_history': flatten_history(history.get('diversity', []))
            }
        except Exception as e:
            print(f"Note: Could not load state: {e}")
            return {
                'generation': 0,
                'best_ever_score': 0,
                'best_ever_fitness': float('-inf'),
                'best_ever_genome': None,
                'best_fitness_history': [],
                'avg_fitness_history': [],
                'best_score_history': [],
                'diversity_history': []
            }
    
    def get_champion(self, generation=None):
        """Get champion from specific generation or best overall"""
        try:
            self.data = self._load_or_create()
            
            if generation is not None:
                for champ in self.data['hall_of_fame']:
                    if champ.get('generation') == generation:
                        champ_copy = champ.copy()
                        champ_copy['genome'] = self._deserialize_genome(champ_copy.get('genome'))
                        return champ_copy
                return None
            
            # Return best overall
            if self.data['hall_of_fame']:
                best = self.data['hall_of_fame'][0].copy()
                best['genome'] = self._deserialize_genome(best.get('genome'))
                return best
            return None
        except Exception as e:
            print(f"Note: Could not get champion: {e}")
            return None
    
    def get_top_champions(self, n=10):
        """Get top N champions of all time"""
        try:
            self.data = self._load_or_create()
            
            top = []
            for champ in self.data['hall_of_fame'][:n]:
                champ_copy = champ.copy()
                champ_copy['genome'] = self._deserialize_genome(champ_copy.get('genome'))
                top.append(champ_copy)
            return top
        except Exception as e:
            print(f"Note: Could not get top champions: {e}")
            return []
    
    def get_strategies(self, recent=20):
        """Get recent strategies"""
        try:
            return self.data['strategies'][-recent:]
        except Exception as e:
            return []
    
    def get_history(self):
        """Get full history data"""
        return self.data.get('history', {})
    
    def get_behavior_stats(self):
        """Get behavioral statistics"""
        return self.data.get('behavior_stats', {
            'zigzag_tendency': [],
            'self_collision_rate': [],
            'avg_turn_frequency': []
        })
    
    def get_summary(self):
        """Get a summary of ancestral memory"""
        try:
            self.data = self._load_or_create()
            
            meta = self.data.get('metadata', {})
            current = self.data.get('current_state', {})
            hof = self.data.get('hall_of_fame', [])
            strategies = self.data.get('strategies', [])
            behavior = self.data.get('behavior_stats', {})
                        
            # Calculate top score from hall of fame
            top_score = 0
            top_gen = 0
            if hof:
                valid_scores = []
                for champ in hof:
                    score = champ.get('score', 0)
                    if isinstance(score, (int, float)):
                        valid_scores.append((score, champ.get('generation', 0)))
                    elif isinstance(score, list) and score:
                        valid_scores.append((float(score[0]), champ.get('generation', 0)))
                
                if valid_scores:
                    top_score, top_gen = max(valid_scores, key=lambda x: x[0])
            
            latest_strategy = 'None'
            if strategies and isinstance(strategies, list):
                last_strat = strategies[-1] if strategies else None
                if isinstance(last_strat, dict):
                    latest_strategy = last_strat.get('name', 'None')
            
            zigzag_trend = "N/A"
            if behavior and isinstance(behavior, dict):
                zigzag_list = behavior.get('zigzag_tendency', [])
                if zigzag_list and isinstance(zigzag_list, list):
                    recent = zigzag_list[-5:] if len(zigzag_list) >= 5 else zigzag_list
                    if recent:
                        valid_values = []
                        for item in recent:
                            if isinstance(item, dict):
                                val = item.get('value', 0)
                                if isinstance(val, (int, float)):
                                    valid_values.append(val)
                        if valid_values:
                            avg_zigzag = sum(valid_values) / len(valid_values)
                            zigzag_trend = f"{avg_zigzag:.2f}"
            
            self_collision = "N/A"
            if behavior and isinstance(behavior, dict):
                collision_list = behavior.get('self_collision_rate', [])
                if collision_list and isinstance(collision_list, list):
                    last_collision = collision_list[-1] if collision_list else None
                    if isinstance(last_collision, dict):
                        val = last_collision.get('value', 0)
                        if isinstance(val, (int, float)):
                            self_collision = f"{val:.1f}%"
            
            best_fitness = current.get('best_ever_fitness', 0)
            if best_fitness == float('-inf') or best_fitness == -1e9:
                best_fitness_display = "-inf"
            else:
                try:
                    if best_fitness > 1e6:
                        best_fitness_display = f"{best_fitness/1e6:.1f}M"
                    elif best_fitness > 1e3:
                        best_fitness_display = f"{best_fitness/1e3:.1f}K"
                    else:
                        best_fitness_display = f"{float(best_fitness):.2f}"
                except:
                    best_fitness_display = "0.00"
            
            summary = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                 🧬 ANCESTRAL MEMORY VAULT 🧬                 ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  📊 METADATA:
    ║     Created: {str(meta.get('created', 'Unknown'))[:10]}
    ║     Last Update: {str(meta.get('last_updated', 'Unknown'))[:10]}
    ║     Generations: {meta.get('total_generations', 0)}
    ║     Champions: {meta.get('total_champions', 0)}
    ║
    ║  🏆 CURRENT BEST:
    ║     Score: {current.get('best_ever_score', 0)}
    ║     Fitness: {best_fitness_display}
    ║     Generation: {current.get('generation', 0)}
    ║
    ║  🌟 HALL OF FAME:
    ║     Top Score: {int(top_score)} (Gen {int(top_gen) if top_gen else 0})
    ║     Total Champions: {len(hof)}
    ║
    ║  💡 STRATEGIES DISCOVERED: {len(strategies)}
    ║     Latest: {latest_strategy}
    ║
    ║  📈 BEHAVIOR TRENDS:
    ║     Zigzag Tendency: {zigzag_trend}
    ║     Self-Collision: {self_collision}
    ╚══════════════════════════════════════════════════════════════╝
    """
            return summary
        except Exception as e:
            return f"Error generating summary: {e}"
    
    def _save(self):
        """Save data to JSON file with retry logic"""
        if self._saving:
            return
        
        self._saving = True
        try:
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            
            clean_data = self._convert_to_python_types(self.data)
            
            temp_file = self.filename + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(clean_data, f, indent=2, ensure_ascii=False)
            
            if os.path.exists(self.filename):
                os.remove(self.filename)
            os.rename(temp_file, self.filename)
            
            time.sleep(0.1)
            
        except Exception as e:
            pass
        finally:
            self._saving = False
    
    def export_history(self, format='json'):
        """Export history in different formats"""
        if format == 'json':
            return self.data.get('history', {})
        elif format == 'csv':
            history = self.data.get('history', {})
            csv_lines = ["generation,best_fitness,avg_fitness,best_score,diversity"]
            gens = history.get('generations', [])
            best_f = history.get('best_fitness', [])
            avg_f = history.get('avg_fitness', [])
            best_s = history.get('best_scores', [])
            div = history.get('diversity', [])
            
            for i in range(len(gens)):
                csv_lines.append(
                    f"{gens[i]},"
                    f"{best_f[i] if i < len(best_f) else 0},"
                    f"{avg_f[i] if i < len(avg_f) else 0},"
                    f"{best_s[i] if i < len(best_s) else 0},"
                    f"{div[i] if i < len(div) else 0}"
                )
            return "\n".join(csv_lines)
        return None

class StrategyDetector:
    """
    Detects and records strategies during gameplay
    """
    
    def __init__(self, memory):
        self.memory = memory
        self.detected = set()
        self.milestones = set()
    
    def detect(self, game, generation, current_score):
        """Detect strategies from current game state"""
        strategies = []
        
        try:
            if isinstance(current_score, (list, np.ndarray)):
                score_val = int(current_score[0]) if len(current_score) > 0 else 0
            else:
                score_val = int(current_score) if current_score is not None else 0
        except:
            score_val = 0
        
        for milestone in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            if score_val >= milestone and milestone not in self.milestones:
                self.milestones.add(milestone)
                strategies.append({
                    'name': f"Score {milestone}",
                    'desc': f"Achieved {milestone} points!",
                    'milestone': milestone
                })
        
        if not hasattr(game, 'get_state'):
            return strategies
        
        try:
            state = game.get_state()
            head = state['head']
            food = state['food']
            snake_len = len(state['snake'])
            
            # Precision targeting
            if abs(head[0] - food[0]) < 100 and abs(head[1] - food[1]) < 100:
                key = "precision"
                if key not in self.detected:
                    self.detected.add(key)
                    strategies.append({
                        'name': "Precision Targeting",
                        'desc': "Direct approach to food discovered",
                        'milestone': None
                    })
            
            # Wall hugging
            if head[0] < 100 or head[0] > 600 or head[1] < 100 or head[1] > 600:
                key = "wall_hug"
                if key not in self.detected:
                    self.detected.add(key)
                    strategies.append({
                        'name': "Wall Hugging",
                        'desc': "Using walls for safety",
                        'milestone': None
                    })
            
            # Tail avoidance
            if snake_len > 10:
                key = "tail_avoid"
                if key not in self.detected:
                    self.detected.add(key)
                    strategies.append({
                        'name': "Tail Avoidance",
                        'desc': "Learned to avoid own tail",
                        'milestone': None
                    })
            
            # Space management
            if snake_len > 20:
                key = "space_mgmt"
                if key not in self.detected:
                    self.detected.add(key)
                    strategies.append({
                        'name': "Space Management",
                        'desc': "Efficient use of board space",
                        'milestone': None
                    })
            
            # Detect zigzag pattern
            if hasattr(game, 'last_direction') and hasattr(game, 'moves_made'):
                if len(game.moves_made) > 20:
                    # Check for alternating pattern
                    moves = game.moves_made[-20:]
                    alternations = 0
                    for i in range(1, len(moves)):
                        if moves[i] != moves[i-1]:
                            # Check if it's a perpendicular turn
                            if (moves[i-1] in ['up', 'down'] and moves[i] in ['left', 'right']) or \
                               (moves[i-1] in ['left', 'right'] and moves[i] in ['up', 'down']):
                                alternations += 1
                    
                    if alternations > 10 and "zigzag" not in self.detected:
                        self.detected.add("zigzag")
                        strategies.append({
                            'name': "Zigzag Pattern",
                            'desc': "Discovered efficient zigzag movement",
                            'milestone': None
                        })
            
        except Exception as e:
            pass
        
        # Record strategies
        for strat in strategies:
            try:
                self.memory.add_strategy(
                    generation=generation,
                    strategy_name=strat['name'],
                    description=strat['desc'],
                    score_milestone=strat['milestone']
                )
            except:
                pass
        
        return strategies

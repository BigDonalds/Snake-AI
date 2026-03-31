import threading
import os
import time
import msvcrt

from game import Game
from visualizer import SnakeVisualizer
from genetic_ai.genetic_algorithm import NeuralGeneticAlgorithm, SelectionMethod, ReplacementStrategy
from memory.ancestors_memory import AncestorsMemory, StrategyDetector

# Game Configuration
CONFIG = {
    'GAME_WIDTH': 700,
    'GAME_HEIGHT': 700,
    'SPEED': 30,
    'SQUARE_SIZE': 50,
    'BODY_PARTS': 3,
    'SNAKE_COLOR': "#00FF00",
    'FOOD_COLOR': "#FF0000",
    'BACKGROUND_COLOR': "#000000"
}

class SnakeEvolutionSystem:
    """
    Main system that runs training in background and handles on-demand visualization
    """
    
    def __init__(self):
        # Create ancestral memory directory if it doesn't exist
        os.makedirs("memory", exist_ok=True)
        os.makedirs("analysis", exist_ok=True)
        
        # Create ancestral memory
        self.memory = AncestorsMemory("memory/snake_ancestry.json")
        print(self.memory.get_summary())
        
        self.ai = NeuralGeneticAlgorithm(
            population_size=100,
            input_size=37,
            elite_size=5,
            mutation_rate=0.15,
            crossover_rate=0.7,
            tournament_size=3,
            use_speciation=True,
            selection_method=SelectionMethod.TOURNAMENT,
            replacement_strategy=ReplacementStrategy.ELITIST,
            archive_size=500,
            novelty_weight=0.1
        )
        
        # Try to load previous memory
        if os.path.exists("memory/snake_ancestry.json"):
            self.ai.load_checkpoint()
            print(f"📀 Loaded checkpoint - Generation {self.ai.generation}, Best Score: {self.ai.best_ever_score}")
        else:
            print("🆕 Starting fresh lineage...")
        
        # Create strategy detector
        self.strategy_detector = StrategyDetector(self.memory)
        
        # Training control
        self.training_active = True
        self.training_thread = None
        self.game = None
        
        # Visualization control
        self.visualizer_active = False
        self.visualizer_window = None
        
        # Statistics
        self.start_time = time.time()
        self.episodes_completed = 0
        self.last_stats_time = time.time()
        self.stats_interval = 30  # Print stats every 30 seconds
        
        print("\n" + "="*60)
        print("🐍 SNAKE AI EVOLUTION SYSTEM - TRAINING MODE 🐍")
        print("="*60)
        print(f"Current Best Score: {self.ai.best_ever_score}")
        print(f"Generation: {self.ai.generation}")
        print(f"Input Size: {self.ai.input_size}")
        print(f"Population Size: {self.ai.population_size}")
        print("\nControls:")
        print("  • Press 'v' key to open visualizer")
        print("  • Press Ctrl+C to quit")
        print("="*60 + "\n")
        print("Training started...\n")
    
    def check_keyboard(self):
        while self.training_active:
            if msvcrt.kbhit():
                try:
                    key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                    if key == 'v' and not self.visualizer_active:
                        # Open visualizer in a separate thread so it doesn't block
                        viz_thread = threading.Thread(target=self.run_visualizer, daemon=True)
                        viz_thread.start()
                except Exception as e:
                    pass
            time.sleep(0.1)
    
    def get_elapsed_time(self):
        """Get formatted elapsed time"""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def run_visualizer(self):
        """Run the visualizer"""
        if self.visualizer_active:
            return
        
        self.visualizer_active = True
        print(f"\n🎥 Opening visualizer...")
        
        # Create a game instance for visualization
        visual_game = Game(CONFIG, ai_algorithm=self.ai, training_mode=True)
        
        # Store original state
        original_pop = self.ai.population.copy() if self.ai.population else []
        original_idx = self.ai.current_individual_idx
        
        # Create visualizer
        visualizer = SnakeVisualizer(visual_game, self.ai, CONFIG, 
                                    self.memory, self.strategy_detector)
        
        # Run visualizer
        try:
            visualizer.create_window()
            visualizer.window.mainloop()
        except Exception as e:
            print(f"Visualizer error: {e}")
        finally:
            # Restore original population
            if original_pop:
                self.ai.population = original_pop
                self.ai.current_individual_idx = original_idx
            self.visualizer_active = False
            print("\n✅ Visualizer closed. Training continues...")
            print("Press 'v' to open visualizer | Ctrl+C to quit")
    
    def print_periodic_stats(self):
        """Print periodic training statistics"""
        current_time = time.time()
        if current_time - self.last_stats_time > self.stats_interval:
            # Get current stats
            stats = self.ai.get_stats() if hasattr(self.ai, 'get_stats') else {}
            avg_score = stats.get('avg_score', 0)
            avg_survival = stats.get('avg_survival', 0)
            scoring_count = stats.get('scoring_count', 0)
            wall_deaths = stats.get('wall_deaths', 0)
            diversity = stats.get('diversity', 0)
            
            # Clear line and print stats
            print(f"\r📊 Gen {self.ai.generation} | Best: {self.ai.best_ever_score} | "
                  f"Avg Score: {avg_score:.1f} | Scoring: {scoring_count}/{self.ai.population_size} | "
                  f"Survival: {avg_survival:.1f} steps | Wall Deaths: {wall_deaths} | "
                  f"Diversity: {diversity:.3f} | Episodes: {self.episodes_completed} | "
                  f"Time: {self.get_elapsed_time()}", end="", flush=True)
            
            # Add newline if training has been going on for a while
            if self.episodes_completed % 100 == 0:
                print()
            
            self.last_stats_time = current_time
    
    def training_loop(self):
        """Main training loop running in background"""
        # Create game for training
        self.game = Game(CONFIG, ai_algorithm=self.ai, training_mode=True)
        
        episode = 0
        
        try:
            while self.training_active:
                # Run one episode
                self.game.reset()
                episode += 1
                self.episodes_completed += 1
                
                # Run until game over
                while not self.game.game_over_flag and self.training_active:
                    self.game.step()
                
                # Print periodic stats
                self.print_periodic_stats()
                
                time.sleep(0.001)
                
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.training_active = False
    
    def print_status(self):
        """Print current training status"""
        stats = self.ai.get_stats() if hasattr(self.ai, 'get_stats') else {}
        print("\n" + "="*60)
        print("📊 TRAINING STATUS")
        print("="*60)
        print(f"Generation: {self.ai.generation}")
        print(f"Best Score: {self.ai.best_ever_score}")
        print(f"Best Fitness: {self.ai.best_ever_fitness:.2f}")
        print(f"Avg Score: {stats.get('avg_score', 0):.2f}")
        print(f"Avg Fitness: {stats.get('avg_fitness', 0):.2f}")
        print(f"Avg Survival: {stats.get('avg_survival', 0):.1f} steps")
        print(f"Scoring Individuals: {stats.get('scoring_count', 0)}/{self.ai.population_size}")
        print(f"Wall Deaths: {stats.get('wall_deaths', 0)}")
        print(f"Diversity: {stats.get('diversity', 0):.3f}")
        print(f"Species: {stats.get('species_count', 0)}")
        print(f"Stagnation: {stats.get('stagnation', 0)} gens")
        print(f"Episodes: {self.episodes_completed}")
        print(f"Runtime: {self.get_elapsed_time()}")
        print("="*60)
    
    def save_and_exit(self):
        """Save progress and exit"""
        print("\n\n💾 Saving to ancestral memory...")
        
        print()
        
        # Save checkpoint
        self.ai.save_checkpoint()
        
        # Print final stats
        self.print_status()
        
        # Create a fresh memory instance to ensure we read the latest file
        final_memory = AncestorsMemory("memory/snake_ancestry.json")
        print("\n" + final_memory.get_summary())
        print("👋 Goodbye!")
    
    def start(self):
        """Start the system"""
        # Start training in background thread
        self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.training_thread.start()
        
        # Start keyboard listener thread
        keyboard_thread = threading.Thread(target=self.check_keyboard, daemon=True)
        keyboard_thread.start()
        
        # Main loop - just keep alive and handle Ctrl+C
        try:
            while self.training_active:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⏸️ Shutting down...")
        finally:
            self.training_active = False
            time.sleep(0.5)
            self.save_and_exit()

def main():
    system = SnakeEvolutionSystem()
    system.start()

if __name__ == "__main__":
    main()
import tkinter as tk
import random
from datetime import datetime

from snake import Snake
from analysis.run_data import RunData, MoveRecord, DeathReason

class Food:
    def __init__(self, snake, config):
        self.config = config
        self.coordinates = []
        self.respawn(snake)
    
    def respawn(self, snake):
        """Respawn food at a random location not occupied by snake and within boundaries"""
        square_size = self.config['SQUARE_SIZE']
        game_width = self.config['GAME_WIDTH']
        game_height = self.config['GAME_HEIGHT']
        
        # Calculate grid dimensions
        grid_width = game_width // square_size
        grid_height = game_height // square_size
        
        # Get all occupied positions
        occupied = snake.get_body_positions()
        
        # Collect all valid grid positions
        valid_positions = []
        for x in range(grid_width):
            for y in range(grid_height):
                pos = [x * square_size, y * square_size]
                if pos not in occupied:
                    valid_positions.append(pos)
        
        # If there are valid positions, choose randomly
        if valid_positions:
            self.coordinates = random.choice(valid_positions)
        else:
            # If snake fills the entire grid (impossible win condition), place food at a default safe position
            # This should never happen in normal gameplay
            self.coordinates = [0, 0]

class Game:
    def __init__(self, config, ai_algorithm=None, training_mode=False):
        self.config = config
        self.training_mode = training_mode
        self.ai = ai_algorithm
        
        # Initialize game state FIRST (before any display setup)
        self.snake = None
        self.food = None
        self.score = 0
        self.steps_without_food = 0
        self.max_steps_without_food = 200
        self.game_over_flag = False
        self.last_direction = 'down'
        self.current_individual_idx = 0
        self.last_positions = []
        self.max_history = 10
        
        # Track all positions visited
        self.steps = 0  # Total steps in current episode
        self.position_history = []  # Track all positions visited
        self.moves_made = []  # Track moves for strategy detection
        self.last_action_scores = None  # Store AI decision scores
        self.food_eaten_this_step = False  # Track if food was eaten
        
        # Move history for visualizer
        self.move_history = []  # List of move records for current run
        
        # Run data tracking
        self.current_run_data = None
        self.current_step = 0
        
        # Now setup display if needed
        if not training_mode:
            self._setup_display()
        else:
            self.window = None
            self.canvas = None
            self.label = None
        
        # Initialize the game
        self.reset()
    
    def _setup_display(self):
        """Setup the display for visual mode"""
        self.window = tk.Tk()
        self.window.title("Snake Game AI - Genetic Algorithm")
        self.window.resizable(False, False)
        
        self.label = tk.Label(
            self.window, 
            text=f"Score:{self.score}", 
            font=('consolas', 40)
        )
        self.label.pack()
        
        self.canvas = tk.Canvas(
            self.window, 
            bg=self.config['BACKGROUND_COLOR'], 
            height=self.config['GAME_HEIGHT'], 
            width=self.config['GAME_WIDTH']
        )
        self.canvas.pack()
        
        # Center window
        self.window.update()
        window_width = self.window.winfo_width()
        window_height = self.window.winfo_height()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        
        x = int((screen_width/2) - (window_width/2))
        y = int((screen_height/2) - (window_height/2))
        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    def reset(self):
        """Reset the game to initial state with snake in center of map"""
        # Create snake with proper centered spawn
        self.snake = Snake(self.config, centered=True)
        self.food = Food(self.snake, self.config)
        self.score = 0
        self.steps_without_food = 0
        self.game_over_flag = False
        self.last_direction = 'down'
        self.last_positions = []
        self.current_step = 0
        self.steps = 0  # Reset steps counter
        self.position_history = []  # Clear position history
        self.moves_made = []  # Clear moves made
        self.last_action_scores = None
        self.food_eaten_this_step = False
        self.move_history = []  # Clear move history for new run
        
        # Record initial position
        initial_head = self.snake.get_head_position()
        self.position_history.append(tuple(initial_head))
        
        # Create new run data for this episode
        if self.ai and hasattr(self.ai, 'population') and self.current_individual_idx < len(self.ai.population):
            individual = self.ai.population[self.current_individual_idx]
            agent_id = f"gen{individual.generation}_ind{self.current_individual_idx}_{datetime.now().strftime('%H%M%S')}"
            self.current_run_data = RunData(
                agent_id=agent_id,
                generation=self.ai.generation,
                genome=individual.genome
            )
        
        if not self.training_mode and self.canvas:
            self._draw_initial()
    
    def _draw_initial(self):
        """Draw initial game objects"""
        self.canvas.delete("all")
        # Draw snake
        for x, y in self.snake.get_body_positions():
            self.canvas.create_rectangle(
                x, y, x + self.config['SQUARE_SIZE'], y + self.config['SQUARE_SIZE'],
                fill=self.config['SNAKE_COLOR'], tag="snake"
            )
        
        # Draw food
        self.canvas.create_oval(
            self.food.coordinates[0], self.food.coordinates[1],
            self.food.coordinates[0] + self.config['SQUARE_SIZE'],
            self.food.coordinates[1] + self.config['SQUARE_SIZE'],
            fill=self.config['FOOD_COLOR'], tag="food"
        )
        
        self.label.config(text=f"Score:{self.score}")
    
    def update_display(self):
        """Update the display for visual mode"""
        if self.training_mode or not self.canvas:
            return
        
        self.canvas.delete("all")
        
        # Draw snake
        for x, y in self.snake.get_body_positions():
            self.canvas.create_rectangle(
                x, y, x + self.config['SQUARE_SIZE'], y + self.config['SQUARE_SIZE'],
                fill=self.config['SNAKE_COLOR'], tag="snake"
            )
        
        # Draw food
        self.canvas.create_oval(
            self.food.coordinates[0], self.food.coordinates[1],
            self.food.coordinates[0] + self.config['SQUARE_SIZE'],
            self.food.coordinates[1] + self.config['SQUARE_SIZE'],
            fill=self.config['FOOD_COLOR'], tag="food"
        )
        
        self.label.config(text=f"Score:{self.score}")
    
    def check_boundary_collision(self, head_x, head_y):
        """Check if snake hits the boundary"""
        if head_x < 0 or head_x >= self.config['GAME_WIDTH'] or \
           head_y < 0 or head_y >= self.config['GAME_HEIGHT']:
            return True
        return False
    
    def is_valid_direction(self, new_direction):
        """Check if direction change is valid (can't reverse)"""
        opposites = {
            'up': 'down',
            'down': 'up',
            'left': 'right',
            'right': 'left'
        }
        return new_direction != opposites.get(self.last_direction, '')
    
    def record_move(self, direction, head_x, head_y, action_scores=None, food_eaten=False):
        """Record the current state as a move in history"""
        move_record = {
            'step': self.current_step,
            'direction': direction,
            'head_position': (head_x, head_y),
            'food_position': (self.food.coordinates[0], self.food.coordinates[1]),
            'score': self.score,
            'snake_length': len(self.snake.get_body_positions()),
            'action_scores': action_scores,
            'food_eaten': food_eaten
        }
        self.move_history.append(move_record)
        return move_record
    
    def step(self, direction=None):
        """
        Execute one game step
        Returns: (reward, game_over, score)
        """
        if self.game_over_flag:
            return -10, True, self.score
        
        # Get head position before moving
        head_before = self.snake.get_head_position()
        
        # Track position history
        self.last_positions.append(head_before)
        if len(self.last_positions) > self.max_history:
            self.last_positions.pop(0)
        
        # Track all positions visited
        self.position_history.append(tuple(head_before))
        if len(self.position_history) > 1000:
            self.position_history = self.position_history[-1000:]
        
        # Get AI decision if no direction provided
        action_scores = None
        if direction is None and self.ai:
            direction = self.ai.get_action(self)
            if direction and not self.is_valid_direction(direction):
                direction = self.last_direction
            # Get action scores from AI
            if hasattr(self.ai, 'last_action_scores'):
                action_scores = self.ai.last_action_scores
        elif direction is None:
            direction = self.last_direction
        
        # Track moves made for strategy detection
        if direction:
            self.moves_made.append(direction)
            if len(self.moves_made) > 500:
                self.moves_made = self.moves_made[-500:]
        
        self.last_direction = direction
        
        # Move snake
        head_x, head_y = self.snake.move(direction)
        self.steps_without_food += 1
        self.current_step += 1
        self.steps += 1
        
        # Reset food eaten flag
        self.food_eaten_this_step = False
        
        # Check for death
        death_reason = None
        death_position = (head_x, head_y)
        
        if self.check_boundary_collision(head_x, head_y):
            self.game_over_flag = True
            death_reason = "wall"
        elif self.snake.check_self_collision():
            self.game_over_flag = True
            death_reason = "self"
        
        # Check food collision
        ate_food = (head_x == self.food.coordinates[0] and 
                   head_y == self.food.coordinates[1])
        
        if ate_food:
            self.score += 1
            self.steps_without_food = 0
            self.food_eaten_this_step = True
            self.food.respawn(self.snake)
            reward = 10
            
            # Record food eaten
            if self.current_run_data:
                self.current_run_data.add_food_eaten((head_x, head_y))
                
                # Check for milestones
                if self.score % 5 == 0:
                    self.current_run_data.add_milestone(self.score)
        else:
            self.snake.remove_tail()
            reward = -0.1
        
        # Penalize for taking too long without eating
        if self.steps_without_food > self.max_steps_without_food:
            self.game_over_flag = True
            death_reason = "timeout"
            reward = -5
        
        # Bonus for exploring new areas
        if self._is_exploring():
            reward += 0.05
        
        # RECORD THE MOVE IN HISTORY (for visualizer)
        self.record_move(
            direction=direction,
            head_x=head_x,
            head_y=head_y,
            action_scores=action_scores,
            food_eaten=ate_food
        )
        
        # Record in RunData if available
        if self.current_run_data and not self.game_over_flag:
            move_record = MoveRecord(
                step=self.current_step,
                direction=direction,
                head_position=(head_x, head_y),
                food_position=(self.food.coordinates[0], self.food.coordinates[1]),
                score=self.score,
                snake_length=len(self.snake.get_body_positions()),
                action_scores=action_scores
            )
            self.current_run_data.add_move(move_record)
        
        # If game over, end the run
        if self.game_over_flag and self.current_run_data:
            death_reason_enum = None
            if death_reason == "wall":
                death_reason_enum = DeathReason.WALL_COLLISION
            elif death_reason == "self":
                death_reason_enum = DeathReason.SELF_COLLISION
            elif death_reason == "timeout":
                death_reason_enum = DeathReason.STARVATION
            
            # End the run with final data
            self.current_run_data.end_run(
                score=self.score,
                fitness=0,  # Will be updated by AI
                death_reason=death_reason_enum,
                death_position=death_position
            )
            
            # Pass to AI's end_episode with string death reason
            if self.ai and hasattr(self.ai, 'end_episode'):
                self.ai.end_episode(
                    self, 
                    reward, 
                    self.score, 
                    death_reason=death_reason,
                    death_position=death_position
                )
        
        # Update display if in visual mode
        self.update_display()
        
        return reward, self.game_over_flag, self.score
    
    def _is_exploring(self):
        """Check if snake is exploring new territory"""
        if len(self.last_positions) < 2:
            return False
        
        current = self.snake.get_head_position()
        return current not in self.last_positions[:-1]
    
    def get_state(self):
        """Get current game state for AI"""
        return {
            'snake': self.snake.get_body_positions(),
            'head': self.snake.get_head_position(),
            'food': self.food.coordinates,
            'score': self.score,
            'direction': self.last_direction,
            'steps_without_food': self.steps_without_food,
            'game_width': self.config['GAME_WIDTH'],
            'game_height': self.config['GAME_HEIGHT'],
            'square_size': self.config['SQUARE_SIZE']
        }
    
    def get_features(self):
        """Extract features for genetic algorithm"""
        state = self.get_state()
        head_x, head_y = state['head']
        food_x, food_y = state['food']
        
        features = []
        
        # 1. Distance to food (normalized)
        food_dx = (food_x - head_x) / state['game_width']
        food_dy = (food_y - head_y) / state['game_height']
        features.extend([food_dx, food_dy])
        
        # 2. Manhattan distance to food
        manhattan = (abs(food_x - head_x) + abs(food_y - head_y)) / state['square_size']
        features.append(min(manhattan / 20, 1.0))
        
        # 3. Danger in each direction
        for direction in ['up', 'down', 'left', 'right']:
            safety = self.snake.get_safety_rating(
                direction, 
                state['game_width'], 
                state['game_height']
            )
            features.append(1 - safety)
        
        # 4. Distance to walls
        features.append(head_x / state['game_width'])
        features.append((state['game_width'] - head_x) / state['game_width'])
        features.append(head_y / state['game_height'])
        features.append((state['game_height'] - head_y) / state['game_height'])
        
        # 5. Snake length
        max_possible = (state['game_width'] // state['square_size']) * \
                      (state['game_height'] // state['square_size'])
        features.append(len(state['snake']) / max_possible)
        
        # 6. Food direction
        if food_x > head_x:
            features.extend([1, 0, 0, 0])
        elif food_x < head_x:
            features.extend([0, 1, 0, 0])
        elif food_y > head_y:
            features.extend([0, 0, 1, 0])
        elif food_y < head_y:
            features.extend([0, 0, 0, 1])
        else:
            features.extend([0, 0, 0, 0])
        
        # 7. Current direction
        if state['direction'] == 'up':
            features.extend([1, 0, 0, 0])
        elif state['direction'] == 'down':
            features.extend([0, 1, 0, 0])
        elif state['direction'] == 'left':
            features.extend([0, 0, 1, 0])
        elif state['direction'] == 'right':
            features.extend([0, 0, 0, 1])
        else:
            features.extend([0, 0, 0, 0])
        
        # 8. Free space count
        free_space = self._count_surrounding_free()
        features.append(free_space)
        
        # 9. Steps without food
        features.append(min(state['steps_without_food'] / 100, 1.0))
        
        return features
    
    def _count_surrounding_free(self):
        """Count free spaces in immediate surroundings"""
        state = self.get_state()
        head_x, head_y = state['head']
        square_size = state['square_size']
        
        directions = [
            (0, -square_size), (0, square_size),
            (-square_size, 0), (square_size, 0)
        ]
        
        free_count = 0
        for dx, dy in directions:
            new_x, new_y = head_x + dx, head_y + dy
            
            if (0 <= new_x < state['game_width'] and 
                0 <= new_y < state['game_height']):
                if [new_x, new_y] not in state['snake'][1:]:
                    free_count += 1
        
        return free_count / 4 if free_count > 0 else 0
    
    def start(self):
        """Start the game loop"""
        if self.training_mode and self.ai:
            self._training_loop()
        else:
            self._visual_loop()
    
    def _training_loop(self):
        """Run training loop without display"""
        episode = 0
        
        try:
            while True:
                self.reset()
                if hasattr(self.ai, 'population') and self.ai.population:
                    self.current_individual_idx = episode % len(self.ai.population)
                total_reward = 0
                steps = 0
                
                while not self.game_over_flag:
                    reward, game_over, score = self.step()
                    total_reward += reward
                    steps += 1
                    
                    if game_over:
                        break
                
                episode += 1
                
                if episode % 100 == 0:
                    print(f"Completed {episode} episodes...")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
    
    def _visual_loop(self):
        """Run visual loop with display"""
        def game_loop():
            if not self.game_over_flag:
                self.step()
                self.window.after(self.config['SPEED'], game_loop)
        
        game_loop()
        self.window.mainloop()
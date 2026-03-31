import tkinter as tk
from tkinter import ttk
from collections import deque
from typing import Optional, List, Dict, Any, Tuple

from analysis.run_data import RunData, MoveRecord

class SnakeVisualizer:
    
    def __init__(self, game, ai, config, memory=None, strategy_detector=None):
        self.game = game
        self.ai = ai
        self.config = config
        self.memory = memory
        self.strategy_detector = strategy_detector
        
        # Replay state
        self.current_run: Optional[RunData] = None
        self.current_move_index = 0
        self.playing = False
        self.paused = False
        self.speed_multiplier = 1.0
        self.game_speed = 150  # ms per frame
        
        # Snake body positions (reconstructed from move history)
        self.snake_body: List[Tuple[int, int]] = []
        
        # Position history for route dots (separate from snake body)
        self.route_history = deque(maxlen=20)
        
        # UI elements
        self.window = None
        self.canvas = None
        
        # Colors
        self.colors = {
            'bg': '#2b2b2b',
            'snake_head': '#00ff00',
            'snake_body': '#00aa00',
            'food': '#ff4444',
            'text': '#ffffff',
            'log_bg': '#1a1a1a',
            'log_text': '#00ff00',
            'warning': '#ffff00',
            'discovery': '#ffaa00',
            'danger': '#ff4444',
            'safe': '#00ff00',
            'control_bg': '#1a1a1a',
            'button_bg': '#444444',
            'info_bg': '#333333'
        }
        
    def create_window(self):
        self.window = tk.Tk()
        self.window.title("🐍 Snake AI Replay Visualizer")
        self.window.configure(bg=self.colors['bg'])
        self.window.geometry("1000x800")
        self.window.minsize(900, 700)
        
        # Show selection screen first
        self._show_selection_screen()
        
        # Bind Ctrl+C to close
        self.window.bind('<Control-c>', lambda e: self.close())
        self.window.bind('<Control-C>', lambda e: self.close())
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        
        return self.window
    
    def run(self):
        """Alternative entry point"""
        self.create_window()
        self.window.mainloop()
    
    def _show_selection_screen(self):
        """Show the initial selection screen"""
        for widget in self.window.winfo_children():
            widget.destroy()
        
        # Main frame
        main_frame = tk.Frame(self.window, bg=self.colors['bg'])
        main_frame.pack(expand=True, fill=tk.BOTH, padx=50, pady=50)
        
        # Title
        title_label = tk.Label(main_frame, 
                               text="🐍 Snake AI Replay Visualizer",
                               bg=self.colors['bg'],
                               fg=self.colors['snake_head'],
                               font=('Arial', 24, 'bold'))
        title_label.pack(pady=20)
        
        # Selection buttons frame
        button_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        button_frame.pack(pady=30)
        
        # Check if AI has generation_stats
        has_stats = hasattr(self.ai, 'generation_stats') and self.ai.generation_stats
        
        if has_stats and self.ai.generation_stats:
            latest_stats = self.ai.generation_stats[-1]
            
            # Best run button
            if hasattr(latest_stats, 'best_run') and latest_stats.best_run:
                best_btn = tk.Button(button_frame,
                                     text="🏆 Best Agent",
                                     command=lambda: self._start_replay(latest_stats.best_run, "Best Agent"),
                                     bg='#aa5500', fg='white',
                                     font=('Arial', 12, 'bold'),
                                     width=20, height=2)
                best_btn.pack(pady=8)
            
            # Worst run button
            if hasattr(latest_stats, 'worst_run') and latest_stats.worst_run:
                worst_btn = tk.Button(button_frame,
                                      text="💀 Worst Agent",
                                      command=lambda: self._start_replay(latest_stats.worst_run, "Worst Agent"),
                                      bg='#aa0000', fg='white',
                                      font=('Arial', 12, 'bold'),
                                      width=20, height=2)
                worst_btn.pack(pady=8)
            
        else:
            no_stats_label = tk.Label(button_frame,
                                     text="No replay data available yet.\nRun the AI first to generate some runs.",
                                     bg=self.colors['bg'],
                                     fg=self.colors['warning'],
                                     font=('Arial', 14),
                                     justify=tk.CENTER)
            no_stats_label.pack(pady=30)
        
        quit_btn = tk.Button(main_frame,
                            text="Quit",
                            command=self.close,
                            bg='#aa0000', fg='white',
                            font=('Arial', 12),
                            width=15)
        quit_btn.pack(pady=20)
    
    def _reconstruct_snake_body(self, up_to_step: int):
        """Reconstruct the snake body from move history up to a specific step"""
        if not self.current_run or not self.current_run.moves:
            return []
        
        # Get all moves up to current step
        moves = self.current_run.moves[:up_to_step + 1]
        if not moves:
            return []
        
        # Get the snake length at this step
        current_move = moves[-1]
        snake_length = current_move.snake_length
        
        # Extract all head positions from moves
        all_positions = []
        for move in moves:
            # Handle both object and dictionary formats
            if hasattr(move, 'head_position'):
                all_positions.append(move.head_position)
            elif isinstance(move, dict):
                all_positions.append(move.get('head_position', (0, 0)))
            else:
                all_positions.append((0, 0))
        
        # The snake body is the last 'snake_length' positions
        if len(all_positions) >= snake_length:
            body = all_positions[-snake_length:]
        else:
            body = all_positions.copy()
            while len(body) < snake_length:
                body.insert(0, body[0])
        
        return body
    
    def _start_replay(self, run_data: RunData, title: str):
        """Start replaying a run"""
        self.current_run = run_data
        self.current_move_index = 0
        self.snake_body = []
        self.route_history.clear()
        
        # Clear window and show replay interface
        for widget in self.window.winfo_children():
            widget.destroy()
        
        # Create replay layout
        self._create_replay_layout(title)
        
        # Initialize with first move
        if self.current_run and self.current_run.moves:
            self._apply_replay_move(0)
            
            self.play()
    
    def _create_replay_layout(self, title: str):
        """Create the replay viewing layout"""
        # Configure grid
        self.window.grid_rowconfigure(0, weight=0)  # Control bar
        self.window.grid_rowconfigure(1, weight=0)  # Progress bar
        self.window.grid_rowconfigure(2, weight=1)  # Content
        self.window.grid_columnconfigure(0, weight=1)
        
        # Top control bar
        control_frame = tk.Frame(self.window, bg=self.colors['control_bg'], height=60)
        control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        control_frame.grid_propagate(False)
        
        # Title
        title_label = tk.Label(control_frame, 
                              text=f"🎬 {title}",
                              bg=self.colors['control_bg'],
                              fg=self.colors['snake_head'],
                              font=('Arial', 12, 'bold'))
        title_label.pack(side=tk.LEFT, padx=10)
        
        # Control buttons
        self.play_btn = tk.Button(control_frame, text="▶ Play", command=self.play,
                                  bg='#00aa00', fg='white', width=8)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        
        self.pause_btn = tk.Button(control_frame, text="⏸ Pause", command=self.pause,
                                   bg='#aa5500', fg='white', width=8)
        self.pause_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = tk.Button(control_frame, text="⏹ Stop", command=self.stop,
                                  bg='#aa0000', fg='white', width=8)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        # Speed buttons
        speed_frame = tk.Frame(control_frame, bg=self.colors['control_bg'])
        speed_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(speed_frame, text="Speed:", bg=self.colors['control_bg'], 
                fg=self.colors['text'], font=('Arial', 10)).pack(side=tk.LEFT)
        
        self.speed_1x_btn = tk.Button(speed_frame, text="1x", 
                                      command=lambda: self.set_speed(1.0),
                                      bg='#00aa00', fg='white', width=3)
        self.speed_1x_btn.pack(side=tk.LEFT, padx=2)
        
        self.speed_3x_btn = tk.Button(speed_frame, text="3x", 
                                      command=lambda: self.set_speed(3.0),
                                      bg='#444444', fg='white', width=3)
        self.speed_3x_btn.pack(side=tk.LEFT, padx=2)
        
        self.speed_5x_btn = tk.Button(speed_frame, text="5x", 
                                      command=lambda: self.set_speed(5.0),
                                      bg='#444444', fg='white', width=3)
        self.speed_5x_btn.pack(side=tk.LEFT, padx=2)
        
        self.speed_7x_btn = tk.Button(speed_frame, text="7x", 
                                      command=lambda: self.set_speed(7.0),
                                      bg='#444444', fg='white', width=3)
        self.speed_7x_btn.pack(side=tk.LEFT, padx=2)
        
        self.speed_label = tk.Label(control_frame, text=f"Current: 1.0x", 
                                    bg=self.colors['control_bg'], fg=self.colors['text'],
                                    font=('Arial', 10, 'bold'))
        self.speed_label.pack(side=tk.RIGHT, padx=10)
        
        back_btn = tk.Button(control_frame, text="← Back to Selection", 
                             command=self._show_selection_screen,
                             bg='#444444', fg='white', width=15)
        back_btn.pack(side=tk.RIGHT, padx=10)
        
        # Progress bar
        progress_frame = tk.Frame(self.window, bg=self.colors['bg'], height=40)
        progress_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        progress_frame.grid_propagate(False)
        
        self.progress_label = tk.Label(progress_frame, text="Step: 0/0", 
                                       bg=self.colors['bg'], fg=self.colors['text'],
                                       font=('Arial', 10))
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                            maximum=100, length=500)
        self.progress_bar.pack(side=tk.LEFT, padx=10)
        
        self.step_back_btn = tk.Button(progress_frame, text="⏪ Step Back", 
                                       command=self.step_back,
                                       bg='#444444', fg='white', width=10)
        self.step_back_btn.pack(side=tk.LEFT, padx=2)
        
        self.step_forward_btn = tk.Button(progress_frame, text="⏩ Step Forward", 
                                          command=self.step_forward,
                                          bg='#444444', fg='white', width=10)
        self.step_forward_btn.pack(side=tk.LEFT, padx=2)
        
        # Main content area
        main_paned = tk.PanedWindow(self.window, orient=tk.HORIZONTAL, 
                                     bg=self.colors['bg'], sashwidth=5)
        main_paned.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        
        # Left frame - Game canvas
        left_frame = tk.Frame(main_paned, bg=self.colors['bg'])
        main_paned.add(left_frame, width=650, minsize=400)
        
        canvas_frame = tk.Frame(left_frame, bg=self.colors['bg'])
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            bg='black',
            highlightthickness=2,
            highlightbackground='#444444'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        score_frame = tk.Frame(left_frame, bg=self.colors['bg'], height=40)
        score_frame.pack(fill=tk.X, pady=5)
        score_frame.pack_propagate(False)
        
        self.score_label = tk.Label(score_frame, text="Score: 0", 
                                    bg=self.colors['bg'], fg=self.colors['text'],
                                    font=('Arial', 14, 'bold'))
        self.score_label.pack(side=tk.LEFT, padx=10)
        
        # Right frame - Info panels
        right_frame = tk.Frame(main_paned, bg=self.colors['bg'], width=300)
        main_paned.add(right_frame, width=300, minsize=250)
        right_frame.pack_propagate(False)
        
        # Scrollable right frame
        canvas = tk.Canvas(right_frame, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = tk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add info panels
        self._create_agent_info_panel(scrollable_frame)
        self._create_death_panel(scrollable_frame)
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Keyboard shortcuts
        self.window.bind('<space>', lambda e: self.toggle_pause())
        self.window.bind('<Right>', lambda e: self.step_forward())
        self.window.bind('<Left>', lambda e: self.step_back())
        self.window.bind('<Home>', lambda e: self.go_to_start())
        self.window.bind('<End>', lambda e: self.go_to_end())
        self.window.bind('<Control-c>', lambda e: self.close())
        self.window.bind('<Control-C>', lambda e: self.close())
    
    def _create_agent_info_panel(self, parent):
        frame = tk.LabelFrame(parent, text="🧬 Agent Information", 
                             bg=self.colors['bg'], fg=self.colors['text'],
                             font=('Arial', 10, 'bold'))
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.agent_info = tk.Text(frame, height=6,
                                   bg=self.colors['info_bg'],
                                   fg=self.colors['text'],
                                   font=('Courier', 9),
                                   wrap=tk.WORD)
        self.agent_info.pack(fill=tk.X, padx=5, pady=5)
    
    def _create_death_panel(self, parent):
        frame = tk.LabelFrame(parent, text="💀 Death Analysis", 
                             bg=self.colors['bg'], fg=self.colors['text'],
                             font=('Arial', 10, 'bold'))
        frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.death_info = tk.Text(frame, height=4,
                                   bg=self.colors['info_bg'],
                                   fg=self.colors['text'],
                                   font=('Courier', 9),
                                   wrap=tk.WORD)
        self.death_info.pack(fill=tk.X, padx=5, pady=5)
    
    def _apply_replay_move(self, move_index: int):
        """Apply a specific replay move"""
        if not self.current_run or move_index >= len(self.current_run.moves):
            return
        
        move = self.current_run.moves[move_index]
        self.current_move_index = move_index
        
        # Reconstruct the snake body at this point
        self.snake_body = self._reconstruct_snake_body(move_index)
        
        # Add to route history (for trail effect, separate from snake body)
        self.route_history.append(move.head_position)
        
        # Update display
        self._draw_replay_state(move)
        self._update_info_displays()
        self._update_progress()
    
    def _draw_replay_state(self, move: MoveRecord):
        """Draw the current replay state with proper snake body"""
        if not self.canvas:
            return
        
        self.canvas.delete("all")
        
        # Get current canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Calculate scale factor
        game_width = self.config['GAME_WIDTH']
        game_height = self.config['GAME_HEIGHT']
        
        scale_x = canvas_width / game_width
        scale_y = canvas_height / game_height
        scale = min(scale_x, scale_y)
        
        # Center the game
        offset_x = (canvas_width - game_width * scale) / 2
        offset_y = (canvas_height - game_height * scale) / 2
        square = self.config['SQUARE_SIZE'] * scale
        
        # Draw grid
        for x in range(0, game_width, self.config['SQUARE_SIZE']):
            x1 = offset_x + x * scale
            self.canvas.create_line(x1, offset_y, x1, offset_y + game_height * scale, 
                                   fill='#333333', width=1)
        for y in range(0, game_height, self.config['SQUARE_SIZE']):
            y1 = offset_y + y * scale
            self.canvas.create_line(offset_x, y1, offset_x + game_width * scale, y1,
                                   fill='#333333', width=1)
        
        # Draw route trail (separate from snake body)
        if len(self.route_history) > 1:
            route_points = []
            for pos in list(self.route_history)[:-1]:  # Exclude current head
                px = offset_x + pos[0] * scale + square/2
                py = offset_y + pos[1] * scale + square/2
                route_points.append((px, py))
            
            # Draw connecting lines for trail
            for i in range(len(route_points) - 1):
                self.canvas.create_line(
                    route_points[i][0], route_points[i][1],
                    route_points[i+1][0], route_points[i+1][1],
                    fill='#00ff00',
                    width=1,
                    dash=(2, 2)
                )
            
            # Draw trail dots
            for i, (px, py) in enumerate(route_points):
                alpha = 0.2 + (i / len(route_points)) * 0.3
                self.canvas.create_oval(
                    px - 2, py - 2, px + 2, py + 2,
                    fill='#00ff00',
                    outline=''
                )
        
        # Draw snake body (excluding head)
        if self.snake_body and len(self.snake_body) > 1:
            # Body is all positions except the last one (which is the head)
            body_positions = self.snake_body[:-1]
            for pos in body_positions:
                x1 = offset_x + pos[0] * scale
                y1 = offset_y + pos[1] * scale
                x2 = x1 + square
                y2 = y1 + square
                
                # Draw body segment
                self.canvas.create_rectangle(
                    x1 + 2, y1 + 2,
                    x2 - 2, y2 - 2,
                    fill=self.colors['snake_body'],
                    outline='#006600',
                    width=1
                )
        
        # Draw snake head
        if self.snake_body:
            head_x, head_y = self.snake_body[-1]  # Last position is head
        else:
            head_x, head_y = move.head_position
        
        x1 = offset_x + head_x * scale
        y1 = offset_y + head_y * scale
        x2 = x1 + square
        y2 = y1 + square
        
        # Draw snake head
        self.canvas.create_rectangle(
            x1 + 2, y1 + 2,
            x2 - 2, y2 - 2,
            fill=self.colors['snake_head'],
            outline='#006600',
            width=2
        )
        
        # Draw eyes on head
        eye_size = square / 8
        
        if move.direction == 'right':
            self.canvas.create_oval(x2 - eye_size*3, y1 + eye_size,
                                  x2 - eye_size, y1 + eye_size*3,
                                  fill='black')
            self.canvas.create_oval(x2 - eye_size*3, y2 - eye_size*3,
                                  x2 - eye_size, y2 - eye_size,
                                  fill='black')
        elif move.direction == 'left':
            self.canvas.create_oval(x1 + eye_size, y1 + eye_size,
                                  x1 + eye_size*3, y1 + eye_size*3,
                                  fill='black')
            self.canvas.create_oval(x1 + eye_size, y2 - eye_size*3,
                                  x1 + eye_size*3, y2 - eye_size,
                                  fill='black')
        elif move.direction == 'up':
            self.canvas.create_oval(x1 + eye_size, y1 + eye_size,
                                  x1 + eye_size*3, y1 + eye_size*3,
                                  fill='black')
            self.canvas.create_oval(x2 - eye_size*3, y1 + eye_size,
                                  x2 - eye_size, y1 + eye_size*3,
                                  fill='black')
        else:  # down
            self.canvas.create_oval(x1 + eye_size, y2 - eye_size*3,
                                  x1 + eye_size*3, y2 - eye_size,
                                  fill='black')
            self.canvas.create_oval(x2 - eye_size*3, y2 - eye_size*3,
                                  x2 - eye_size, y2 - eye_size,
                                  fill='black')
        
        # Draw food
        food_x, food_y = move.food_position
        x1 = offset_x + food_x * scale
        y1 = offset_y + food_y * scale
        x2 = x1 + square
        y2 = y1 + square
        
        self.canvas.create_oval(
            x1 + 5, y1 + 5,
            x2 - 5, y2 - 5,
            fill=self.colors['food'],
            outline='#aa0000',
            width=2
        )
        
        # Update score
        if hasattr(self, 'score_label'):
            self.score_label.config(text=f"Score: {move.score}")
    
    def _update_info_displays(self):
        """Update information panels"""
        if not self.current_run:
            return
        
        run = self.current_run
        
        agent_text = f"Agent ID: {run.agent_id}\n"
        agent_text += f"Generation: {run.generation}\n"
        agent_text += f"Final Score: {run.final_score}\n"
        agent_text += f"Final Fitness: {run.final_fitness:.2f}\n"
        agent_text += f"Total Steps: {run.total_steps}\n"
        
        # Handle death_reason
        death_reason_str = run.death_reason
        if hasattr(run.death_reason, 'value'):
            death_reason_str = run.death_reason.value
        elif hasattr(run.death_reason, 'name'):
            death_reason_str = run.death_reason.name
        agent_text += f"Death Reason: {death_reason_str}"
        
        self.agent_info.delete(1.0, tk.END)
        self.agent_info.insert(1.0, agent_text)
        
        # Death analysis text
        death_text = f"Death Position: {run.death_position}\n"
        death_text += f"Path Efficiency: {run.path_efficiency:.2f}\n"
        death_text += f"Exploration Rate: {run.exploration_rate:.2f}\n"
        death_text += f"Cycle Detected: {run.cycle_detected}"
        
        self.death_info.delete(1.0, tk.END)
        self.death_info.insert(1.0, death_text)
    
    def _update_progress(self):
        """Update progress bar"""
        if not self.current_run:
            return
        
        total_steps = len(self.current_run.moves)
        self.progress_var.set((self.current_move_index / max(1, total_steps)) * 100)
        self.progress_label.config(text=f"Step: {self.current_move_index}/{total_steps}")
    
    def set_speed(self, multiplier: float):
        """Set playback speed"""
        self.speed_multiplier = multiplier
        self.speed_label.config(text=f"Current: {multiplier:.1f}x")
        
        for btn, speed in [(self.speed_1x_btn, 1.0), (self.speed_3x_btn, 3.0), 
                           (self.speed_5x_btn, 5.0), (self.speed_7x_btn, 7.0)]:
            if abs(self.speed_multiplier - speed) < 0.1:
                btn.config(bg='#00aa00')
            else:
                btn.config(bg='#444444')
    
    def play(self):
        """Start playing"""
        self.playing = True
        self.paused = False
        self.play_btn.config(bg='#00aa00')
        self.pause_btn.config(bg='#aa5500')
        self._replay_loop()
    
    def pause(self):
        """Pause playback"""
        self.paused = not self.paused
        if self.paused:
            self.pause_btn.config(bg='#aa0000')
        else:
            self.pause_btn.config(bg='#aa5500')
            self._replay_loop()
    
    def stop(self):
        """Stop playback and reset"""
        self.playing = False
        self.paused = False
        self.play_btn.config(bg='#00aa00')
        self.pause_btn.config(bg='#aa5500')
        self.go_to_start()
    
    def step_forward(self):
        """Move forward one step"""
        if not self.current_run:
            return
        
        next_index = self.current_move_index + 1
        if next_index < len(self.current_run.moves):
            self._apply_replay_move(next_index)
    
    def step_back(self):
        """Move back one step"""
        if not self.current_run:
            return
        
        prev_index = self.current_move_index - 1
        if prev_index >= 0:
            self._apply_replay_move(prev_index)
    
    def go_to_start(self):
        """Go to the first step"""
        if self.current_run and self.current_run.moves:
            self._apply_replay_move(0)
    
    def go_to_end(self):
        """Go to the last step"""
        if self.current_run and self.current_run.moves:
            self._apply_replay_move(len(self.current_run.moves) - 1)
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.pause()
    
    def _replay_loop(self):
        """Main replay loop"""
        if not self.playing or self.paused or not self.window:
            return
        
        next_index = self.current_move_index + 1
        if next_index < len(self.current_run.moves):
            self._apply_replay_move(next_index)
            delay = int(self.game_speed / self.speed_multiplier)
            self.window.after(delay, self._replay_loop)
        else:
            self.playing = False
            self.paused = False
            self.play_btn.config(bg='#00aa00')
    
    def close(self):
        """Close the application"""
        if self.window:
            self.window.quit()
            self.window.destroy()
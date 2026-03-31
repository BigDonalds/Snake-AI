<h1>Snake AI Evolution System</h1>

<p>This project implements an evolutionary artificial intelligence system that trains a neural network to play the classic Snake game. Using a genetic algorithm with advanced features like speciation, novelty search, and behavioral analysis, the system evolves increasingly capable snake agents over many generations.</p>

<p>The system operates in two main modes:</p>

<ul>
    <li><strong>Training Mode</strong> – runs in the background, continuously evolving the population and logging performance data.</li>
    <li><strong>Visualizer Mode</strong> – a separate window that can be launched on-demand to replay the best, worst, or any run from the latest generation for detailed analysis.</li>
</ul>

<p>The AI's entire evolutionary history, including the genomes of the best agents and discovered strategies, is stored in a JSON-based "ancestral memory" file, allowing experiments to be paused and resumed. The visualizer provides a deep dive into agent behavior, reconstructing the snake's path and providing detailed metrics on its performance and cause of death.</p>

<hr>

<h2>Features</h2>
<ul>
    <li><strong>🧬 Genetic Algorithm</strong> with population management, speciation, and elitism.</li>
    <li><strong>🧠 Modular Neural Network</strong> with configurable layers, activations (ReLU, Leaky ReLU, etc.), and normalization.</li>
    <li><strong>📊 Comprehensive Statistics</strong> – tracks fitness, score, path efficiency, and exploration rates for every individual.</li>
    <li><strong>🎮 On-Demand Visualizer</strong> – replay any run from a generation to analyze the agent's decision-making and movement patterns.</li>
    <li><strong>💾 Ancestral Memory</strong> – saves the entire evolutionary lineage, including the Hall of Fame, discovered strategies, and behavioral trends.</li>
    <li><strong>🔍 Behavioral Analysis</strong> – detects and logs agent strategies like "wall hugging," "precision targeting," and "zigzag patterns."</li>
    <li><strong>🎛️ Dynamic Adaptation</strong> – automatically adjusts mutation rates and selection pressures when the population stagnates.</li>
</ul>

<hr>

<h2>Requirements</h2>
<ul>
    <li>Python 3.14.2</li>
    <li>Install dependencies using: <code>pip install -r requirements.txt</code></li>
</ul>

<hr>

<h2>Project Structure</h2>

<pre><code>.
├── main.py                   # Entry point for the training system
├── game.py                   # Core Snake game logic and environment
├── snake.py                  # Snake class for body management and movement
├── visualizer.py             # Tkinter-based replay interface
├── memory/                   # Directory for ancestral memory JSON files
│   └── ancestors_memory.py   # Handles saving/loading of evolutionary history
├── genetic_ai/               # Core AI and evolutionary logic
│   ├── genetic_algorithm.py  # Main GA implementation
│   └── neural_network.py     # Modular neural network and optimizer
├── analysis/                 # Data collection and statistics
│   ├── generation_stats.py   # Aggregated stats for a whole generation
│   ├── replay_manager.py     # Data provider for the visualizer
│   └── run_data.py           # Data model for a single agent's run
├── requirements.txt          # Python dependencies
└── README.md
</code></pre>

<hr>

<h2>How to Run</h2>

<p>Execute the main script from the project root directory:</p>
<pre><code>python main.py
</code></pre>

<p>This will start the training process in the background. The console will display real-time statistics, including the current generation, best score, and population metrics.</p>

<h3>Controls While Training</h3>
<ul>
    <li>Press the <code>v</code> key to open the visualizer window without stopping the training process.</li>
    <li>Press <code>Ctrl+C</code> to gracefully stop training, save all progress, and exit.</li>
</ul>

<hr>

<h2>How It Works</h2>

<p>The system follows this evolutionary cycle:</p>
<ol>
    <li><strong>Initialization</strong>: A population of 100 agents is created with varied neural network weight initialization strategies (e.g., He, Xavier, sparse).</li>
    <li><strong>Evaluation</strong>: Each agent plays a full game of Snake. During the game, the system records its actions, score, and movement path.</li>
    <li><strong>Fitness Calculation</strong>: A multi-objective fitness score is calculated based on survival time, score, wall avoidance, efficiency, and exploration. Agents that die quickly by hitting a wall are heavily penalized.</li>
    <li><strong>Behavioral Profiling</strong>: Each agent's behavior (e.g., path efficiency, turn frequency) is extracted to promote diversity and detect strategies.</li>
    <li><strong>Selection & Variation</strong>: Agents are selected for reproduction using tournament selection. Offspring are created via crossover and mutation of their parent networks.</li>
    <li><strong>Speciation & Elitism</strong>: The population is divided into species based on genome similarity to maintain diversity. The top-performing agents (elites) are carried over to the next generation unchanged.</li>
    <li><strong>Stagnation Handling</strong>: If the population stops improving, the mutation rate is increased, and "breakout" agents with randomly initialized genomes are added to escape local optima.</li>
</ol>

<p>After each generation, a <code>GenerationStats</code> object is created, summarizing the generation's performance. This data is used by the visualizer to offer replay options.</p>

<hr>

<h2>Visualizer & Replay System</h2>

<p>The visualizer (<code>visualizer.py</code>) is a key feature for analyzing agent behavior. It provides:</p>
<ul>
    <li>A selection screen to choose the run to replay (Best, Worst, etc.).</li>
    <li>Playback controls (play, pause, step forward/backward, speed control).</li>
    <li>A visual reconstruction of the snake's movement, including a faded trail of its path.</li>
    <li>Detailed information panels showing agent ID, generation, final score, and cause of death.</li>
    <li>Death analysis metrics like path efficiency, exploration rate, and cycle detection.</li>
</ul>

<p>When you press <code>v</code> during training, the system creates a new visualizer window that queries the latest <code>GenerationStats</code> to populate its selection screen. This allows you to observe the best-performing agents of the most recent generation while training continues in the background.</p>

<hr>

<h2>Output & Data</h2>

<p>All evolutionary data is saved to the <code>memory/</code> directory in a single JSON file (<code>snake_ancestry.json</code>). This file contains:</p>
<ul>
    <li><strong>Hall of Fame</strong>: A list of the top-scoring agents across all generations, including their genomes.</li>
    <li><strong>Discovered Strategies</strong>: A log of when the AI first achieved certain milestones (e.g., "Score 10", "Wall Hugging").</li>
    <li><strong>Evolutionary History</strong>: The best and average fitness and score for each generation.</li>
    <li><strong>Behavioral Trends</strong>: Records of zigzag tendencies and self-collision rates over time.</li>
</ul>

<p>This file allows the training to be resumed exactly where it left off. The system also periodically prints generation summaries to the console, providing a real-time view of progress.</p>

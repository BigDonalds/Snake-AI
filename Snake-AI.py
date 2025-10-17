from tkinter import *
import random
import collections

GAME_WIDTH = 700
GAME_HEIGHT = 600
SPEED = 35
SQUARE_SIZE = 50
BODY_PARTS = 2
SNAKE_COLOR = "#0040FF"
FOOD_COLOR = "#EEFA08"
BACKGROUND_COLOR = "#000000"

class Snake:
    def __init__(self):
        self.body_size = BODY_PARTS
        self.coordinates = []
        self.squares = []

        for i in range(0, BODY_PARTS):
            self.coordinates.append((0, 0))

        for x, y in self.coordinates:
            square = canvas.create_rectangle(x, y, x + SQUARE_SIZE, y + SQUARE_SIZE, fill=SNAKE_COLOR, tag="snake")
            self.squares.append(square)

class Food:
    def __init__(self, snake):
        while True:
            x = random.randint(0, (GAME_WIDTH // SQUARE_SIZE) - 1) * SQUARE_SIZE
            y = random.randint(0, (GAME_HEIGHT // SQUARE_SIZE) - 1) * SQUARE_SIZE

            if (x, y) not in snake.coordinates:
                break
        
        self.coordinates = (x, y)
        canvas.create_oval(x, y, x + SQUARE_SIZE, y + SQUARE_SIZE, fill=FOOD_COLOR, tag="food")

def bfs(start, goal, snake):
    queue = collections.deque([start])
    came_from = {start: None}
    visited = set()

    while queue:
        current = queue.popleft()

        if current == goal:
            break

        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x, y = current
            dx, dy = direction
            neighbor = (x + dx * SQUARE_SIZE, y + dy * SQUARE_SIZE)

            if (neighbor not in came_from 
                and 0 <= neighbor[0] < GAME_WIDTH 
                and 0 <= neighbor[1] < GAME_HEIGHT 
                and neighbor not in snake.coordinates
                and neighbor not in visited):
                queue.append(neighbor)
                came_from[neighbor] = current
                visited.add(neighbor)

    if goal not in came_from:
        return None

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def flood_fill_space(snake_body):
    head = snake_body[0]
    queue = collections.deque([head])
    visited = set(snake_body)
    space_count = 0

    while queue:
        current = queue.popleft()
        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + direction[0] * SQUARE_SIZE, current[1] + direction[1] * SQUARE_SIZE)
            if (0 <= neighbor[0] < GAME_WIDTH 
                and 0 <= neighbor[1] < GAME_HEIGHT 
                and neighbor not in visited):
                queue.append(neighbor)
                visited.add(neighbor)
                space_count += 1
    return space_count

def is_safe_to_move(snake, next_head):
    temp_snake = snake.coordinates.copy()
    temp_snake.insert(0, next_head)
    temp_snake.pop()
    return flood_fill_space(temp_snake) >= len(snake.coordinates)

def find_largest_safe_move(snake):
    x, y = snake.coordinates[0]
    max_space = -1
    best_move = None

    # Prioritize vertical moves (up, down) before horizontal (left, right)
    for direction in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        dx, dy = direction
        neighbor = (x + dx * SQUARE_SIZE, y + dy * SQUARE_SIZE)
        
        if (0 <= neighbor[0] < GAME_WIDTH 
            and 0 <= neighbor[1] < GAME_HEIGHT 
            and neighbor not in snake.coordinates):
            temp_snake = snake.coordinates.copy()
            temp_snake.insert(0, neighbor)
            temp_snake.pop()
            space_count = flood_fill_space(temp_snake)
            if space_count > max_space:
                max_space = space_count
                best_move = neighbor

    return best_move

def can_escape(snake, next_head, limit):
    temp_snake = snake.coordinates.copy()
    temp_snake.insert(0, next_head)
    temp_snake.pop()
    return flood_fill_space(temp_snake) >= len(snake.coordinates) + limit

def find_limited_vertical_move(snake, vertical_limit):
    x, y = snake.coordinates[0]
    max_space = -1
    best_move = None
    vertical_moves = 0

    # Only move vertically within the limit
    for direction in [(0, -1), (0, 1)]:  # Up and down
        dx, dy = direction
        neighbor = (x + dx * SQUARE_SIZE, y + dy * SQUARE_SIZE)
        
        if (0 <= neighbor[0] < GAME_WIDTH 
            and 0 <= neighbor[1] < GAME_HEIGHT 
            and neighbor not in snake.coordinates):
            temp_snake = snake.coordinates.copy()
            temp_snake.insert(0, neighbor)
            temp_snake.pop()
            space_count = flood_fill_space(temp_snake)
            if space_count > max_space and vertical_moves < vertical_limit:
                max_space = space_count
                best_move = neighbor
                vertical_moves += 1

    return best_move

def next_turn(snake, food):
    global direction
    head = snake.coordinates[0]
    tail = snake.coordinates[-1]
    
    # Check the number of free spaces left
    free_spaces = flood_fill_space(snake.coordinates)

    # Set a vertical move limit based on remaining space
    vertical_limit = 3  # You can adjust this limit as needed
    
    if free_spaces < 10:  # Endgame phase detected
        # Move vertically but with a limit
        next_move = find_limited_vertical_move(snake, vertical_limit)
        
        # After vertical moves, make sure the snake can still escape
        if not next_move or not can_escape(snake, next_move, vertical_limit):
            # If no safe vertical move, switch to horizontal or largest safe move
            next_move = find_largest_safe_move(snake)
            
    else:
        # Regular movement logic
        path_to_food = bfs(head, food.coordinates, snake)
        path_to_tail = bfs(head, tail, snake)
        
        if path_to_food and len(path_to_food) > 1 and is_safe_to_move(snake, path_to_food[1]):
            next_move = path_to_food[1]
        else:
            if path_to_tail and len(path_to_tail) > 1 and is_safe_to_move(snake, path_to_tail[1]):
                next_move = path_to_tail[1]
            else:
                next_move = find_largest_safe_move(snake)

    if not next_move:
        next_move = head

    # Update the snake's position
    snake.coordinates.insert(0, next_move)
    x, y = next_move
    square = canvas.create_rectangle(x, y, x + SQUARE_SIZE, y + SQUARE_SIZE, fill=SNAKE_COLOR)
    snake.squares.insert(0, square)

    if next_move == food.coordinates:
        global score
        score += 1
        label.config(text="Score:{}".format(score))
        canvas.delete("food")
        food = Food(snake)
    else:
        del snake.coordinates[-1]
        canvas.delete(snake.squares[-1])
        del snake.squares[-1]

    if check_collisions(snake):
        game_over()
    else:
        window.after(SPEED, next_turn, snake, food)

def change_direction(new_direction):
    global direction
    if new_direction == 'left' and direction != 'right':
        direction = new_direction
    elif new_direction == 'right' and direction != 'left':
        direction = new_direction
    elif new_direction == 'up' and direction != 'down':
        direction = new_direction
    elif new_direction == 'down' and direction != 'up':
        direction = new_direction

def check_collisions(snake):
    x, y = snake.coordinates[0]
    if x < 0 or x >= GAME_WIDTH or y < 0 or y >= GAME_HEIGHT:
        return True
    if snake.coordinates[0] in snake.coordinates[1:]:
        return True
    return False

def restart_game():
    global snake, food, direction, score
    canvas.delete(ALL)
    snake = Snake()
    food = Food(snake)
    direction = 'down'
    score = 0
    label.config(text="Score:{}".format(score))
    next_turn(snake, food)

def game_over():
    canvas.delete(ALL)
    canvas.create_text(canvas.winfo_width() / 2, canvas.winfo_height() / 2,
                       font=('consolas', 70), text="Collision", fill="red", tag="gameover")
    window.after(2000, restart_game)

window = Tk()
window.title("Snake game")
window.resizable(False, False)

score = 0
direction = 'down'

label = Label(window, text="Score:{}".format(score), font=('consolas', 40))
label.pack()

canvas = Canvas(window, bg=BACKGROUND_COLOR, height=GAME_HEIGHT, width=GAME_WIDTH)
canvas.pack()

window.update()

window_width = window.winfo_width()
window_height = window.winfo_height()
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))

window.geometry(f"{window_width}x{window_height}+{x}+{y}")

window.bind('w', lambda event: change_direction('up'))
window.bind('s', lambda event: change_direction('down'))
window.bind('a', lambda event: change_direction('left'))
window.bind('d', lambda event: change_direction('right'))

snake = Snake()
food = Food(snake)

next_turn(snake, food)

window.mainloop()

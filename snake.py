import copy

class Snake:
    def __init__(self, config, initial_positions=None, centered=True):
        self.config = config
        self.direction = 'down'
        
        if initial_positions:
            self.coordinates = copy.deepcopy(initial_positions)
        else:
            # Calculate center of the grid
            square_size = config['SQUARE_SIZE']
            game_width = config['GAME_WIDTH']
            game_height = config['GAME_HEIGHT']
            
            # Center coordinates (aligned to grid)
            center_x = (game_width // (2 * square_size)) * square_size
            center_y = (game_height // (2 * square_size)) * square_size
            
            if centered:
                # Initialize snake with head at center, body segments behind it
                self.coordinates = []
                for i in range(config['BODY_PARTS']):
                    # Body parts extend upward from center (so head is at front)
                    self.coordinates.append([center_x, center_y - (i * square_size)])
                # Ensure all body parts are within bounds
                # If any part goes out of bounds, adjust
                for i, coord in enumerate(self.coordinates):
                    if coord[1] < 0:
                        self.coordinates[i][1] = 0
            else:
                self.coordinates = []
                for i in range(config['BODY_PARTS']):
                    self.coordinates.append([0, 0])
    
    def move(self, direction):
        """Move the snake in the given direction, return new head position"""
        x, y = self.coordinates[0]
        square_size = self.config['SQUARE_SIZE']
        
        if direction == "up":
            y -= square_size
        elif direction == "down":
            y += square_size
        elif direction == "left":
            x -= square_size
        elif direction == "right":
            x += square_size
        
        self.coordinates.insert(0, [x, y])
        return (x, y)
    
    def remove_tail(self):
        """Remove the tail of the snake"""
        if len(self.coordinates) > 1:
            self.coordinates.pop()
    
    def check_self_collision(self, head_x=None, head_y=None):
        """Check if snake collides with itself"""
        if head_x is None or head_y is None:
            head_x, head_y = self.coordinates[0]
        
        for body_part in self.coordinates[1:]:
            if head_x == body_part[0] and head_y == body_part[1]:
                return True
        return False
    
    def get_head_position(self):
        """Get current head position"""
        return self.coordinates[0]
    
    def get_body_positions(self):
        """Get all snake body positions"""
        return self.coordinates.copy()
    
    def get_safety_rating(self, direction, game_width, game_height):
        """Rate how safe a direction is (0 = death, 1 = safe)"""
        head_x, head_y = self.coordinates[0]
        square_size = self.config['SQUARE_SIZE']
        
        if direction == "up":
            new_x, new_y = head_x, head_y - square_size
        elif direction == "down":
            new_x, new_y = head_x, head_y + square_size
        elif direction == "left":
            new_x, new_y = head_x - square_size, head_y
        elif direction == "right":
            new_x, new_y = head_x + square_size, head_y
        else:
            return 0
        
        # Check walls
        if new_x < 0 or new_x >= game_width or new_y < 0 or new_y >= game_height:
            return 0
        
        # Check self collision
        if [new_x, new_y] in self.coordinates[1:]:
            return 0
        
        return 1

    def clone(self):
        """Create a deep copy of the snake"""
        return Snake(self.config, self.coordinates, centered=False)
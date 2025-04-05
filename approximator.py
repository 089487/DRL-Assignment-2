import copy
import random
import math
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------

def rot90(pattern, board_size):
    return [(j,board_size-i-1) for i,j in pattern]
def reflection(pattern, board_size):
    return [(i,board_size-j-1) for i,j in pattern]

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        print(patterns)
        self.board_size = board_size
        patterns = [tuple(pattern) for pattern in patterns]
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = {pattern : defaultdict(float) for pattern in patterns}
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = {pattern : self.generate_symmetries(pattern) for pattern in patterns}

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        symmetries = []
        last_board = copy.deepcopy(pattern)
        for i in range(4):
            last_board = rot90(last_board,self.board_size)
            for j in range(2):
                last_board = reflection(last_board,self.board_size)
                symmetries.append(last_board)
                print(last_board)
        return tuple(symmetries)

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple([self.tile_to_index(board[x][y]) for x,y in coords])

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        Sum = 0
        for pattern in self.patterns:
            for sym_pattern in self.symmetry_patterns[pattern]:
                feature = self.get_feature(board, sym_pattern)
                Sum += self.weights[pattern][feature]
        return Sum
    def update(self, board, delta):
        # TODO: Update weights based on the TD error.
        for pattern in self.patterns:
            for sym_pattern in self.symmetry_patterns[pattern]:
                feature = self.get_feature(board, sym_pattern)
                self.weights[pattern][feature]+=delta
def get_approximator():
    patterns = [
    # straight
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 1), (1, 1), (2, 1), (3, 1)],
    # Square patterns (2x3)
    [(0, 0), (1, 0), (2, 0),
    (0, 1), (1, 1), (2, 1)],
    [(0, 1), (1, 1), (2, 1),
    (0, 2), (1, 2), (2, 2)]
    ]
    approximator = NTupleApproximator(board_size=4,patterns=patterns)
    with open('approximator.pkl', 'rb') as f:
        approximator = pickle.load(f)
    
    print("Approximator loaded successfully!")
    return approximator
import functools
import math
import random
import sys

import numpy as np
import pygame
import tensorflow as tf

from neatnn.neat import NeatNN
from neatnn.dont_stop_game import NNAI, DontStop

from neatnn.utils import chunks

NeatNN.prepare(4, 4)

game = DontStop()
print("Preparing players")
generation_size = 1000
players_nn = [NeatNN.mutate(NeatNN()) for _ in range(generation_size)]

generation = 0
while True:
    print(f"Generation {generation}")
    generation = generation + 1

    print("Wrapping player objects")
    players = []
    for network_def in players_nn:
        players.append(NNAI(network_def))

    chunk_size = 50
    for chunk_index, chunk in enumerate(chunks(players, chunk_size)):
        print(f"Running chunk {1 + chunk_index}/{len(players)/chunk_size:.0f}")
        game.run(chunk)


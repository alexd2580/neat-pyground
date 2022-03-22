"""Main interleaved game/evolution routine."""
import logging

# from neatnn.falling_obstacles import NNAI, Game
from neatnn.circular_movement import NNAI, CircularMovement
# from neatnn.dont_stop_game import NNAI, DontStop
from neatnn.neat import NeatNN, Population
from neatnn.utils import chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NeatNN.setup(4, 4)
game = CircularMovement()
# game = DontStop()
population = Population(500)
chunk_size = 50

while True:
    players = [NNAI(nn) for nn in population.everybody()]
    for index, chunk in enumerate(chunks(players, chunk_size)):
        logger.info(f"Running chunk {1 + index}/{len(players)/chunk_size:.0f}")
        game.reset()
        game.run(chunk)

    population.next_generation()

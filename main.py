"""Main interleaved game/evolution routine."""
import logging

# from neatnn.circular_movement import NNAI, CircularMovement
from neatnn.dont_stop_game import NNAI, DontStop
from neatnn.neat import NeatNN, Population
from neatnn.utils import chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NeatNN.setup(2, 4)
# game = CircularMovement()
game = DontStop()
population = Population(500)
chunk_size = 50

while True:
    players = [NNAI(nn) for nn in population.everybody()]
    # for chunk in chunks(players, chunk_size):
    #     # logger.info(f"Running chunk {1 + chunk_index}/{len(players)/chunk_size:.0f}")
    #     game.reset()
    #     game.run(chunk)

    game.reset()
    game.run(players)

    population.next_generation()

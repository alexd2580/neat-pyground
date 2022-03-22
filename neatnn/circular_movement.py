"""Don't stop game."""
import math
from random import randint

import neatnn.pygame as base
import pygame
from neatnn.nn import RecurrentEvaluator


class Player(base.Player):
    """Base class for the "Don't Stop" game."""

    def __init__(self):
        """Initialize a player at `(100, 100)`."""
        self._x = 720  # 1440 / 2
        self._y = 450  # 900 / 2
        self._color = (randint(0, 255), randint(0, 255), randint(0, 255), 255)

        self._ticks_alive = 0
        self._dead = False

    @property
    def is_dead(self):  # noqa: D102
        return self._dead

    @property
    def x(self):  # noqa: D102
        return self._x

    @property
    def y(self):  # noqa: D102
        return self._y

    @property
    def score(self):  # noqa: D102
        return self._ticks_alive

    @property
    def color(self):  # noqa: D102
        return self._color

    def update(self, game):  # noqa: D102
        if self._dead:
            return

        up, left, down, right = self._last_inputs = self.get_inputs(game)

        dx = (1 if right else 0) - (1 if left else 0)
        dy = (1 if down else 0) - (1 if up else 0)
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist > 0:
            speed = 5
            self._x = self._x + dx * speed / dist
            self._y = self._y + dy * speed / dist

        dist_2 = (self._x - game.circle_x) ** 2 + (self._y - game.circle_y) ** 2
        if dist_2 > CircularMovement.RADIUS_2:
            self._dead = True
            return

        self._ticks_alive = self._ticks_alive + 1

    def render(self, surface):  # noqa: D102
        pygame.draw.circle(
            surface,
            self._color,
            (int(self._x), int(self._y)),
            20 if self._dead else 10,
            2 if self._dead else 0,
        )


class NNAI(Player):
    """Pure python NN AI-controlled player."""

    def __init__(self, nn_def):  # noqa: D107
        Player.__init__(self)
        self.nn_def = nn_def
        self._nn = RecurrentEvaluator(nn_def)

    def get_inputs(self, game):  # noqa: D102
        result = self._nn(
            [game.circle_x / 1440, self.x / 1440, game.circle_x / 1440, (self.x - game.circle_y) / 900, self.y / 900]
        )
        return [a > 0.5 for a in result]

    def update(self, game):
        """Copy the score to the NN after each tick."""
        super().update(game)
        self.nn_def.set_score(self.score)


class CircularMovement(base.Game):
    """A trivial game where you die when you stop moving."""

    RADIUS = 25
    RADIUS_2 = 25 ** 2

    def __init__(self):
        """Initialize the game and stuff."""
        super().__init__()
        self.reset()

    def reset(self):
        """Reset the circle position."""
        self.circle_x = 720
        self.circle_y = 450
        self._tick = 0

    def update(self, players):
        """Check if players are in circle and move circle."""
        self.circle_x = 720 + (10 + self._tick / 1000) * 500 * math.sin(self._tick / 480)
        self.circle_y = 450 + (10 + self._tick / 1000) * 300 * math.sin(self._tick / 120)

        self._tick = self._tick + 1

    def render(self, players):
        """Render the player's score in the top-right corner."""
        color = (50, 50, 50, 255)
        pos = (int(self.circle_x), int(self.circle_y))
        pygame.draw.circle(self.screen, color, pos, self.RADIUS)

        lines = [
            (f"{player.score:.2f} {player._last_inputs}", player.color)
            for player in players
            if not player.is_dead
        ]
        self.render_text(lines, (1000, 10))

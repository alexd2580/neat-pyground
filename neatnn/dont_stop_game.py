"""Don't stop game."""
import math
from random import randint

import neatnn.pygame as base
import pygame
from neatnn.nn import RecurrentEvaluator
from neatnn.utils import randbool


class Player(base.Player):
    """Base class for the "Don't Stop" game."""

    _rect_size = 100
    _max_ticks_in_rect = 50

    def __init__(self):
        """Initialize a player at `(100, 100)`."""
        self._x = self._lx = randint(50, 1390)
        self._y = self._ly = randint(50, 850)
        self._color = (randint(0, 255), randint(0, 255), randint(0, 255), 255)

        self._distance_travelled = 0
        self._dead = False
        self._ticks_in_rect = 0

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
        return self._distance_travelled

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
            dx = dx * speed / dist
            dy = dy * speed / dist

            self._distance_travelled = self._distance_travelled + speed
            self._x = self._x + dx
            self._y = self._y + dy

            if not base.Game.is_on_screen(self._x, self._y):
                self._dead = True

        # Die if > 100.000
        if self._distance_travelled > 100000:
            self._dead = True

        # Die if not moving around.
        if not base.Game.is_in_rect(
            self._x,
            self._y,
            self._lx - self._rect_size / 2,
            self._ly - self._rect_size / 2,
            self._rect_size,
            self._rect_size,
        ):
            self._lx = self._x
            self._ly = self._y
            self._ticks_in_rect = 0
        else:
            self._ticks_in_rect = self._ticks_in_rect + 1
            if self._ticks_in_rect > self._max_ticks_in_rect:
                self._dead = True

    def render(self, surface):  # noqa: D102
        pygame.draw.circle(
            surface,
            self._color,
            (int(self._x), int(self._y)),
            20 if self._dead else 10,
            2 if self._dead else 0,
        )


class Human(Player):
    """A human-controlled player."""

    def get_inputs(self, game):  # noqa: D102
        pressed = pygame.key.get_pressed()
        return (pressed[pygame.K_w], pressed[pygame.K_a], pressed[pygame.K_s], pressed[pygame.K_d])


class NNAI(Player):
    """Pure python NN AI-controlled player."""

    def __init__(self, nn_def):  # noqa: D107
        Player.__init__(self)
        self.nn_def = nn_def
        self._nn = RecurrentEvaluator(nn_def)

    def get_inputs(self, game):  # noqa: D102
        result = self._nn([(720 - self.x) / 100, (450 - self.y) / 100])
        return [a > 0.5 for a in result]

    def update(self, game):
        """Copy the score to the NN after each tick."""
        super().update(game)
        self.nn_def.set_score(self.score)


class DontStop(base.Game):
    """A trivial game where you die when you stop moving."""

    def render(self, players):
        """Render the player's score in the top-right corner."""
        lines = [
            (f"{player.score:.2f} {player._last_inputs}", player.color)
            for player in players
            if not player.is_dead
        ]
        self.render_text(lines, (1000, 10))

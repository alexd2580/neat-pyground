"""Evade falling obstacles."""
import collections
import math
from random import randint, random

import neatnn.pygame as base
import pygame
from neatnn.nn import RecurrentEvaluator

Vector2 = collections.namedtuple("Vector", "x y")


def dot(a, b):
    """Dot product."""
    return a.x * b.x + a.y * b.y


def length_squared(a):
    """Squared length."""
    return dot(a, a)


def distance_squared(a, b):
    """Squared length."""
    return length_squared(sub(a, b))


def add(a, b):
    """Addition."""
    return Vector2(a.x + b.x, a.y + b.y)


def mul(a, f):
    """Multiplication."""
    return Vector2(a.x * f, a.y * f)


def sub(a, b):
    """Subtraction."""
    return Vector2(a.x - b.x, a.y - b.y)


Obstacle = collections.namedtuple("Obstacle", "pos radius dir")


class Player(base.Player):
    """Base class for the "Don't Stop" game."""

    def __init__(self):
        """Initialize a player at `(100, 100)`."""
        self._pos = Vector2(250, 250)
        self._color = (randint(0, 255), randint(0, 255), randint(0, 255), 255)

        self._ticks_alive = 0
        self._dead = False

    @property
    def is_dead(self):  # noqa: D102
        return self._dead

    @property
    def pos(self):  # noqa: D102
        return self._pos

    @property
    def score(self):  # noqa: D102
        return self._ticks_alive

    @property
    def color(self):  # noqa: D102
        return self._color

    def view_distance_squared(self, obstacle, dir):
        d = dot(sub(obstacle.pos, self._pos), dir)
        on_disc = add(self._pos, mul(dir, d))
        on_disc_r_squared = distance_squared(on_disc, obstacle.pos)
        r_sqr = obstacle.radius ** 2
        if on_disc_r_squared > r_sqr:
            return math.inf
        offset_collision = (r_sqr - on_disc_r_squared) ** 0.5
        return d - offset_collision

    def update(self, game):  # noqa: D102
        if self._dead:
            return

        up, left, down, right = self._last_inputs = self.get_inputs(game)

        dx = (1 if right else 0) - (1 if left else 0)
        dy = (1 if down else 0) - (1 if up else 0)
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist > 0:
            speed = 5
            self._pos = add(self._pos, mul(Vector2(dx, dy), speed / dist))

        if not Game.is_on_screen(self._pos.x, self._pos.y):
            self._dead = True
            return

        for obstacle in game.obstacles:
            if distance_squared(obstacle.pos, self._pos) < obstacle.radius ** 2:
                self._dead = True
                return

        self._ticks_alive = self._ticks_alive + 1

    def render(self, surface):  # noqa: D102
        pygame.draw.circle(
            surface,
            self._color,
            (int(self._pos.x), int(self._pos.y)),
            20 if self._dead else 10,
            2 if self._dead else 0,
        )


sqrt_2 = 2 ** 0.5


class NNAI(Player):
    """Pure python NN AI-controlled player."""

    def __init__(self, nn_def):  # noqa: D107
        Player.__init__(self)
        self.nn_def = nn_def
        self._nn = RecurrentEvaluator(nn_def)

    _view_dirs = [
        Vector2(1, 0),
        Vector2(-1, 0),
        Vector2(0, 1),
        Vector2(0, -1),
        Vector2(sqrt_2, sqrt_2),
        Vector2(sqrt_2, -sqrt_2),
        Vector2(-sqrt_2, sqrt_2),
        Vector2(-sqrt_2, -sqrt_2),
    ]

    def get_inputs(self, game):  # noqa: D102
        inputs = [
            min([1.0, *[self.view_distance_squared(o, dir) / 100000 for o in game.obstacles]])
            for dir in NNAI._view_dirs
        ]
        result = self._nn(inputs)
        return [a > 0.5 for a in result]

    def update(self, game):
        """Copy the score to the NN after each tick."""
        super().update(game)
        self.nn_def.set_score(self.score)


class Game(base.Game):
    """Evade falling obstacles."""

    def __init__(self):
        """Initialize the game and stuff."""
        super().__init__()
        self.reset()

    def reset(self):
        """Reset the circle position."""
        self.obstacles = []
        self._tick = 0

    def update(self, players):
        """Check if players are in circle and move circle."""
        # if self._tick % 75 == 0:
        #     pos = Vector2(100 + self._tick % 1240, -200)
        #     radius = 50 + self._tick % 200
        #     dir = Vector2(-5.5 + self._tick % 9, 2 + self._tick % 5)
        if self._tick % 25 == 0:
            pos = Vector2(((self._tick / 25) * 100) % 505, -100)
            radius = 50
            dir = Vector2(0, 4)
            self.obstacles.append(Obstacle(pos, radius, dir))

        self.obstacles = [
            Obstacle(add(obstacle.pos, obstacle.dir), obstacle.radius, obstacle.dir)
            for obstacle in self.obstacles
            if obstacle.pos.y < 550
        ]

        self._tick = self._tick + 1

    def render(self, players):
        """Render the player's score in the top-right corner."""
        color = (50, 50, 50, 255)

        for obstacle in self.obstacles:
            pos = (int(obstacle.pos.x), int(obstacle.pos.y))
            pygame.draw.circle(self.screen, color, pos, obstacle.radius)

        # lines = [
        #     (f"{player.score:.2f} {player._last_inputs}", player.color)
        #     for player in players
        #     if not player.is_dead
        # ]
        # self.render_text(lines, (1000, 10))

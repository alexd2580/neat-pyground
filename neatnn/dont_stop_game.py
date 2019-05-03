"""Don't stop game."""
import math
from random import randint

import pygame

import neatnn.pygame as base
from neatnn.utils import randbool


class Player(base.Player):
    """Base class for the "Don't Stop" game."""

    def __init__(self):
        """Initialize a player at `(100, 100)`."""
        self._x = 100  # 1440 / 2
        self._y = 100  # 900 / 2
        self._color = (randint(0, 255), randint(0, 255), randint(0, 255), 255)

        self._distance_travelled = 0
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
        return self._distance_travelled

    @property
    def color(self):  # noqa: D102
        return self._color

    def update(self):  # noqa: D102
        if self._dead:
            return

        up, left, down, right = self._last_inputs = self.get_inputs()

        dx = (1 if right else 0) - (1 if left else 0)
        dy = (1 if down else 0) - (1 if up else 0)
        dist = math.sqrt(dx ** 2 + dy ** 2)

        speed = 5
        dx = dx * speed / dist
        dy = dy * speed / dist

        self._distance_travelled = self._distance_travelled + speed
        self._x = self._x + dx
        self._y = self._y + dy

        if not base.Game.is_on_screen(self._x, self._y):
            self._dead = True

        # Die if located in an dead fixpoint.
        if (up == down) and (left == right):
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

    def get_inputs(self):  # noqa: D102
        pressed = pygame.key.get_pressed()
        return (pressed[pygame.K_w], pressed[pygame.K_a], pressed[pygame.K_s], pressed[pygame.K_d])


class RandomAI(Player):
    """A random AI-controlled player."""

    def get_inputs(self):  # noqa: D102
        return (randbool(), randbool(), randbool(), randbool())


class NNAITF(Player):
    """Tensorflow NN AI-controlled player."""

    def __init__(self, nn_def, tf_session, input_var, output_var):  # noqa: D107
        """Initialize a tensorflow-NN backed AI.

        `nn_def` (NeatNN) Source of the Neural network.
        `tf_session` (tf.Session) Current tensorflow session.
        `input_var` (List[tf.Variable?]) TF Input variables.
        `output_var` (List[tf.Expression?]) NN output expressions.
        """
        Player.__init__(self)
        self._tf_session = tf_session
        self.nn_def = nn_def
        self._input_var = input_var
        self._output_var = output_var

    def get_inputs(self):  # noqa: D102
        feed_dict = {
            self._input_var[0]: self.x / 1440,
            self._input_var[1]: self.y / 900,
            self._input_var[2]: (1440 - self.x) / 1440,
            self._input_var[3]: (900 - self.y) / 900,
        }
        result = self._tf_session.run(self._output_var, feed_dict=feed_dict)
        return [a > 0.5 for a in result]


class NNAI(Player):
    """Pure python NN AI-controlled player."""

    def __init__(self, nn_def):  # noqa: D107
        Player.__init__(self)
        self.nn_def = nn_def
        self._nn = nn_def.to_python_function()

    def get_inputs(self):  # noqa: D102
        result = self._nn(
            [self.x / 1440, self.y / 900, (1440 - self.x) / 1440, (900 - self.y) / 900]
        )
        return [a > 0.5 for a in result]


class DontStop(base.Game):
    """A trivial game where you die when you stop moving."""

    def render(self, players):
        """Render the player's score in the top-right corner."""
        lines = [(f"{player.score:.2f}", player.color) for player in players]
        self.render_text(lines, (1000, 10))

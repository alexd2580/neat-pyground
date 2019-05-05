"""Base classes for implementing trivial `pygame`s."""
import functools
import math
import sys

import pygame


class Player:
    """Player base class."""

    @property
    def is_dead(self):
        """Get the current player state - alive or dead."""
        raise NotImplementedError("You need to implement the `is_dead` property.")

    def update(self, game):
        """Is called by the `Game.run` method every tick."""
        pass

    def get_inputs(self, game):
        """Retrieve user input.

        Simplifies integration of AI players.
        """
        raise NotImplementedError("You need to implement the `get_inputs` method.")

    def render(self, surface):
        """Render the player onto `surface`.

        Is called by `Game.run` every tick if `Game._graphics` is `True`.
        """
        pass


class Game:
    """Base game class.

    Wraps basic game routines and provides an overridable interface.
    """

    WIDTH = 1440
    HEIGHT = 900

    _graphics = True
    _fps = 60

    @staticmethod
    def is_in_rect(x, y, rx, ry, rw, rh):
        """Check if a point `(x, y)` is in the given rect."""
        return x >= rx and x <= rx + rw and y >= ry and y <= ry + rh

    @staticmethod
    def is_on_screen(x, y):
        """Check if the point is visible on the screen."""
        return Game.is_in_rect(x, y, 0, 0, Game.WIDTH, Game.HEIGHT)

    def __init__(self):
        """Initialize a window."""
        pygame.init()
        self.font = pygame.font.SysFont("dejavusansmono", 12)

        self.window_size = self.WIDTH, self.HEIGHT
        self.full_rect = 0, 0, self.WIDTH, self.HEIGHT
        self.screen = pygame.display.set_mode(self.window_size)
        self.black = pygame.Color(0, 0, 0, 255)

    def reset(self):
        """Override this to easily reset the game state from the main."""
        pass

    def render_text(self, lines, pos):
        """Display text split by lines at position `pos`."""
        lines = [(text, color, self.font.size(text)) for text, color in lines]
        text_width, text_height = functools.reduce(
            lambda total, line: (max(total[0], line[2][0]), total[1] + line[2][1]), lines, (0, 0)
        )

        for text, color, (_, height) in lines:
            text_surface = self.font.render(text, True, color)
            self.screen.blit(text_surface, dest=pos)
            pos = pos[0], pos[1] + height

    def handle_event(self, event):
        """Handle additional events."""
        pass

    def update(self, players):
        """Override this function to update global state in each game tick."""
        pass

    def render(self, players):
        """Override this function to render global stuff in each game tick."""
        pass

    def run(self, players):
        """Run the game."""
        self._running = True
        clock = pygame.time.Clock()

        while self._running and not all(player.is_dead for player in players):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.unicode == "q":
                        sys.exit()
                    elif event.unicode == "n":
                        self._running = False
                    elif event.unicode == "v":
                        self._graphics = not self._graphics
                    elif event.unicode == "-":
                        self._fps = max(1, self._fps - 10)
                    elif event.unicode == "+":
                        self._fps = min(self._fps + 10, 300)
                self.handle_event(event)

            for player in players:
                player.update(self)

            self.update(players)

            if self._graphics:
                self.screen.fill(self.black, self.full_rect)
                self.render(players)
                for player in players:
                    player.render(self.screen)
                pygame.display.flip()

                if self._fps is not None and self._fps < 300:
                    clock.tick(self._fps)

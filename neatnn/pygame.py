"""Base classes for implementing trivial `pygame`s."""
import functools

import pygame


class Player:
    """Player base class."""

    @property
    def is_dead(self):
        """Get the current player state - alive or dead."""
        raise NotImplementedError("You need to implement the `is_dead` property.")

    def update(self):
        """Is called by the `Game.run` method every tick."""
        pass

    def get_inputs(self):
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
    _fps = 30

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

    def handle_events(self):
        """Override this function to handle events in each game tick."""
        pass

    def run(self, players):
        """Run the game."""
        self._running = True
        clock = pygame.time.Clock()

        while self._running and not all(player.is_dead for player in players):
            self.handle_events()
            for player in players:
                player.update()

            if self._graphics:
                self.screen.fill(self.black, self.full_rect)
                lines = [
                    (f"{player.score:.2f} {player._last_inputs}", player.color)
                    for player in players
                ]
                self.render_text(lines, (1000, 10))
                for player in players:
                    player.render(self.screen)
                pygame.display.flip()

                if self._fps is not None:
                    clock.tick(self._fps)

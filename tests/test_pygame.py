"""Tests for the `pygame` module."""

import pytest
from hypothesis import given
from hypothesis.strategies import booleans, integers

from neatnn.pygame import Game, Player


class TestPlayer:
    """Tests for the `Player` class."""

    def test_is_dead(self):
        """`is_dead` raises an exception."""
        player = Player()
        with pytest.raises(NotImplementedError):
            player.is_dead

    def test_update(self):
        """`update` does nothing."""
        player = Player()
        player.update()

    def test_get_inputs(self):
        """`get_inputs` raises an exception."""
        player = Player()
        with pytest.raises(NotImplementedError):
            player.get_inputs()

    def test_render(self):
        """`render` does nothing."""
        player = Player()
        player.render(None)


class TestGame:
    """Tests for the `Game` class."""

    class Game(Game):  # noqa: D106
        def __init__(self):  # noqa: D107
            pass

    class DeadPlayer(Player):  # noqa: D106
        @property
        def is_dead(self):  # noqa: D102
            return True

        def get_inputs(self):  # noqa: D102
            # `get_inputs` is not called on dead players.
            assert False

    class DyingPlayer(Player):  # noqa: D106
        def __init__(self):  # noqa: D107
            self.cycles = 0

        @property
        def is_dead(self):  # noqa: D102
            return self.cycles > 10

        def update(self):  # noqa: D102
            self.cycles = self.cycles + 1

        def get_inputs(self):  # noqa: D102
            pass

    @given(x=integers(min_value=-5, max_value=5), y=integers(min_value=-5, max_value=5))
    def test_is_in_rect_positive(self, x, y):
        """`in_rect` returns true for any point inside a rect."""
        assert Game.is_in_rect(x, y, -5, -5, 10, 10)

    @given(
        x=integers(min_value=-5, max_value=5),
        y=integers(min_value=-5, max_value=5),
        offset_x=booleans(),
        offset_y=booleans(),
    )
    def test_is_in_rect_negative(self, x, y, offset_x, offset_y):
        """`in_rect` returns true for any point inside a rect."""
        x = x + (10 + 1) * (1 if offset_x else -1)
        y = y + (10 + 1) * (1 if offset_y else -1)
        assert not Game.is_in_rect(x, y, -5, -5, 10, 10)

    @given(
        x=integers(min_value=0, max_value=Game.WIDTH),
        y=integers(min_value=0, max_value=Game.HEIGHT),
    )
    def test_is_on_screen_positive(self, x, y):
        """`in_rect` returns true for any point inside a rect."""
        assert Game.is_on_screen(x, y)

    @given(
        x=integers(min_value=0, max_value=Game.WIDTH),
        y=integers(min_value=0, max_value=Game.HEIGHT),
        offset_x=booleans(),
        offset_y=booleans(),
    )
    def test_is_on_screen_negative(self, x, y, offset_x, offset_y):
        """`in_rect` returns true for any point inside a rect."""
        x = x + (Game.WIDTH + 1) * (1 if offset_x else -1)
        y = y + (Game.HEIGHT + 1) * (1 if offset_y else -1)
        assert not Game.is_in_rect(x, y, -5, -5, 10, 10)

    @pytest.mark.skip("TODO: Check if pygame has headless mode.")
    def test___init__(self):
        """`__init__` initializes a `Game` object."""
        Game()

    @pytest.mark.skip("TODO: Check if pygame has headless mode.")
    def test_render_test(self):
        """`render_text` displays text."""
        g = Game()
        g.render_text([], (0, 0))

    def test_handle_events(self):
        """`handle_events` does nothing."""
        g = TestGame.Game()
        g.handle_event(None)

    def test_update(self):
        """`update` does nothing."""
        g = TestGame.Game()
        g.update([])

    def test_render(self):
        """`render` does nothing."""
        g = TestGame.Game()
        g.render([])

    def test_run(self):
        """`run` iterates while the game is `_running` and some players are alive."""
        g = TestGame.Game()
        g._graphics = False
        g.run([TestGame.DeadPlayer()])
        p = TestGame.DyingPlayer()
        g.run([p])
        assert p.cycles == 11
        g.run([TestGame.DyingPlayer(), TestGame.DyingPlayer(), TestGame.DeadPlayer()])

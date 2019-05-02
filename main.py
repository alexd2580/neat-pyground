import functools
import math
import random
import sys

import numpy as np
import pygame
import tensorflow as tf

from neatnn.nn import NeatNN


class Player:
    _distance_travelled = 0
    _dead = False

    def __init__(self):
        self._x = 100  # 1440 / 2
        self._y = 100  # 900 / 2
        self._color = (random_int(256), random_int(256), random_int(256), 255)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def score(self):
        return self._distance_travelled

    @property
    def color(self):
        return self._color

    def update(self):
        if self._dead:
            return

        up, left, down, right = self._last_inputs = self.get_inputs()
        new_x = self._x + (5 if right else 0) - (5 if left else 0)
        new_y = self._y + (5 if down else 0) - (5 if up else 0)
        self._distance_travelled = (
            self._distance_travelled + ((new_x - self._x) ** 2 + (new_y - self._y) ** 2) ** 0.5
        )
        self._x = new_x
        self._y = new_y

        if not Game.is_on_screen(self._x, self._y):
            self._dead = True

        # Die if located in an dead fixpoint.
        if (up == down) and (left == right):
            self._dead = True

    def get_inputs(self):
        raise Exception("method must be overridden")

    def render(self, surface):
        pygame.draw.circle(
            surface,
            self._color,
            (int(self._x), int(self._y)),
            20 if self._dead else 10,
            2 if self._dead else 0,
        )


class Human(Player):
    def get_inputs(self):
        pressed = pygame.key.get_pressed()
        return (pressed[pygame.K_w], pressed[pygame.K_a], pressed[pygame.K_s], pressed[pygame.K_d])


class RandomAI(Player):
    def get_inputs(self):
        return (random_bool(), random_bool(), random_bool(), random_bool())


class NNAITF(Player):
    def __init__(self, nn_def, tf_session, input_var, output_var):
        Player.__init__(self)
        self._tf_session = tf_session
        self.nn_def = nn_def
        self._input_var = input_var
        self._output_var = output_var

    def get_inputs(self):
        feed_dict = {
            self._input_var[0]: self.x / 1440,
            self._input_var[1]: self.y / 900,
            self._input_var[2]: (1440 - self.x) / 1440,
            self._input_var[3]: (900 - self.y) / 900,
        }
        result = self._tf_session.run(self._output_var, feed_dict=feed_dict)
        return [a > 0.5 for a in result]


class NNAI(Player):
    def __init__(self, nn_def):
        Player.__init__(self)
        self.nn_def = nn_def
        self.nn = nn_def.to_python_function()

    def get_inputs(self):
        result = self.nn(
            [self.x / 1440, self.y / 900, (1440 - self.x) / 1440, (900 - self.y) / 900]
        )
        return [a > 0.5 for a in result]


class Game:
    WIDTH = 1440
    HEIGHT = 900

    graphics = True

    @staticmethod
    def is_in_rect(x, y, rx, ry, rw, rh):
        return x > rx and x < rw and y > ry and y < rh

    @staticmethod
    def is_on_screen(x, y):
        return Game.is_in_rect(x, y, 0, 0, Game.WIDTH, Game.HEIGHT)

    def __init__(self):
        pygame.init()
        self.font = pygame.font.SysFont("dejavusansmono", 12)

        self.window_size = self.WIDTH, self.HEIGHT
        self.full_rect = 0, 0, self.WIDTH, self.HEIGHT
        self.screen = pygame.display.set_mode(self.window_size)
        self.black = pygame.Color(0, 0, 0, 255)

    def render_text(self, lines, pos):
        lines = [(text, color, self.font.size(text)) for text, color in lines]
        text_width, text_height = functools.reduce(
            lambda total, line: (max(total[0], line[2][0]), total[1] + line[2][1]), lines, (0, 0)
        )

        for text, color, (_, height) in lines:
            text_surface = self.font.render(text, True, color)
            self.screen.blit(text_surface, dest=pos)
            pos = pos[0], pos[1] + height

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.unicode == "q":
                    sys.exit()
                elif event.unicode == "s":
                    self.running = False
                elif event.unicode == "v":
                    self.graphics = not self.graphics

    def run(self, players):
        self.running = True
        clock = pygame.time.Clock()
        while self.running and not all(player._dead for player in players):
            self.handle_events()
            for player in players:
                player.update()

            if self.graphics:
                self.screen.fill(self.black, self.full_rect)
                lines = [
                    (f"{player.score:.2f} {player._last_inputs}", player.color)
                    for player in players
                ]
                self.render_text(lines, (1000, 10))
                for player in players:
                    player.render(self.screen)
                pygame.display.flip()
                # clock.tick(60)


def chunks(data, chunk_size):
    accum = []
    for d in data:
        accum.append(d)
        if len(accum) == chunk_size:
            yield accum
            accum = []

    if accum:
        yield accum


if __name__ == "__main__":
    NN.prepare(4, 4)

    game = Game()
    print("Preparing players")
    generation_size = 1000
    players_nn = [NN.mutate(NN()) for _ in range(generation_size)]

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

        print("Sorting by score")
        players.sort(reverse=True, key=lambda a: a.score)
        players_nn = [player.nn_def for player in players]

        print("Splitting into species")
        speciess = []
        for nn in players_nn:
            categorized = False
            for species in speciess:
                if NN.is_same_species(nn, species[0]):
                    species.append(nn)
                    categorized = True
                    break

            if not categorized:
                speciess.append([nn])
        print(f"Split population into {len(speciess)} species")
        species_sizes = [len(s) for s in speciess]
        print(f"Average species size: {np.mean(species_sizes)} {np.median(species_sizes)}")

        print(f"Selecting the fittest {NN.thanos * 100:.2f}% from every species")
        speciess = [species[: math.floor(NN.thanos * len(species))] for species in speciess]

        players_nn = [a for species in speciess for a in species]
        num_players = len(players_nn)

        print("Breeding")
        to_breed = int(0.75 * generation_size - num_players)
        for a in range(to_breed):
            i1, i2 = random_int(num_players), random_int(num_players)
            a_i, b_i = min(i1, i2), max(i1, i2)
            players_nn.append(NN.breed(players_nn[a_i], players_nn[b_i]))

        print("Mutating")
        for a in range(int(0.25 * generation_size)):
            i = random_int(num_players)
            players_nn.append(NN.mutate(players_nn[i]))

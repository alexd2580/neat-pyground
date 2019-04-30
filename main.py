import functools
import math
import random
import sys

import numpy as np
import pygame
import tensorflow as tf


def random_int(max_value=10):
    return min(int(random.random() * max_value), max_value - 1)


def random_bool():
    return random.random() > 0.5


def generate_example():
    a = random_int()
    b = random_int()
    result = a + b
    # const_in = tf.constant([[
    #     1 if i == a or i == b + 10 else 0
    #     for i in range(20)
    # ]])
    # const_out = tf.constant([[
    #     1 if i == result else 0
    #     for i in range(20)
    # ]])
    const_in = [1 if i == a or i == b + 10 else 0 for i in range(20)]
    const_out = [1 if i == result else 0 for i in range(20)]
    return a, b, result, const_in, const_out


def generate_feed_dict(input_var, output_var, num_examples=None):
    data = [generate_example() for _ in range(num_examples)]
    batch_in = [in_tensor for _, _, _, in_tensor, _ in data]
    batch_out = [out_tensor for _, _, _, _, out_tensor in data]
    return {input_var: batch_in, output_var: batch_out}


def attach_layer(current_input, current_size, next_size):
    weights = tf.Variable(tf.random_normal([current_size, next_size], stddev=0.03))
    bias = tf.Variable(tf.random_normal([next_size]))
    return tf.nn.sigmoid(tf.add(bias, tf.matmul(current_input, weights)))


def create_nn(input_size=None, hidden_sizes=None, output_size=None):
    """Create a new neural network.

    The first dimension of the in/out layers is the number of examples.
    """
    if input_size is None:
        raise Exception("Can't have `None` as input size")
    if output_size is None:
        raise Exception("Can't have `None` as output size")
    if hidden_sizes is None:
        hidden_sizes = []
    if any(value is None for value in hidden_sizes):
        raise Exception("Can't have `None` as hidden size")

    hidden_sizes.append(output_size)

    # Terminal layers.
    input = tf.placeholder(tf.float32, [None, input_size])
    expected_output = tf.placeholder(tf.float32, [None, output_size])

    current_output = input
    current_size = input_size

    for hidden_size in hidden_sizes:
        current_output = attach_layer(current_output, current_size, hidden_size)
        current_size = hidden_size

    return input, expected_output, current_output


def gradient_descent():
    input, expected_output, output = create_nn(input_size=10, hidden_sizes=[10], output_size=2)

    # Define correctness.
    numerical_output = tf.math.argmax(output, 1)
    numerical_expected_output = tf.math.argmax(expected_output, 1)
    correctness_bool_list = tf.equal(numerical_output, numerical_expected_output)
    correctness_list = tf.cast(correctness_bool_list, tf.float32)
    correctness = tf.reduce_mean(correctness_list)

    # Define optimizer.
    cost = 1 - correctness
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        # Train.
        ITERATIONS = 100000
        BATCH_SIZE = 100
        num_batches = int(ITERATIONS / BATCH_SIZE)
        for epoch in range(10):
            average_correctness = 0
            for batch_index in range(num_batches):
                feed_dict = generate_feed_dict(input, expected_output, num_examples=BATCH_SIZE)
                _, correctness = sess.run([optimiser, correctness], feed_dict=feed_dict)
                average_correctness += correctness / num_batches

            print(f"Epoch {epoch}\t: correctness = {average_correctness:.3f}")
        print("Training complete.")

        for i in range(100):
            a, b, sum_ab, in_vals, _ = generate_example()
            feed_dict = {input: [in_vals]}
            nn_out = sess.run(numerical_output, feed_dict=feed_dict)
            print(f"{a}\t+ {b}\t= {sum_ab}\t?= {nn_out}")


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

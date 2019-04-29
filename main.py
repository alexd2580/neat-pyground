import functools
import random
import sys

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


class NN:
    # connections: [(Index, Start, End, Active, Weight)]
    _connections = []

    _num_in = None
    _num_out = None

    _next_node_index = None
    _next_gene_index = None

    @staticmethod
    def prepare(num_in, num_out):
        NN._num_in = num_in
        NN._num_out = num_out

        NN._next_node_index = num_in + num_out
        NN._next_gene_index = 0

    @staticmethod
    def breed(a, b):
        max_dominant_gene = max([index for index, _, _, _, _ in a._connections])

        def find_gene(index, pool):
            return next((x for x in pool if x[0] == index), None)

        new_connections = []
        for index in range(max_dominant_gene + 1):
            gene_a = find_gene(index, a._connections)
            gene_b = find_gene(index, b._connections)
            if not gene_a and not gene_b:
                continue

            if not gene_a or not gene_b:
                new_connections.append(gene_a or gene_b)
            else:
                new_connections.append(gene_a if random_bool() else gene_b)

        return NN(connections=new_connections)

    @staticmethod
    def add_link(connections):
        start_nodes = list(set([*[start for _, start, _, _, _ in connections], *range(NN._num_in)]))
        end_nodes = list(
            set(
                [
                    *[end for _, _, end, _, _ in connections],
                    *[index + NN._num_in for index in range(NN._num_out)],
                ]
            )
        )

        gene_index = NN._next_gene_index
        NN._next_gene_index = NN._next_gene_index + 1
        weight = random.random() * 4 - 2

        # Repeat until not cyclic.
        def is_parent_of(a, b):
            if a == b:
                return True
            return any([is_parent_of(end, b) for _, start, end, _, _ in connections if start == a])

        start_node, end_node = 0, 0
        while is_parent_of(end_node, start_node):
            start_node = start_nodes[random_int(len(start_nodes))]
            end_node = end_nodes[random_int(len(end_nodes))]

        connections.append((gene_index, start_node, end_node, True, weight))

    @staticmethod
    def split_link(connections):
        if len(connections) == 0:
            return NN.add_link(connections)

        random_connection_index = random_int(len(connections))
        index, start, end, _, weight = connections[random_connection_index]
        connections[random_connection_index] = (index, start, end, False, weight)

        gene_index = NN._next_gene_index
        NN._next_gene_index = NN._next_gene_index + 2

        node_index = NN._next_node_index
        NN._next_node_index = NN._next_node_index + 1

        connections.extend(
            [
                (gene_index, start, node_index, True, 1.0),
                (gene_index + 1, node_index, end, True, weight),
            ]
        )
        return connections

    @staticmethod
    def shift_link(connections):
        if len(connections) == 0:
            return NN.add_link(connections)
        random_connection_index = random_int(len(connections))
        index, start, end, active, weight = connections[random_connection_index]
        connections[random_connection_index] = (index, start, end, active, weight * random.random() * 4 - 2)
        return connections

    @staticmethod
    def randomize(connections):
        if len(connections) == 0:
            return NN.add_link(connections)
        random_connection_index = random_int(len(connections))
        index, start, end, active, weight = connections[random_connection_index]
        connections[random_connection_index] = (index, start, end, active, random.random() * 4 - 2)
        return connections

    @staticmethod
    def mutate(a):
        possible_mutations = [NN.add_link, NN.split_link, NN.shift_link, NN.randomize]
        mutation_f = possible_mutations[random_int(len(possible_mutations))]
        connections = [*a._connections]
        mutation_f(connections)
        return NN(connections)

    def __init__(self, connections=None):
        self._connections = connections or []

    def to_tensorflow_network(self):
        nodes = {}

        # Create input nodes.
        for index in range(NN._num_in):
            nodes[index] = tf.Variable(0.0, name=f"input{index}")

        # Create temporary placeholders.
        for index in range(NN._num_out):
            nodes[NN._num_in + index] = tf.Variable(0.0, name=f"out{index}")

        # Sort connections by their target.
        connections_by_end = {}
        for connection in self._connections:
            prev = connections_by_end.get(connection[2]) or []
            prev.append(connection)
            connections_by_end[connection[2]] = prev

        while connections_by_end:
            creatable_end_nodes = {
                index: connections
                for index, connections in connections_by_end.items()
                if all(start in nodes for _, start, _, _, _ in connections)
            }

            for index, connections in creatable_end_nodes.items():
                # `connections` is all connections required for a certain end node.
                # Filter active connections.
                connections = [c for c in connections if c[3]]

                end_node = tf.constant(0.0)
                for _, start, _, _, weight in connections:
                    end_node = tf.nn.sigmoid(tf.add(end_node, tf.multiply(nodes[start], weight)))

                nodes[index] = end_node
                del connections_by_end[index]

        inputs = [nodes[index] for index in range(NN._num_in)]
        outputs = [nodes[index + NN._num_in] for index in range(NN._num_out)]
        return inputs, outputs


class Player:
    _distance_travelled = 0
    _dead = False

    def __init__(self):
        self._x = 1440 / 2
        self._y = 900 / 2
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


class NNAI(Player):
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
                clock.tick(60)


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
    num_players = 100
    players_nn = [NN.mutate(NN()) for _ in range(num_players)]

    generation = 0
    while True:
        print(f"Generation {generation}")
        generation = generation + 1

        print("Converting networks")
        tf_networks = [(nn, nn.to_tensorflow_network()) for nn in players_nn]

        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)

            print("Wrapping player objects")
            players = []
            for network_def, (inputs, outputs) in tf_networks:
                players.append(NNAI(network_def, sess, inputs, outputs))

            chunk_size = 25
            for chunk_index, chunk in enumerate(chunks(players, 25)):
                print(f"Running chunk {chunk_index}/{len(players)/25:.0f}")
                game.run(chunk)

            print("Sorting by score")
            players.sort(reverse=True, key=lambda a: a.score)
            players_nn = [player.nn_def for player in players]
            print("Selecting the fittest")
            thanos = 50
            players_nn = players_nn[0:thanos]

            print("Breeding")
            for a in range(int(num_players / 3)):
                i1, i2 = random_int(thanos), random_int(thanos)
                a_i, b_i = min(i1, i2), max(i1, i2)
                players_nn.append(NN.breed(players_nn[a_i], players_nn[b_i]))

            print("Mutating")
            for a in range(int(num_players / 2)):
                i = random_int(thanos)
                players_nn.append(NN.mutate(players_nn[i]))

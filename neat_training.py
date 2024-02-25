# -*- coding: utf-8 -*-
"""

"""
import argparse

from mayhem import *

MODE = "training"

import neat
import cv2
from matplotlib import pyplot as plt

# -------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('-width', '--width', help='', type=int, action="store", default=500)
parser.add_argument('-height', '--height', help='', type=int, action="store", default=500)

result = parser.parse_args()
args = dict(result._get_kwargs())

print("Args", args)

# -------------------------------------------------------------------------------------------------

init_pygame()

game_window = GameWindow(args["width"], args["height"])

# -------------------------------------------------------------------------------------------------

class NeatTraining():

    def __init__(self, runs_per_net, max_gen, multi, input_type="ray"):

        self.runs_per_net = runs_per_net
        self.max_gen = max_gen
        self.multi = multi
        self.input_type = input_type

    def render_loaded_genome(self, g):
        config = neat.Config( neat.DefaultGenome, neat.DefaultReproduction,
                              neat.DefaultSpeciesSet, neat.DefaultStagnation,
                              os.path.join(os.getcwd(), "config") )

        #net = neat.nn.RecurrentNetwork.create(g, config)
        net = neat.nn.FeedForwardNetwork.create(g, config)

        neat_env = MayhemEnv(game_window, level=1, max_fps=0, debug_print=1, play_sound=False, motion="gravity", sensor=self.input_type, record_play="", play_recorded="")
        observation, info = neat_env.reset()

        done = False
        while not done:
            action = net.activate(observation)
            #action = np.argmax(net.activate(observation))

            act = np.array([action[0]>0.5, action[1]>0.5, action[2]>0.5]).astype(np.int8)

            observation, reward, done, truncated, info = neat_env.step(act, max_frame=20000)
            neat_env.render(max_fps=0, collision_check=False)

    def load_net(self, net_name=None):

        if not net_name:
            file_list = [ x for x in os.listdir(os.getcwd()) if os.path.isfile(os.path.join(os.getcwd(), x)) and x.startswith("gen") ]

            for fname in file_list:
                with open(fname, 'rb') as f:
                    g = pickle.load(f)

                print('Loaded genome:')
                print(g)

                self.render_loaded_genome(g)
                time.sleep(1)

        with open(net_name, 'rb') as f:
            g = pickle.load(f)

        print('Loaded genome:')
        print(g)

        self.render_loaded_genome(g)

    def train_it(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config')

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        pop = neat.Population(config)
        stats = neat.StatisticsReporter()

        pop.add_reporter(stats)
        pop.add_reporter(neat.StdOutReporter(True))
        pop.add_reporter(CustomNeatReporter(self.multi))

        if self.multi:
            pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), self.eval_genome)
            winner = pop.run(pe.evaluate, self.max_gen)
        else:
            if 0:
                pe = neat.ParallelEvaluator(1, self.eval_genome)
                winner = pop.run(pe.evaluate, self.max_gen)
            else:
                winner = pop.run(self.eval_genomes, self.max_gen)

        # Save the winner.
        with open('winner', 'wb') as f:
            pickle.dump(winner, f)

        print(winner)

    def eval_genome(self, genome, config):
        #for i, g in enumerate(genome):
        #    print(i, g)

        #net = neat.nn.RecurrentNetwork.create(genome, config)
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        fitnesses = []

        for runs in range(self.runs_per_net):

            neat_env = MayhemEnv(game_window, level=1, max_fps=0, debug_print=1, play_sound=False, motion="gravity", sensor=self.input_type, record_play="", play_recorded="")
            
            observation, info = neat_env.reset()

            fitness = 0.0
            done = False

            #cv2.namedWindow("main", cv2.WINDOW_NORMAL)
            #cv2.moveWindow("main", 100, 100)

            while not done:

                #action = np.argmax(net.activate(observation))
                action = net.activate(observation) # [-1.0, -0.17934807670239852, 1.0, -0.3551236740213184]
                #print(action)


                act = np.array([action[0]>0.5, action[1]>0.5, action[2]>0.5]).astype(np.int8)
                #print(act)

                observation, reward, done, truncated, info = neat_env.step(act, max_frame=2000)
                
                if not self.multi:
                    if self.input_type == "ray":
                        neat_env.render(max_fps=0, collision_check=False)

                    elif self.input_type == "pic":
                        sr = neat_env.render(collision_check=True)

                        fp = pygame.surfarray.array3d(sr)

                        fp = cv2.resize(fp, (32, 32))

                        fp = cv2.cvtColor(fp, cv2.COLOR_RGB2BGR)  # pygame => cv2 color format
                        fp = cv2.cvtColor(fp, cv2.COLOR_BGR2GRAY) # grey it

                        #cv2.imshow('main', cv2.transpose(fp))
                        #cv2.waitKey(1)  

                        fp = np.reshape(fp, 32*32) # flat, 1d
                        neat_env.frame_pic = fp

                #print(observation)

                fitness += reward
                #print("fitness=", fitness)

                # dump network
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_d:
                            now = dt.datetime.now()
                            net_name = f"gen_{genome.fitness}_{now.hour}h{now.minute}m{now.second}s"
                            with open(net_name, 'wb') as f:
                                pickle.dump(genome, f)
                            print("Dumped ", net_name)


            # here we are done
            fitnesses.append(fitness)

        # done for all run per net
        mean_fit = np.mean(fitnesses)
        print(mean_fit)
        return mean_fit

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = self.eval_genome(genome, config)

# -------------------------------------------------------------------------------------------------

class CustomNeatReporter(neat.reporting.BaseReporter):

    def __init__(self, multi):
        self.multi = multi

        self.generation = None
        self.data_to_plot1 = []
        self.data_to_plot2 = []

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):

        if not self.multi:
            fitnesses = [c.fitness for c in population.values()]

            self.data_to_plot2.append(np.mean(fitnesses))
            self.data_to_plot1.append(best_genome.fitness)

            plt.plot(self.data_to_plot1)
            plt.plot(self.data_to_plot2)        
            plt.pause(0.0001)

        if best_genome.fitness > 1000:
            now = dt.datetime.now()
            net_name = f"gen{self.generation}_{best_genome.fitness}_{now.hour}h{now.minute}m{now.second}s"

            with open(net_name, 'wb') as f:
                pickle.dump(best_genome, f)

            print(f"=> Dumped genome with fitness={best_genome.fitness} : {net_name}")

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

NEAT_LOAD_WINNER  = 0   #
NEAT_MAX_GEN      = 100 # stop if this number is reach (if not before per other criteria)
NEAT_RUNS_PER_NET = 1   # useful if init position is random
NEAT_MULTI        = 0   # multiprocess, if true no display

if NEAT_MULTI:
    pygame.display.iconify()

neat_training = NeatTraining(NEAT_RUNS_PER_NET, NEAT_MAX_GEN, NEAT_MULTI, input_type="ray")

if NEAT_LOAD_WINNER:
    #neat_training.load_net(net_name="gen2_1068.048876452548_22h31m52s")
    neat_training.load_net(net_name=None)
else:
    neat_training.train_it()

# Neat on pics input
if 0:
    env = MayhemEnv(game_window, vsync=0, render_game=False, nb_player=args["nb_player"], mode=args["run_mode"], motion=args["motion"],
                    sensor=args["sensor"], record_play=args["record_play"], play_recorded=args["play_recorded"])

    if CV2_FOUND:
        neat_training = NeatTraining(NEAT_RUNS_PER_NET, NEAT_MAX_GEN, NEAT_MULTI, input_type="pic")
        neat_training.train_it()
    else:
        print("OpenCV not found")

    if 0:   
        neat_env = MayhemEnv(game_window, vsync=False, render_game=False, nb_player=1, mode="training", motion="gravity", sensor="ray", record_play="", play_recorded="")
        observation, info = neat_env.reset()

        fitness = 0.0
        done = False

        cv2.namedWindow("main", cv2.WINDOW_NORMAL)
        cv2.moveWindow("main", 100, 100)

        while not done:
            action = -1
            observation, reward, done, truncated, info = neat_env.step(action, max_frame=4000)
            sr = neat_env.render(collision_check=False)
            
            if neat_env.game.debug_on_screen:

                # https://stackoverflow.com/questions/47614396/how-do-you-convert-3d-array-in-pygame-to-an-vaid-input-in-opencv-python
                f = pygame.surfarray.array3d(sr)

                f = cv2.resize(f, (128, 128))    

                f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)  # pygame => cv2 color format
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) # grey it

                #f = np.reshape(f, (128, 128))
                cv2.imshow('main', cv2.transpose(f))
                cv2.waitKey(1)  

                f = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB) # cv2 => pygame color format

                sr = pygame.surfarray.make_surface(f)

                neat_env.game.window.blit(sr, (neat_env.ship_1.view_left + neat_env.ship_1.view_width, neat_env.ship_1.view_top))
                pygame.display.flip()

            fitness += reward
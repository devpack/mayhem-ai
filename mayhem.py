# -*- coding: utf-8 -*-
"""
Because this game never dies and deserved to meet Python and AI.

The original game by Espen Skoglund (http://hol.abime.net/3853) was born in the early 90s on the Commodore Amiga. That was the great time of MC68000 Assembly.

Around 2000 we made a PC version (https://github.com/devpack/mayhem) of the game in C++.

It was then ported to Raspberry Pi by Martin O'Hanlon (https://github.com/martinohanlon/mayhem-pi), even new gfx levels were added.

----

It was early time this game had its own Python version. Pygame (https://www.pygame.org/docs) SDL wrapper has been used as backend.

The ultimate goal porting it to Python is to create a friendly AI environment (like Gym (https://gym.openai.com/envs/#atari)) which could easily be used with deep learning frameworks. AI players in Mayhem are coming !

Anthony Prieur
anthony.prieur@gmail.com
"""

"""
Usage example:

python mayhem.py --width=1500 --height=900 --nb_player=2 --sensor=ray -rm=game
python mayhem.py --width=1500 --height=900 --nb_player=1 --sensor=ray -rm=training

python3 mayhem.py --sensor=ray --motion=gravity
python3 mayhem.py --sensor=ray --motion=thrust
python3 mayhem.py --motion=thrust
python3 mayhem.py -r=played1.dat --motion=gravity
python3 mayhem.py -pr=played1.dat --motion=gravity
"""

import os, sys, argparse, random, math, time, multiprocessing
from random import randint
import numpy as np
import datetime as dt

import pygame
from pygame.locals import *
from pygame import gfxdraw

try:
    import cPickle as pickle
except ImportError:
    import pickle

import gym
from gym import spaces

from stable_baselines3 import A2C, PPO

# -------------------------------------------------------------------------------------------------
# General

global game_window

DEBUG_SCREEN = 1 # print debug info on the screen
DEBUG_TEXT_XPOS = 0

MAX_FPS = 60

# -------------------------------------------------------------------------------------------------
# The map is splitted into 24 zones, 4x * 6y
# 0 1 2 3
# 4 5 6 7
# ....
# 20 21 22 23

MAP_WIDTH  = 792
MAP_HEIGHT = 1200

ZONE_X_NUM  = 4
ZONE_Y_NUM  = 6
ZONE_X_SIZE = MAP_WIDTH  / ZONE_X_NUM
ZONE_Y_SIZE = MAP_HEIGHT / ZONE_Y_NUM

WHITE    = (255, 255, 255)
RED      = (255, 0, 0)
LVIOLET  = (128, 0, 128)

# -------------------------------------------------------------------------------------------------
# Player views

SHIP1_X = 473        # ie left
SHIP1_Y = 303        # ie top

SHIP2_X = 520        # ie left
SHIP2_Y = 955        # ie top

SHIP3_X = 75         # ie left
SHIP3_Y = 1015       # ie top

SHIP4_X = 451        # ie left
SHIP4_Y = 501        # ie top

USE_MINI_MASK = True # mask the size of the ship (instead of the player view size)

# -------------------------------------------------------------------------------------------------
# Sensor

RAY_AMGLE_STEP = 45
RAY_BOX_SIZE   = 400
RAY_MAX_LEN    = ((RAY_BOX_SIZE/2) * math.sqrt(2)) # for now we are at the center of the ray mask box

# -------------------------------------------------------------------------------------------------
# SHIP dynamics

SLOW_DOWN_COEF = 2.0 # somehow the C++ version is slower with the same physics coef ?

SHIP_MASS = 0.9
SHIP_THRUST_MAX = 0.32 / SLOW_DOWN_COEF
SHIP_ANGLESTEP = 5
SHIP_ANGLE_LAND = 30
SHIP_MAX_LIVES = 100
SHIP_SPRITE_SIZE = 32

iG       = 0.07 / SLOW_DOWN_COEF
#iG = 0

iXfrott  = 0.984
iYfrott  = 0.99
iCoeffax = 0.6
iCoeffay = 0.6
iCoeffvx = 0.6
iCoeffvy = 0.6
iCoeffimpact = 0.02
MAX_SHOOT = 20

# -------------------------------------------------------------------------------------------------
# Levels / controls

CURRENT_LEVEL = 1

PLATFORMS_1 = [ ( 464, 513, 333 ),
                ( 60, 127, 1045 ),
                ( 428, 497, 531 ),
                ( 504, 568, 985 ),
                ( 178, 241, 875 ),
                ( 8, 37, 187 ),
                ( 302, 351, 271 ),
                ( 434, 521, 835 ),
                ( 499, 586, 1165 ),
                ( 68, 145, 1181 ) ]

SHIP_1_INPUT = "HUMAN" # "NETWORK", "AI"
SHIP_1_KEYS = {"left":pygame.K_LEFT, "right":pygame.K_RIGHT, "thrust":pygame.K_KP_PERIOD, "shoot":pygame.K_KP_ENTER, "shield":pygame.K_KP0}
SHIP_1_JOY  = 1 # 0 means no joystick, =!0 means joystck number SHIP_1_JOY - 1

SHIP_2_INPUT = "HUMAN" # "NETWORK", "AI"
SHIP_2_KEYS = {"left":pygame.K_w, "right":pygame.K_x, "thrust":pygame.K_v, "shoot":pygame.K_g, "shield":pygame.K_c}
SHIP_2_JOY  = 2 # 0 means no joystick, =!0 means joystck number SHIP_1_JOY - 1

SHIP_3_INPUT = "HUMAN" # "NETWORK", "AI"
SHIP_3_KEYS = {"left":pygame.K_w, "right":pygame.K_x, "thrust":pygame.K_v, "shoot":pygame.K_g, "shield":pygame.K_c}
SHIP_3_JOY  = 0 # 0 means no joystick, =!0 means joystck number SHIP_1_JOY - 1

SHIP_4_INPUT = "HUMAN" # "NETWORK", "AI"
SHIP_4_KEYS = {"left":pygame.K_w, "right":pygame.K_x, "thrust":pygame.K_v, "shoot":pygame.K_g, "shield":pygame.K_c}
SHIP_4_JOY  = 0 # 0 means no joystick, =!0 means joystck number SHIP_1_JOY - 1

# -------------------------------------------------------------------------------------------------
# Assets

MAP_1 = os.path.join("assets", "level1", "Mayhem_Level1_Map_256c.bmp")

SOUND_THURST  = os.path.join("assets", "default", "sfx_loop_thrust.wav")
SOUND_EXPLOD  = os.path.join("assets", "default", "sfx_boom.wav")
SOUND_BOUNCE  = os.path.join("assets", "default", "sfx_rebound.wav")
SOUND_SHOOT   = os.path.join("assets", "default", "sfx_shoot.wav")
SOUND_SHIELD  = os.path.join("assets", "default", "sfx_loop_shield.wav")

SHIP_1_PIC        = os.path.join("assets", "default", "ship1_256c.bmp")
SHIP_1_PIC_THRUST = os.path.join("assets", "default", "ship1_thrust_256c.bmp")
SHIP_1_PIC_SHIELD = os.path.join("assets", "default", "ship1_shield_256c.bmp")

SHIP_2_PIC        = os.path.join("assets", "default", "ship2_256c.bmp")
SHIP_2_PIC_THRUST = os.path.join("assets", "default", "ship2_thrust_256c.bmp")
SHIP_2_PIC_SHIELD = os.path.join("assets", "default", "ship2_shield_256c.bmp")

SHIP_3_PIC        = os.path.join("assets", "default", "ship3_256c.bmp")
SHIP_3_PIC_THRUST = os.path.join("assets", "default", "ship3_thrust_256c.bmp")
SHIP_3_PIC_SHIELD = os.path.join("assets", "default", "ship3_shield_256c.bmp")

SHIP_4_PIC        = os.path.join("assets", "default", "ship4_256c.bmp")
SHIP_4_PIC_THRUST = os.path.join("assets", "default", "ship4_thrust_256c.bmp")
SHIP_4_PIC_SHIELD = os.path.join("assets", "default", "ship4_shield_256c.bmp")

# -------------------------------------------------------------------------------------------------
# Training

START_POSITIONS = [(430, 730), (473, 195), (647, 227), (645, 600), (647, 950), (510, 1070), (298, 1037), \
                   (273, 777), (275, 506), (70, 513), (89, 208), (434, 452), (289, 153)]

# NN input
NORMALIZE_INPUT = 1

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

class Shot():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.xposprecise = 0
        self.yposprecise = 0
        self.dx = 0
        self.dy = 0

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

class Ship():

    def __init__(self, mode, screen_width, screen_height, ship_number, nb_player, xpos, ypos, ship_pic, ship_pic_thrust, ship_pic_shield, lives):

        margin_size = 0
        w_percent = 1.0
        h_percent = 1.0

        if mode == "training":
            self.view_width = screen_width
            self.view_height = screen_height
            self.view_left = margin_size
            self.view_top = margin_size
        else:
            self.view_width  = int((screen_width * w_percent) / 2)
            self.view_height = int((screen_height * h_percent) / 2)

            if ship_number == 1:
                self.view_left = margin_size
                self.view_top = margin_size

            elif ship_number == 2:
                self.view_left = margin_size + self.view_width + margin_size
                self.view_top = margin_size

            elif ship_number == 3:
                self.view_left = margin_size
                self.view_top = margin_size + self.view_height + margin_size

            elif ship_number == 4:
                self.view_left = margin_size + self.view_width + margin_size
                self.view_top = margin_size + self.view_height + margin_size

        self.init_xpos = xpos
        self.init_ypos = ypos
        
        self.xpos = xpos
        self.ypos = ypos
        self.xposprecise = xpos
        self.yposprecise = ypos

        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0

        self.impactx = 0.0
        self.impacty = 0.0

        self.angle  = 0.0
        self.thrust = 0.0
        self.shield = False
        self.shoot  = False
        self.shoot_delay = False
        self.landed = False
        self.bounce = False
        self.explod = False

        self.lives = lives
        self.shots = []

        self.visited_zones = set()

        # sound
        self.sound_thrust = pygame.mixer.Sound(SOUND_THURST)
        self.sound_explod = pygame.mixer.Sound(SOUND_EXPLOD)
        self.sound_bounce = pygame.mixer.Sound(SOUND_BOUNCE)
        self.sound_shoot  = pygame.mixer.Sound(SOUND_SHOOT)
        self.sound_shield = pygame.mixer.Sound(SOUND_SHIELD)

        # controls
        self.thrust_pressed = False
        self.left_pressed   = False
        self.right_pressed  = False
        self.shoot_pressed  = False
        self.shield_pressed = False

        # ship pic: 32x32, black (0,0,0) background, no alpha
        self.ship_pic = pygame.image.load(ship_pic).convert()
        self.ship_pic.set_colorkey( (0, 0, 0) ) # used for the mask, black = background, not the ship
        self.ship_pic_thrust = pygame.image.load(ship_pic_thrust).convert()
        self.ship_pic_thrust.set_colorkey( (0, 0, 0) ) # used for the mask, black = background, not the ship
        self.ship_pic_shield = pygame.image.load(ship_pic_shield).convert()
        self.ship_pic_shield.set_colorkey( (0, 0, 0) ) # used for the mask, black = background, not the ship

        self.image = self.ship_pic
        self.mask = pygame.mask.from_surface(self.image)

        self.ray_surface = pygame.Surface((RAY_BOX_SIZE, RAY_BOX_SIZE))

    def reset(self, env):
        if env.mode == "training" and 1:
            self.xpos, self.ypos = random.choice(START_POSITIONS)
        else:
            self.xpos = self.init_xpos
            self.ypos = self.init_ypos

        self.xposprecise = self.xpos
        self.yposprecise = self.ypos

        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0

        self.impactx = 0.0
        self.impacty = 0.0

        self.angle  = 0.0
        self.thrust = 0.0
        self.shield = False
        self.shoot  = False
        self.shoot_delay = False
        self.landed = False
        self.bounce = False
        self.explod = False

        self.thrust_pressed = False
        self.left_pressed   = False
        self.right_pressed  = False
        self.shoot_pressed  = False
        self.shield_pressed = False

        self.visited_zones = set()

        self.lives -= 1

        if env.render_game:
            self.sound_thrust.stop()
            self.sound_shoot.stop()
            self.sound_shield.stop()
            self.sound_bounce.stop()

            if env.mode != "training":
                self.sound_explod.play()

    def step(self, env, action):

        if not env.play_recorded:
    
            self.thrust_pressed = False
            self.left_pressed   = False
            self.right_pressed  = False
            self.shoot_pressed  = False
            self.shield_pressed = False

            if action[0]:
                self.thrust_pressed = True
            if action[1]:
                self.left_pressed   = True
            elif action[2]:
                self.right_pressed  = True

            # record play ?
            if env.record_play:
                env.played_data.append((self.left_pressed, self.right_pressed, self.thrust_pressed, self.shield_pressed, self.shoot_pressed))

        # play recorded
        else:
            try:
                data_i = env.played_data[env.frames]

                self.left_pressed   = True if data_i[0] else False
                self.right_pressed  = True if data_i[1] else False
                self.thrust_pressed = True if data_i[2] else False
                self.shield_pressed = True if data_i[3] else False
                self.shoot_pressed  = True if data_i[4] else False
            except:
                print("End of playback")
                print("Frames=", env.frames)
                print("%s seconds" % int(env.frames/MAX_FPS))
                sys.exit(0)

        self.do_move(env)

    def update(self, env):

        # normal play
        if not env.play_recorded:
            pass

            # record play ?
            if env.record_play:
                env.played_data.append((self.left_pressed, self.right_pressed, self.thrust_pressed, self.shield_pressed, self.shoot_pressed))

        # play recorded
        else:
            try:
                data_i = env.played_data[env.frames]

                self.left_pressed   = True if data_i[0] else False
                self.right_pressed  = True if data_i[1] else False
                self.thrust_pressed = True if data_i[2] else False
                self.shield_pressed = True if data_i[3] else False
                self.shoot_pressed  = True if data_i[4] else False

            except:
                print("End of playback")
                print("Frames=", env.frames)
                print("%s seconds" % int(env.frames/MAX_FPS))
                sys.exit(0)


        self.do_move(env)

    def do_move(self, env):

        if env.motion == "basic":

            # pic
            if self.left_pressed or self.right_pressed:
                self.image = self.ship_pic_thrust
            else:
                self.image = self.ship_pic

            #
            dx = dy = 0

            if self.left_pressed:
                dx = -1
            if self.right_pressed:
                dx = 1

            self.xpos += dx
            self.ypos += dy

        elif env.motion == "thrust":

            # pic
            if self.thrust_pressed:
                self.image = self.ship_pic_thrust
            else:
                self.image = self.ship_pic

            # angle
            if self.left_pressed:
                self.angle += SHIP_ANGLESTEP
            if self.right_pressed:
                self.angle -= SHIP_ANGLESTEP

            self.angle = self.angle % 360

            if self.thrust_pressed:
                coef = 2
                self.xposprecise -= coef * math.cos( math.radians(90 - self.angle) )
                self.yposprecise -= coef * math.sin( math.radians(90 - self.angle) )
                
                # transfer to screen coordinates
                self.xpos = int(self.xposprecise)
                self.ypos = int(self.yposprecise)

        elif env.motion == "gravity":
    
            self.image = self.ship_pic
            self.thrust = 0.0
            self.shield = False

            # shield
            if self.shield_pressed:
                self.image = self.ship_pic_shield
                self.shield = True
                if env.render_game:
                    self.sound_thrust.stop()

                if env.render_game:
                    if not pygame.mixer.get_busy():
                        self.sound_shield.play(-1)
            else:
                self.shield = False
                if env.render_game:
                    self.sound_shield.stop()

                # thrust
                if self.thrust_pressed:
                    self.image = self.ship_pic_thrust

                    #self.thrust += 0.1
                    #if self.thrust >= SHIP_THRUST_MAX:
                    self.thrust = SHIP_THRUST_MAX

                    if env.mode != "training":
                        if env.render_game:
                            if not pygame.mixer.get_busy():
                                self.sound_thrust.play(-1)

                    self.landed = False

                else:
                    self.thrust = 0.0
                    if env.render_game:
                        self.sound_thrust.stop()

            # shoot delay
            if self.shoot_pressed and not self.shoot:
                self.shoot_delay = True
            else:
                self.shoot_delay = False

            # shoot
            if self.shoot_pressed:
                self.shoot = True

                if self.shoot_delay:
                    if len(self.shots) < MAX_SHOOT:
                        if env.render_game:
                            if not pygame.mixer.get_busy():
                                self.sound_shoot.play()

                        self.add_shots()
            else:
                self.shoot = False
                if env.render_game:
                    self.sound_shoot.stop()

            #
            self.bounce = False

            if not self.landed:
                # angle
                if self.left_pressed:
                    self.angle += SHIP_ANGLESTEP
                if self.right_pressed:
                    self.angle -= SHIP_ANGLESTEP

                # 
                self.angle = self.angle % 360

                # https://gafferongames.com/post/integration_basics/
                self.ax = self.thrust * -math.cos( math.radians(90 - self.angle) ) # ax = thrust * sin1
                self.ay = iG + (self.thrust * -math.sin( math.radians(90 - self.angle))) # ay = g + thrust * (-cos1)

                # shoot when shield is on
                if self.impactx or self.impacty:
                    self.ax += iCoeffimpact * self.impactx
                    self.ay += iCoeffimpact * self.impacty
                    self.impactx = 0.
                    self.impacty = 0.

                self.vx = self.vx + (iCoeffax * self.ax) # vx += coeffa * ax
                self.vy = self.vy + (iCoeffay * self.ay) # vy += coeffa * ay

                self.vx = self.vx * iXfrott # on freine de xfrott
                self.vy = self.vy * iYfrott # on freine de yfrott

                self.xposprecise = self.xposprecise + (iCoeffvx * self.vx) # xpos += coeffv * vx
                self.yposprecise = self.yposprecise + (iCoeffvy * self.vy) # ypos += coeffv * vy

            else:
                self.vx = 0.
                self.vy = 0.
                self.ax = 0.
                self.ay = 0.

            # transfer to screen coordinates
            self.xpos = int(self.xposprecise)
            self.ypos = int(self.yposprecise)

            # landed ?
            if env.mode != "training": # at the moment no landing in training (because NEAT algo is too lazy)
                self.is_landed(env)

        #
        # rotate
        self.image_rotated = pygame.transform.rotate(self.image, self.angle)
        self.mask = pygame.mask.from_surface(self.image_rotated)

        rect = self.image_rotated.get_rect()
        self.rot_xoffset = int( ((SHIP_SPRITE_SIZE - rect.width)/2) )  # used in draw() and collide_map()
        self.rot_yoffset = int( ((SHIP_SPRITE_SIZE - rect.height)/2) ) # used in draw() and collide_map()

        # number of visited zones for the reward
        self.update_visited_zones()

    def plot_shots(self, map_buffer):
        for shot in list(self.shots): # copy of self.shots
            shot.xposprecise += shot.dx
            shot.yposprecise += shot.dy
            shot.x = int(shot.xposprecise)
            shot.y = int(shot.yposprecise)

            try:
                c = map_buffer.get_at((int(shot.x), int(shot.y)))
                if (c.r != 0) or (c.g != 0) or (c.b != 0):
                    self.shots.remove(shot)

                #gfxdraw.pixel(map_buffer, int(shot.x) , int(shot.y), WHITE)
                pygame.draw.circle(map_buffer, WHITE, (int(shot.x) , int(shot.y)), 1)
                #pygame.draw.line(map_buffer, WHITE, (int(self.xpos + SHIP_SPRITE_SIZE/2), int(self.ypos + SHIP_SPRITE_SIZE/2)), (int(shot.x), int(shot.y)))

            # out of surface
            except IndexError:
                self.shots.remove(shot)

        if 0:
            for i in range(len(self.shots)):
                try:
                    shot1 = self.shots[i]
                    shot2 = self.shots[i+1]
                    pygame.draw.line(map_buffer, WHITE, (int(shot1.x), int(shot1.y)), (int(shot2.x), int(shot2.y)))
                except IndexError:
                    pass

    def add_shots(self):
        shot = Shot()

        shot.x = (self.xpos+15) + 18 * -math.cos(math.radians(90 - self.angle))
        shot.y = (self.ypos+16) + 18 * -math.sin(math.radians(90 - self.angle))
        shot.xposprecise = shot.x
        shot.yposprecise = shot.y
        shot.dx = 5.1 * -math.cos(math.radians(90 - self.angle))
        shot.dy = 5.1 * -math.sin(math.radians(90 - self.angle))
        shot.dx += self.vx / 3.5
        shot.dy += self.vy / 3.5

        self.shots.append(shot)

    def is_landed(self, env):

        for plaform in PLATFORMS_1:
            xmin  = plaform[0] - (SHIP_SPRITE_SIZE - 23)
            xmax  = plaform[1] - (SHIP_SPRITE_SIZE - 9)
            yflat = plaform[2] - (SHIP_SPRITE_SIZE - 2)

            #print(self.ypos, yflat)

            if ((xmin <= self.xpos) and (self.xpos <= xmax) and
               ((self.ypos == yflat) or ((self.ypos-1) == yflat) or ((self.ypos-2) == yflat) or ((self.ypos-3) == yflat) ) and
               (self.vy > 0) and (self.angle<=SHIP_ANGLE_LAND or self.angle>=(360-SHIP_ANGLE_LAND)) ):

                self.vy = - self.vy / 1.2
                self.vx = self.vx / 1.1
                self.angle = 0
                self.ypos = yflat
                self.yposprecise = yflat

                if ( (-1.0/SLOW_DOWN_COEF <= self.vx) and (self.vx < 1.0/SLOW_DOWN_COEF) and (-1.0/SLOW_DOWN_COEF < self.vy) and (self.vy < 1.0/SLOW_DOWN_COEF) ):
                    self.landed = True
                    self.bounce = False
                else:
                    self.bounce = True
                    if env.render_game:
                        self.sound_bounce.play()

                return True

        return False

    def do_test_collision(self):
        test_it = True

        for plaform in PLATFORMS_1:
            xmin  = plaform[0] - (SHIP_SPRITE_SIZE - 23)
            xmax  = plaform[1] - (SHIP_SPRITE_SIZE - 9)
            yflat = plaform[2] - (SHIP_SPRITE_SIZE - 2)

            #if ((xmin<=self.xpos) and (self.xpos<=xmax) and ((self.ypos==yflat) or ((self.ypos-1)==yflat) or ((self.ypos-2)==yflat) or ((self.ypos-3)==yflat))  and  (self.angle<=SHIP_ANGLE_LAND or self.angle>=(360-SHIP_ANGLE_LAND)) ):
            #    test_it = False
            #    break
            if (self.shield and (xmin<=self.xpos) and (self.xpos<=xmax) and ((self.ypos==yflat) or ((self.ypos-1)==yflat) or ((self.ypos-2)==yflat) or ((self.ypos-3)==yflat) or ((self.ypos+1)==yflat)) and  (self.angle<=SHIP_ANGLE_LAND or self.angle>=(360-SHIP_ANGLE_LAND)) ):
                test_it = False
                break
            if ((self.thrust) and (xmin<=self.xpos) and (self.xpos<=xmax) and ((self.ypos==yflat) or ((self.ypos-1)==yflat) or ((self.ypos+1)==yflat) )):
                test_it = False
                break

        return test_it

    def draw(self, map_buffer):
        #game_window.blit(self.image_rotated, (self.view_width/2 + self.view_left + self.rot_xoffset, self.view_height/2 + self.view_top + self.rot_yoffset))
        map_buffer.blit(self.image_rotated, (self.xpos + self.rot_xoffset, self.ypos + self.rot_yoffset))

    def collide_map(self, map_buffer, map_buffer_mask):

        # ship size mask
        if USE_MINI_MASK:
            mini_area = Rect(self.xpos, self.ypos, SHIP_SPRITE_SIZE, SHIP_SPRITE_SIZE)
            mini_subsurface = map_buffer.subsurface(mini_area)
            mini_subsurface.set_colorkey( (0, 0, 0) ) # used for the mask, black = background
            mini_mask = pygame.mask.from_surface(mini_subsurface)

            if self.do_test_collision():
                offset = (self.rot_xoffset, self.rot_yoffset) # pos of the ship

                if mini_mask.overlap(self.mask, offset): # https://stackoverflow.com/questions/55817422/collision-between-masks-in-pygame/55818093#55818093
                    self.explod = True

        # player view size mask
        else:
            if self.do_test_collision():
                offset = (self.xpos + self.rot_xoffset, self.ypos + self.rot_yoffset) # pos of the ship

                if map_buffer_mask.overlap(self.mask, offset): # https://stackoverflow.com/questions/55817422/collision-between-masks-in-pygame/55818093#55818093
                    self.explod = True

    # 
    def collide_ship(self, ships):
        for ship in ships:
            if self != ship:
                offset = ((ship.xpos - self.xpos), (ship.ypos - self.ypos))
                if self.mask.overlap(ship.mask, offset):
                    self.explod = True
                    ship.explod = True

    # 
    def collide_shots(self, ships):
        for ship in ships:
            if self != ship:
                for shot in self.shots:
                    try:
                        if ship.mask.get_at((shot.x - ship.xpos, shot.y - ship.ypos)):
                            if not ship.shield:
                                ship.explod = True
                                return
                            else:
                                ship.impactx = shot.dx
                                ship.impacty = shot.dy
                    # out of ship mask => no collision
                    except IndexError:
                        pass

    def ray_sensor(self, env, render_game=True):
        # TODO use smaller map masks
        # TODO use only 0 to 90 degres ray mask quadran: https://github.com/Rabbid76/PyGameExamplesAndAnswers/blob/master/examples/minimal_examples/pygame_minimal_mask_intersect_surface_line_2.py

        # clipping translation for window coordinates
        rx = self.xpos - self.view_width/2
        ry = self.ypos - self.view_height/2

        dx = 0
        dy = 0

        if rx < 0:
            dx = rx
        elif rx > (MAP_WIDTH - self.view_width):
            dx = rx - (MAP_WIDTH - self.view_width)
        if ry < 0:
            dy = ry
        elif ry > (MAP_HEIGHT - self.view_height):
            dy = ry - (MAP_HEIGHT - self.view_height)

        #sub_area1 = Rect(rx, ry, self.view_width, self.view_height)
        #self.env.game.window.blit(self.env.game.map_buffer, (self.view_left, self.view_top), sub_area1)

        # in window coord, center of the player view
        ship_window_pos = (int(self.view_width/2) + self.view_left + SHIP_SPRITE_SIZE/2 + dx, int(self.view_height/2) + self.view_top + SHIP_SPRITE_SIZE/2 + dy)
        #print("ship_window_pos", ship_window_pos)

        ray_surface_center = (int(RAY_BOX_SIZE/2), int(RAY_BOX_SIZE/2))

        wall_distances = []

        # 30 degres step
        for angle in range(0, 359, RAY_AMGLE_STEP):

            c = math.cos(math.radians(angle))
            s = math.sin(math.radians(angle))

            flip_x = c < 0
            flip_y = s < 0

            filpped_map_mask = env.game.flipped_masks_map_buffer[flip_x][flip_y]

            # ray final point
            x_dest = ray_surface_center[0] + RAY_BOX_SIZE/2 * abs(c)
            y_dest = ray_surface_center[1] + RAY_BOX_SIZE/2 * abs(s)

            #x_dest = ray_surface_center[0] + RAY_BOX_SIZE * abs(c)
            #y_dest = ray_surface_center[1] + RAY_BOX_SIZE * abs(s)

            self.ray_surface.fill((0, 0, 0))
            self.ray_surface.set_colorkey((0, 0, 0))
            pygame.draw.line(self.ray_surface, WHITE, ray_surface_center, (x_dest, y_dest))
            ray_mask = pygame.mask.from_surface(self.ray_surface)
            pygame.draw.circle(self.ray_surface, RED, ray_surface_center, 3)

            # offset = ray mask (left/top) coordinate in the map (ie where to put our lines mask in the map)
            if flip_x:
                offset_x = MAP_WIDTH - (self.xpos+SHIP_SPRITE_SIZE/2) - int(RAY_BOX_SIZE/2)
            else:
                offset_x = self.xpos+SHIP_SPRITE_SIZE/2 - int(RAY_BOX_SIZE/2)

            if flip_y:
                offset_y = MAP_HEIGHT - (self.ypos+SHIP_SPRITE_SIZE/2) - int(RAY_BOX_SIZE/2)
            else:
                offset_y = self.ypos+SHIP_SPRITE_SIZE/2 - int(RAY_BOX_SIZE/2)

            #print("offset", offset_x, offset_y)
            hit = filpped_map_mask.overlap(ray_mask, (int(offset_x), int(offset_y)))
            #print("hit", hit)

            if hit is not None and (hit[0] != self.xpos+SHIP_SPRITE_SIZE/2 or hit[1] != self.ypos+SHIP_SPRITE_SIZE/2):
                hx = MAP_WIDTH-1 - hit[0] if flip_x else hit[0]
                hy = MAP_HEIGHT-1 - hit[1] if flip_y else hit[1]
                hit = (hx, hy)
                #print("new hit", hit)

                # go back to screen coordinates
                dx_hit = hit[0] - (self.xpos+SHIP_SPRITE_SIZE/2)
                dy_hit = hit[1] - (self.ypos+SHIP_SPRITE_SIZE/2)

                if render_game:
                    pygame.draw.line(env.game.window, LVIOLET, ship_window_pos, (ship_window_pos[0] + dx_hit, ship_window_pos[1] + dy_hit))
                    #pygame.draw.circle(map, RED, hit, 2)

                # Note: this is the distance from the center of the ship, not the borders
                #dist_wall = math.sqrt(dx_hit**2 + dy_hit**2)
                dist_wall = np.linalg.norm(np.array((0, 0)) - np.array((dx_hit, dy_hit)))

                # so remove
                dist_wall -= (SHIP_SPRITE_SIZE/2 - 1)

            # Not hit: too far
            else:
                dist_wall = RAY_MAX_LEN

            if dist_wall < 0:
                dist_wall = 0

            wall_distances.append(dist_wall)

            #print("Sensor for angle=%s, dist wall=%.2f" % (str(angle), dist_wall))

            #env.game.window.blit(RAY_SURFACE, (self.view_left + self.view_width, self.view_top))
            #map.blit(RAY_SURFACE, (int(offset_x), int(offset_y)))

        return wall_distances

    # The map is splitted into 24 zones, 4x * 6y
    # 0 1 2 3
    # 4 5 6 7
    # ....
    # 20 21 22 23
    def update_visited_zones(self):
        zone_number = (self.xpos // ZONE_X_SIZE) + ((self.ypos // ZONE_Y_SIZE) * ZONE_X_NUM)
        self.visited_zones.add(zone_number)
        #print(self.visited_zones)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

#class MayhemEnv():
class MayhemEnv(gym.Env):
    
    def __init__(self, game, vsync=True, render_game=True, nb_player=2, mode="game", motion="gravity", sensor="", record_play="", play_recorded=""):
        #super.__init__(self, t, obj)
        
        self.myfont = pygame.font.SysFont('Arial', 20)

        self.render_game = render_game
        self.vsync = vsync
        self.nb_player = nb_player

        # screen
        self.game = game
        self.game.window.fill((0, 0, 0))

        self.mode   = mode   # training or game
        self.motion = motion # basic, thrust, gravity
        self.sensor = sensor

        # record / play recorded
        self.record_play = record_play
        self.played_data = [] # [(0,0,0), (0,0,1), ...] (left, right, thrust)

        self.play_recorded = play_recorded

        if self.play_recorded:
            with open(self.play_recorded, "rb") as f:
                self.played_data = pickle.load(f)

        # FPS
        self.clock = pygame.time.Clock()
        self.paused = False
        self.frames = 0

        self.nb_dead = 0

        self.ships = []

        if self.mode == "game":
            self.ship_1 = Ship(self.mode, self.game.screen_width, self.game.screen_height, 1, self.nb_player, SHIP1_X, SHIP1_Y, \
                                   SHIP_1_PIC, SHIP_1_PIC_THRUST, SHIP_1_PIC_SHIELD, SHIP_MAX_LIVES - self.nb_dead)

            self.ship_2 = Ship(self.mode, self.game.screen_width, self.game.screen_height, 2, self.nb_player, SHIP2_X, SHIP2_Y, \
                               SHIP_2_PIC, SHIP_2_PIC_THRUST, SHIP_2_PIC_SHIELD, SHIP_MAX_LIVES - self.nb_dead)

            self.ship_3 = Ship(self.mode, self.game.screen_width, self.game.screen_height, 3, self.nb_player, SHIP3_X, SHIP3_Y, \
                               SHIP_3_PIC, SHIP_3_PIC_THRUST, SHIP_3_PIC_SHIELD, SHIP_MAX_LIVES - self.nb_dead)

            self.ship_4 = Ship(self.mode, self.game.screen_width, self.game.screen_height, 4, self.nb_player, SHIP4_X, SHIP4_Y, \
                               SHIP_4_PIC, SHIP_4_PIC_THRUST, SHIP_4_PIC_SHIELD, SHIP_MAX_LIVES - self.nb_dead)

            self.ships.append(self.ship_1)
            
            if self.nb_player >= 2:
                self.ships.append(self.ship_2)
            if self.nb_player >= 3:
                self.ships.append(self.ship_3)
            if self.nb_player >= 4:
                self.ships.append(self.ship_4)

        # -- training params

        # left, right, thrust
        self.action_space = spaces.MultiBinary(3) # np.array([1,0,1]).astype(np.int8)

        # angle vx vy ax ay + 8 dist
        # [-1.          0.         -0.1384108   0.         -0.25        1.
        #   -0.24606602 -0.44138566 -0.30105048  1.          1.         -0.5898296
        #    1.        ]

        #low  = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]).astype(np.float32)
        #high = np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0]).astype(np.float32)

        # thrust, left, right, angle vx vy ax ay + 8 dist
        low  = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]).astype(np.float32)
        high = np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0]).astype(np.float32)

        self.observation_space = spaces.Box(low, high)
        
        self.done = False
        self.frame_pic = None

        if self.mode == "training":
            self.ship_1 = Ship(self.mode, self.game.screen_width, self.game.screen_height, 1, 1, 430, 730, \
                               SHIP_1_PIC, SHIP_1_PIC_THRUST, SHIP_1_PIC_SHIELD, SHIP_MAX_LIVES)

        #self.model = PPO.load("/home/tony/Desktop/dev/devpack-github/mayhem-ai/logs/models/PPO/460000.zip", env=self)
        #self.obs = self.reset()

    def close(self):
        # TODO pygame cleanup
        pass

    def main_loop(self):

        # exit on Quit
        while True:

            # real training loop is done in reset() / step() / render()
            # practice_loop is just a free flight
            if self.mode == "training":
                self.practice_loop()

            # game
            elif self.mode == "game":
                self.game_loop()

            for ship in self.ships:
                ship.sound_thrust.stop()
                ship.sound_bounce.stop()
                ship.sound_shield.stop()
                ship.sound_shoot.stop()

                if env.mode != "training":
                    ship.sound_explod.play()

            self.nb_dead += 1

            # record play ?
            self.record_it()

    def record_it(self):
        if self.record_play:
            with open(self.record_play, "wb") as f:
                pickle.dump(self.played_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            time.sleep(0.1)
            print("Frames=", self.frames)
            print("%s seconds" % int(self.frames/MAX_FPS))
            sys.exit(0)

    def ship_key_down(self, key, ship, key_mapping):

        # ship_x
        if key == key_mapping["left"]:
            ship.left_pressed = True
        if key == key_mapping["right"]:
            ship.right_pressed = True
        if key == key_mapping["thrust"]:
            ship.thrust_pressed = True
        if key == key_mapping["shoot"]:
            ship.shoot_pressed = True
        if key == key_mapping["shield"]:
            ship.shield_pressed = True

    def ship_key_up(self, key, ship, key_mapping):

        # ship_x
        if key == key_mapping["left"]:
            ship.left_pressed = False
        if key == key_mapping["right"]:
            ship.right_pressed = False
        if key == key_mapping["thrust"]:
            ship.thrust_pressed = False
        if key == key_mapping["shoot"]:
            ship.shoot_pressed = False
        if key == key_mapping["shield"]:
            ship.shield_pressed = False

    def ship_joystick(self, ship, joy_mapping):

        if joy_mapping:
            try:
                if pygame.joystick.Joystick(joy_mapping-1).get_button(0):
                    ship.thrust_pressed = True
                else:
                    ship.thrust_pressed = False

                if pygame.joystick.Joystick(joy_mapping-1).get_button(5):
                    ship.shoot_pressed = True
                else:
                    ship.shoot_pressed = False

                if pygame.joystick.Joystick(joy_mapping-1).get_button(1):
                    ship.shield_pressed = True
                else:
                    ship.shield_pressed = False

                horizontal_axis = pygame.joystick.Joystick(joy_mapping-1).get_axis(0)

                if int(round(horizontal_axis)) == 1:
                    ship.right_pressed = True
                else:
                    ship.right_pressed = False

                if int(round(horizontal_axis)) == -1:
                    ship.left_pressed = True
                else:
                    ship.left_pressed = False
            except:
                pass

    def game_loop(self):

        # Game Main Loop
        while True:

            # -- pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.record_it()
                    sys.exit(0)

                # - key down
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.record_it()
                        sys.exit(0)
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                    elif event.key == pygame.K_DELETE:
                        self.ship_1.explod = True

                    if SHIP_1_INPUT == "HUMAN":
                        self.ship_key_down(event.key, self.ship_1, SHIP_1_KEYS)
                    if SHIP_2_INPUT == "HUMAN":
                        self.ship_key_down(event.key, self.ship_2, SHIP_2_KEYS)
                    if SHIP_3_INPUT == "HUMAN":
                        self.ship_key_down(event.key, self.ship_3, SHIP_3_KEYS)
                    if SHIP_4_INPUT == "HUMAN":
                        self.ship_key_down(event.key, self.ship_4, SHIP_4_KEYS)

                # - key up
                elif event.type == pygame.KEYUP:

                    if SHIP_1_INPUT == "HUMAN":
                        self.ship_key_up(event.key, self.ship_1, SHIP_1_KEYS)
                    if SHIP_2_INPUT == "HUMAN":
                        self.ship_key_up(event.key, self.ship_2, SHIP_2_KEYS)
                    if SHIP_3_INPUT == "HUMAN":
                        self.ship_key_up(event.key, self.ship_3, SHIP_3_KEYS)
                    if SHIP_4_INPUT == "HUMAN":
                        self.ship_key_up(event.key, self.ship_4, SHIP_4_KEYS)
                        
            if 0:
                # Joystick
                if SHIP_1_JOY and SHIP_1_INPUT == "HUMAN":
                    self.ship_joystick(self.ship_1, SHIP_1_JOY)
                if SHIP_2_JOY and SHIP_2_INPUT == "HUMAN":
                    self.ship_joystick(self.ship_2, SHIP_2_JOY)
                if SHIP_3_JOY and SHIP_3_INPUT == "HUMAN":
                    self.ship_joystick(self.ship_3, SHIP_3_JOY)
                if SHIP_4_JOY and SHIP_4_INPUT == "HUMAN":
                    self.ship_joystick(self.ship_4, SHIP_4_JOY)

            #action, _states = self.model.predict(self.obs, deterministic=False)
            #self.obs, rewards, done, info = self.step(action)

            # -- 
            if not self.paused:

                # clear screen
                self.game.window.fill((0,0,0))

                # map copy (TODO reduce)
                self.game.map_buffer.blit(self.game.map, (0, 0))

                # update ship pos
                for ship in self.ships:
                    ship.update(self)

                # collide_map and ship tp ship
                for ship in self.ships:
                    ship.collide_map(self.game.map_buffer, self.game.map_buffer_mask)

                for ship in self.ships:
                    ship.collide_ship(self.ships)
                    
                for ship in self.ships:
                    ship.plot_shots(self.game.map_buffer)

                for ship in self.ships:
                    ship.collide_shots(self.ships)

                # blit ship in the map
                for ship in self.ships:
                    ship.draw(self.game.map_buffer)

                for ship in self.ships:

                    # clipping to avoid black when the ship is close to the edges
                    rx = ship.xpos - ship.view_width/2
                    ry = ship.ypos - ship.view_height/2
                    if rx < 0:
                        rx = 0
                    elif rx > (MAP_WIDTH - ship.view_width):
                        rx = (MAP_WIDTH - ship.view_width)
                    if ry < 0:
                        ry = 0
                    elif ry > (MAP_HEIGHT - ship.view_height):
                        ry = (MAP_HEIGHT - ship.view_height)

                    # blit the map area around the ship on the screen
                    sub_area1 = Rect(rx, ry, ship.view_width, ship.view_height)
                    self.game.window.blit(self.game.map_buffer, (ship.view_left, ship.view_top), sub_area1)

                # sensors
                if self.sensor == "ray":
                    for ship in self.ships:
                        ship.ray_sensor(self)

                for ship in self.ships:
                    if ship.explod:
                        ship.reset(self)

                # debug on screen
                self.screen_print_info()

                cv = (225, 225, 225)
                pygame.draw.line( self.game.window, cv, (0, int(self.game.screen_height/2)), (self.game.screen_width, int(self.game.screen_height/2)) )
                pygame.draw.line( self.game.window, cv, (int(self.game.screen_width/2), 0), (int(self.game.screen_width/2), (self.game.screen_height)) )

                # display
                pygame.display.flip()
                self.frames += 1

            self.clock.tick(MAX_FPS) # https://python-forum.io/thread-16692.html

            #print(self.clock.get_fps())

    def practice_loop(self):

        # Game Main Loop
        while not self.ship_1.explod:

            # clear screen
            self.game.window.fill((0,0,0))

            # pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        sys.exit(0)
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                    elif event.key == pygame.K_DELETE:
                        self.ship_1.explod = True

            if not self.paused:            
                # map copy (TODO reduce)
                self.game.map_buffer.blit(self.game.map, (0, 0))

                self.ship_1.update(self)

                # collision
                self.ship_1.collide_map(self.game.map_buffer, self.game.map_buffer_mask)

                # blit ship in the map
                self.ship_1.draw(self.game.map_buffer)

                # clipping to avoid black when the ship is close to the edges
                rx = self.ship_1.xpos - self.ship_1.view_width/2
                ry = self.ship_1.ypos - self.ship_1.view_height/2
                if rx < 0:
                    rx = 0
                elif rx > (MAP_WIDTH - self.ship_1.view_width):
                    rx = (MAP_WIDTH - self.ship_1.view_width)
                if ry < 0:
                    ry = 0
                elif ry > (MAP_HEIGHT - self.ship_1.view_height):
                    ry = (MAP_HEIGHT - self.ship_1.view_height)

                # blit the map area around the ship on the screen
                sub_area1 = Rect(rx, ry, self.ship_1.view_width, self.ship_1.view_height)
                self.game.window.blit(self.game.map_buffer, (self.ship_1.view_left, self.ship_1.view_top), sub_area1)

                # sensors
                if self.sensor == "ray":
                    self.ship_1.ray_sensor(self)

                # debug on screen
                self.screen_print_info()

                # display
                pygame.display.flip()

                self.frames += 1
                #print(self.clock.get_fps())

            self.clock.tick(MAX_FPS) # https://python-forum.io/thread-16692.html


    def screen_print_info(self):
        # debug text
        if DEBUG_SCREEN:
            ship_pos = self.myfont.render('Pos: %s %s' % (self.ship_1.xpos, self.ship_1.ypos), False, (255, 255, 255))
            self.game.window.blit(ship_pos, (DEBUG_TEXT_XPOS + 5, 30))

            ship_va = self.myfont.render('vx=%.2f, vy=%.2f, ax=%.2f, ay=%.2f' % (self.ship_1.vx,self.ship_1.vy, self.ship_1.ax, self.ship_1.ay), False, (255, 255, 255))
            self.game.window.blit(ship_va, (DEBUG_TEXT_XPOS + 5, 55))

            ship_angle = self.myfont.render('Angle: %s' % (self.ship_1.angle,), False, (255, 255, 255))
            self.game.window.blit(ship_angle, (DEBUG_TEXT_XPOS + 5, 80))

            dt = self.myfont.render('Frames: %s' % (self.frames,), False, (255, 255, 255))
            self.game.window.blit(dt, (DEBUG_TEXT_XPOS + 5, 105))

            if self.mode == "game":
                fps = self.myfont.render('FPS: %.2f' % self.clock.get_fps(), False, (255, 255, 255))
                self.game.window.blit(fps, (DEBUG_TEXT_XPOS + 5, 130))
            elif self.mode == "training":
                try:
                    step_dist = self.myfont.render('Step dist: %s' % self.step_dist, False, (255, 255, 255))
                    self.game.window.blit(step_dist, (DEBUG_TEXT_XPOS + 5, 130)) 

                    reward = self.myfont.render('Reward: %s' % self.reward, False, (255, 255, 255))
                    self.game.window.blit(reward, (DEBUG_TEXT_XPOS + 5, 155))  

                    zones = self.myfont.render('Zones: %s' % str(len(self.ship_1.visited_zones)), False, (255, 255, 255))
                    self.game.window.blit(zones, (DEBUG_TEXT_XPOS + 5, 180))  

                except:
                    pass 

            #ship_lives = self.myfont.render('Lives: %s' % (self.ship_1.lives,), False, (255, 255, 255))
            #self.game.window.blit(ship_lives, (DEBUG_TEXT_XPOS + 5, 105))

    # training only
    def reset(self):
        self.frames = 0
        self.done = False
        self.paused = False
        self.ship_1.reset(self)

        #new_state = [old_thrust_pressed, old_left_pressed, old_right_pressed, angle_norm, vx_norm, vy_norm, ax_norm, ay_norm]
        #new_state.extend(wall_distances)

        new_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        if self.sensor == "ray":
            wall_distances = [0, 0, 0, 0, 0, 0, 0, 0]
            wall_distances = [-1, -1, -1, -1, -1, -1, -1, -1]
            new_state.extend(wall_distances)
        elif self.sensor == "pic":
            new_state.extend([0]*32*32)

        return np.array(new_state, dtype=np.float32)

    # training only
    def step(self, action, max_frame=2000):

        # pause ?
        if self.paused:
            return None, None, None, {}
            
        # needed for ray sensor
        self.game.window.fill((0,0,0))

        done = False

        old_xpos = self.ship_1.xposprecise
        old_ypos = self.ship_1.yposprecise

        if self.ship_1.thrust_pressed:
            old_thrust_pressed = 1.0
        else:
            old_thrust_pressed = -1.0

        if self.ship_1.left_pressed:
            old_left_pressed = 1.0
        else:
            old_left_pressed = -1.0
            
        if self.ship_1.right_pressed:
            old_right_pressed = 1.0
        else:
            old_right_pressed = -1.0

        # -------
        # Update ship xposprecise, yposprecise, xpos, ypos, collided etc
        self.ship_1.step(self, action)

        # https://www.baeldung.com/cs/normalizing-inputs-artificial-neural-network
        # https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
        # min-max: (((x - min) / (max - min)) * (end - start)) + start (typically start=0, end=1)

        if self.sensor == "ray":
            wall_distances = self.ship_1.ray_sensor(self)

            if NORMALIZE_INPUT:
                for i, dist in enumerate(wall_distances):
                    #wall_distances[i] = dist / RAY_MAX_LEN # [0, 1]
                    wall_distances[i] = ((dist / RAY_MAX_LEN)*2) - 1 # [-1, 1]

        # normalized wall_distances in [0, 1] or [-1, 1]
        #print(wall_distances)

        # normalized ship physic params
        if NORMALIZE_INPUT:
            #angle_norm = self.ship_1.angle / (360. - SHIP_ANGLESTEP)
            angle_norm = ((self.ship_1.angle / (360. - SHIP_ANGLESTEP)) * 2) - 1

            # vx range = [-5.5, +5.5] (more or less with default phisical values, for "standard playing")
            # vy range = [-6.5, +8.5] (more or less with default phisical values, for "standard playing")
            # ax range = [-0.16, +0.16] (more or less with default phisical values, for "standard playing")
            # vx range = [-0.12, +0.20] (more or less with default phisical values, for "standard playing")

            vx_min = -5.5  ; vx_max = 5.5
            vy_min = -6.5  ; vy_max = 8.5
            ax_min = -0.16 ; ax_max = 0.16
            ay_min = -0.13 ; ay_max = 0.20

            vx_norm = (((self.ship_1.vx - vx_min) / (vx_max - vx_min)) * 2) - 1
            vy_norm = (((self.ship_1.vy - vy_min) / (vy_max - vy_min)) * 2) - 1
            ax_norm = (((self.ship_1.ax - ax_min) / (ax_max - ax_min)) * 2) - 1
            ay_norm = (((self.ship_1.ay - ay_min) / (ay_max - ay_min)) * 2) - 1

        # raw input
        else:
            angle_norm = self.ship_1.angle
            vy_norm    = self.ship_1.vy
            vx_norm    = self.ship_1.vx
            ay_norm    = self.ship_1.ay
            ax_norm    = self.ship_1.ax

        # a change here have to be reported in reset(), obs env (if neat used => neat config file)
        new_state = [old_thrust_pressed, old_left_pressed, old_right_pressed, angle_norm, vx_norm, vy_norm, ax_norm, ay_norm]

        if self.sensor == "ray":
            new_state.extend(wall_distances)

        elif self.sensor == "pic":
            if self.frame_pic is not None:
                if NORMALIZE_INPUT:
                    new_state.extend( ((self.frame_pic/255)*2)-1 )
                    #new_state.extend( self.frame_pic/255 )
                else:
                    new_state.extend( self.frame_pic )
            else:
                new_state.extend( [0]*32*32 )

        #print(new_state)


        # --- collision (dist from wall is sensor = ray, if sensor = pic, real collision detection)
        collision = self.ship_1.explod

        if self.sensor == "ray":
            for dist in wall_distances:
                #if dist == 0:  # if normalized in [0, 1]
                if dist == -1: # if normalized in [-1, 1]
                    collision = True
                    break

        # --- revard

        done = self.ship_1.explod
        done |= (self.frames > max_frame)
        done |= collision

        # ship has not moved ?
        #d = math.sqrt((old_xpos - self.ship_1.xposprecise)**2 + (old_ypos - self.ship_1.yposprecise)**2)
        self.step_dist = np.linalg.norm(np.array((old_xpos, old_ypos)) - np.array((self.ship_1.xposprecise, self.ship_1.yposprecise)))

        #d_end = math.sqrt((self.ship_1.init_xpos - self.ship_1.xpos)**2 + (self.ship_1.init_ypos - self.ship_1.ypos)**2)
        #d_end = np.linalg.norm(np.array((self.ship_1.init_xpos, self.ship_1.init_ypos)) - np.array((self.ship_1.xpos, self.ship_1.ypos)))

        self.reward = 0

        if 0:
            if self.step_dist < 0.05:
                self.reward = -self.step_dist
            else:
                self.reward = self.step_dist        

            #print("d=", d)

        if done:
            # visited zone
            self.number_visited_zone = len(self.ship_1.visited_zones)
            self.reward = self.number_visited_zone * 2

            if self.number_visited_zone == 1:
                self.reward = 0

            # if ship exploded => bad reward
            if collision or self.ship_1.explod:
                self.reward -= 10

        #self.reward = self.reward / 50.

        self.frames += 1

        if self.sensor == "ray":
            obs = np.array(new_state, dtype=np.float32)
            
            # # angle vx vy ax ay + 8 dist
            # [-1.          0.         -0.1384108   0.         -0.25  
            #  1.             -0.24606602 -0.44138566 -0.30105048  1.          1.         -0.5898296         1.        ]
            #print(obs)

            #print(obs)
            #print("reward=", self.reward)
            return obs, self.reward, done, {}

        elif self.sensor == "pic":
            #print(new_state)
            new_state = np.reshape(new_state, 32*32 + 5)
            return new_state, self.reward, done, {}

    # training only
    def render(self, collision_check=True):

        # pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.record_it()
                sys.exit(0)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.record_it()
                    sys.exit(0)
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_DELETE:
                    self.ship_1.explod = True

        if not self.paused:

            # clear screen: done in self.step()

            # map copy (TODO reduce)
            self.game.map_buffer.blit(self.game.map, (0, 0))

            # collision (when false we use the sensor to detect a collision)
            if collision_check:
                try:
                    self.ship_1.collide_map(self.game.map_buffer, self.game.map_buffer_mask)
                # ValueError: subsurface rectangle outside surface area
                except ValueError:
                    self.ship_1.explod = True
                    
            # blit ship in the map
            self.ship_1.draw(self.game.map_buffer)

            # clipping to avoid black when the ship is close to the edges
            rx = self.ship_1.xpos - self.ship_1.view_width/2
            ry = self.ship_1.ypos - self.ship_1.view_height/2
            if rx < 0:
                rx = 0
            elif rx > (MAP_WIDTH - self.ship_1.view_width):
                rx = (MAP_WIDTH - self.ship_1.view_width)
            if ry < 0:
                ry = 0
            elif ry > (MAP_HEIGHT - self.ship_1.view_height):
                ry = (MAP_HEIGHT - self.ship_1.view_height)

            # blit the map area around the ship on the screen
            sub_area1 = Rect(rx, ry, self.ship_1.view_width, self.ship_1.view_height)
            self.game.window.blit(self.game.map_buffer, (self.ship_1.view_left, self.ship_1.view_top), sub_area1)

            # debug on screen
            self.screen_print_info()

            # display
            # if debug_on_screen is ON, flip() is done after processing debug info
            if not self.game.debug_on_screen:
                pygame.display.flip()

                if self.vsync:
                    self.clock.tick(MAX_FPS)

            return self.game.map_buffer.subsurface(sub_area1)

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

class GameWindow():

    def __init__(self, screen_width, screen_height, mode, debug_on_screen=False):

        pygame.display.set_caption('Mayhem')

        self.debug_on_screen = debug_on_screen

        if mode == "training":
            self.screen_width = 400
            self.screen_height = 400

            if self.debug_on_screen:
                self.window = pygame.display.set_mode((self.screen_width*2, self.screen_height))
            else:
                self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
        else:
            self.screen_width = screen_width
            self.screen_height = screen_height

            flags = pygame.DOUBLEBUF #| pygame.NOFRAME # | pygame.FULLSCREEN 
            self.window = pygame.display.set_mode((screen_width, screen_height), flags)

        # Background
        self.map = pygame.image.load(MAP_1).convert() # .convert_alpha()
        #self.map.set_colorkey( (0, 0, 0) ) # used for the mask, black = background
        #self.map_rect = self.map.get_rect()
        #self.map_mask = pygame.mask.from_surface(self.map)
        #self.mask_map_fx = pygame.mask.from_surface(pygame.transform.flip(self.map, True, False))
        #self.mask_map_fy = pygame.mask.from_surface(pygame.transform.flip(self.map, False, True))
        #self.mask_map_fx_fy = pygame.mask.from_surface(pygame.transform.flip(self.map, True, True))
        #self.flipped_masks = [[self.map_mask, self.mask_map_fy], [self.mask_map_fx, self.mask_map_fx_fy]]

        self.map_buffer = self.map.copy() # pygame.Surface((self.map.get_width(), self.map.get_height()))

        self.map_buffer.set_colorkey( (0, 0, 0) )
        self.map_buffer_mask = pygame.mask.from_surface(self.map_buffer)
        self.mask_map_buffer_fx = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer, True, False))
        self.mask_map_buffer_fy = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer, False, True))
        self.mask_map_buffer_fx_fy = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer, True, True))
        self.flipped_masks_map_buffer = [[self.map_buffer_mask, self.mask_map_buffer_fy], [self.mask_map_buffer_fx, self.mask_map_buffer_fx_fy]]

# -------------------------------------------------------------------------------------------------

def init_pygame():
    pygame.mixer.pre_init(frequency=22050)
    pygame.init()
    #pygame.display.init()

    pygame.mouse.set_visible(False)
    pygame.font.init()
    pygame.mixer.init() # frequency=22050

    #pygame.event.set_blocked((MOUSEMOTION, MOUSEBUTTONUP, MOUSEBUTTONDOWN))

    # joystick
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    print("joystick_count", joystick_count)

    for i in range(joystick_count):
        j = pygame.joystick.Joystick(i)
        j.init()

# -------------------------------------------------------------------------------------------------

def run():
    parser = argparse.ArgumentParser()

    parser.add_argument('-width', '--width', help='', type=int, action="store", default=1200)
    parser.add_argument('-height', '--height', help='', type=int, action="store", default=800)
    parser.add_argument('-np', '--nb_player', help='', type=int, action="store", default=4)

    parser.add_argument('-m', '--motion', help='How the ship moves', action="store", default='gravity', choices=("basic", "thrust", "gravity"))
    parser.add_argument('-r', '--record_play', help='', action="store", default="")
    parser.add_argument('-pr', '--play_recorded', help='', action="store", default="")
    parser.add_argument('-s', '--sensor', help='', action="store", default="", choices=("ray", ""))
    parser.add_argument('-rm', '--run_mode', help='', action="store", default="game", choices=("game", "training", ))

    result = parser.parse_args()
    args = dict(result._get_kwargs())

    print("Args", args)

    # window
    global game_window
    game_window = GameWindow(args["width"], args["height"], args["run_mode"], debug_on_screen=0)

    # Game mode
    if args["run_mode"] == "game":
        env = MayhemEnv(game_window, vsync=True, render_game=True, nb_player=args["nb_player"], mode=args["run_mode"], motion=args["motion"], \
                        sensor=args["sensor"], record_play=args["record_play"], play_recorded=args["play_recorded"])
        env.main_loop()

    # Training mode
    else:
        # Manual testing env mode
        env = MayhemEnv(game_window, vsync=True, render_game=True, nb_player=args["nb_player"], mode=args["run_mode"], motion=args["motion"],
                        sensor=args["sensor"], record_play=args["record_play"], play_recorded=args["play_recorded"])

        env.main_loop()
            
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    init_pygame()
    run()

# -*- coding: utf-8 -*-
"""
Usage example:

python3 mayhem.py --width=512 --height=512 --sensor=ray --fps=60

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

import gymnasium

# -------------------------------------------------------------------------------------------------
# General

DEBUG_TEXT_YPOS = 0
DEBUG_TEXT_XPOS = 0

FONT_SIZE = 18

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

USE_MINI_MASK = True # mask the size of the ship (instead of the player view size)

# -------------------------------------------------------------------------------------------------
# Sensor

RAY_AMGLE_STEP = 20
RAY_BOX_SIZE   = 500
RAY_MAX_LEN    = ((RAY_BOX_SIZE/2) * math.sqrt(2)) # for now we are at the center of the ray mask box

# -------------------------------------------------------------------------------------------------
# SHIP dynamics

SLOW_DOWN_COEF = 1.2

SHIP_MASS = 1.0
SHIP_THRUST_MAX = 0.3 / SLOW_DOWN_COEF
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

SHIP_1_KEYS = {"left":pygame.K_w, "right":pygame.K_x, "thrust":pygame.K_v, "shoot":pygame.K_g, "shield":pygame.K_c}
SHIP_1_KEYS = {"left":pygame.K_LEFT, "right":pygame.K_RIGHT, "thrust":pygame.K_KP_PERIOD, "shoot":pygame.K_KP_ENTER, "shield":pygame.K_KP0}

# -------------------------------------------------------------------------------------------------
# Assets

MAP_1 = os.path.join("assets", "level1", "Mayhem_Level1_Map_256c.bmp")
MAP_2 = os.path.join("assets", "level2", "Mayhem_Level2_Map_256c.bmp")
MAP_3 = os.path.join("assets", "level3", "Mayhem_Level3_Map_256c.bmp")
MAP_4 = os.path.join("assets", "level4", "Mayhem_Level4_Map_256c.bmp")
MAP_5 = os.path.join("assets", "level5", "Mayhem_Level5_Map_256c.bmp")

SOUND_THURST  = os.path.join("assets", "default", "sfx_loop_thrust.wav")
SOUND_EXPLOD  = os.path.join("assets", "default", "sfx_boom.wav")
SOUND_BOUNCE  = os.path.join("assets", "default", "sfx_rebound.wav")
SOUND_SHOOT   = os.path.join("assets", "default", "sfx_shoot.wav")
SOUND_SHIELD  = os.path.join("assets", "default", "sfx_loop_shield.wav")

SHIP_1_PIC        = os.path.join("assets", "default", "ship1_256c.bmp")
SHIP_1_PIC_THRUST = os.path.join("assets", "default", "ship1_thrust_256c.bmp")
SHIP_1_PIC_SHIELD = os.path.join("assets", "default", "ship1_shield_256c.bmp")

# -------------------------------------------------------------------------------------------------

START_POSITIONS = [(430, 730), (473, 195), (647, 227), (645, 600), (647, 950), (510, 1070), (298, 1037), \
                   (273, 777), (275, 506), (70, 513), (89, 208), (434, 452), (289, 153)]

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

class Ship():

    def __init__(self, screen_width, screen_height, ship_number, xpos, ypos, ship_pic, ship_pic_thrust, ship_pic_shield, lives):

        margin_size = 0
        w_percent = 1.0
        h_percent = 1.0

        self.view_width = screen_width
        self.view_height = screen_height
        self.view_left = margin_size
        self.view_top = margin_size

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

        if 1:
            r_ind1 = randint(0, len(START_POSITIONS)-1) 
            self.xpos, self.ypos = START_POSITIONS[r_ind1]
            
            r_ind2 = r_ind1
            while r_ind1==r_ind2:

                r_ind2 = randint(0, len(START_POSITIONS)-1)
                self.xpos_dest, self.ypos_dest = START_POSITIONS[r_ind2]

            #print("Init=", (self.xpos, self.ypos))
            #print("Dest=", (self.xpos_dest, self.ypos_dest))
        else:
            self.xpos, self.ypos = START_POSITIONS[0]
            self.xpos_dest, self.ypos_dest = 400, 100

        #
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

        if env.play_sound:
            self.sound_thrust.stop()
            self.sound_shoot.stop()
            self.sound_shield.stop()
            self.sound_bounce.stop()

            if 1:
                self.sound_explod.play()

    def step(self, env, action=None):

        # we have an action => AI plays
        if action is not None:
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

        # no action, human plays or record play ?
        else:
            # record ?
            if not env.play_recorded:
        
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
                    print("%s seconds" % int(env.frames/env.max_fps))
                    sys.exit(0)

        # move based on L, R, T
        self.do_move(env)

    def do_move(self, env):

        if env.motion == "thrust":

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
                if env.play_sound:
                    self.sound_thrust.stop()

                if env.play_sound:
                    if not pygame.mixer.get_busy():
                        self.sound_shield.play(-1)
            else:
                self.shield = False
                if env.play_sound:
                    self.sound_shield.stop()

                # thrust
                if self.thrust_pressed:
                    self.image = self.ship_pic_thrust

                    #self.thrust += 0.1
                    #if self.thrust >= SHIP_THRUST_MAX:
                    self.thrust = SHIP_THRUST_MAX

                    if env.play_sound:
                        if not pygame.mixer.get_busy():
                            self.sound_thrust.play(-1)

                    self.landed = False

                else:
                    self.thrust = 0.0
                    if env.play_sound:
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
                        if env.play_sound:
                            if not pygame.mixer.get_busy():
                                self.sound_shoot.play()

                        self.add_shots()
            else:
                self.shoot = False
                if env.play_sound:
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
            if 1:
                self.is_landed(env)

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

        for plaform in env.platforms:
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
                    if env.play_sound:
                        self.sound_bounce.play()

                return True

        return False

    def do_test_collision(self, platforms):
        test_it = True

        for plaform in platforms:
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
        map_buffer.blit(self.image_rotated, (self.xpos + self.rot_xoffset, self.ypos + self.rot_yoffset))

    def collide_map(self, map_buffer, map_buffer_mask, platforms):

        # ship size mask
        if USE_MINI_MASK:
            mini_area = Rect(self.xpos, self.ypos, SHIP_SPRITE_SIZE, SHIP_SPRITE_SIZE)
            mini_subsurface = map_buffer.subsurface(mini_area)
            mini_subsurface.set_colorkey( (0, 0, 0) ) # used for the mask, black = background
            mini_mask = pygame.mask.from_surface(mini_subsurface)

            if self.do_test_collision(platforms):
                offset = (self.rot_xoffset, self.rot_yoffset) # pos of the ship

                if mini_mask.overlap(self.mask, offset): # https://stackoverflow.com/questions/55817422/collision-between-masks-in-pygame/55818093#55818093
                    self.explod = True

        # player view size mask
        else:
            if self.do_test_collision(platforms):
                offset = (self.xpos + self.rot_xoffset, self.ypos + self.rot_yoffset) # pos of the ship

                if map_buffer_mask.overlap(self.mask, offset): # https://stackoverflow.com/questions/55817422/collision-between-masks-in-pygame/55818093#55818093
                    self.explod = True

    def collide_ship(self, ships):
        for ship in ships:
            if self != ship:
                offset = ((ship.xpos - self.xpos), (ship.ypos - self.ypos))
                if self.mask.overlap(ship.mask, offset):
                    self.explod = True
                    ship.explod = True

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

    def ray_sensor(self, env, flipped_masks_map_buffer):
        # TODO use smaller map masks
        # TODO use only 0 to 90 degres ray mask quadran: https://github.com/Rabbid76/PyGameExamplesAndAnswers/blob/master/examples/minimal_examples/pygame_minimal_mask_intersect_surface_line_2.py

        # clipping translation for window coordinates
        rx = self.xpos - self.view_width/2
        ry = self.ypos - self.view_height/2

        dx = 0 ; dy = 0

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

            filpped_map_mask = flipped_masks_map_buffer[flip_x][flip_y]

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

#class MayhemEnv():
class MayhemEnv(gymnasium.Env):
    
    def __init__(self, game, level=1, max_fps=60, debug_print=1, play_sound=1, motion="gravity", sensor="", record_play="", play_recorded=""):
        #super.__init__(self, t, obj)
        
        self.myfont = pygame.font.SysFont('Arial', FONT_SIZE)

        self.play_sound = play_sound
        self.max_fps = max_fps

        # screen
        self.game = game
        self.game.window.fill((0, 0, 0))
        self.level = level
        #self.level = randint(1, 5)
        self.debug_print = debug_print

        self.motion = motion # thrust, gravity
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

        # left, right, thrust
        self.action_space = gymnasium.spaces.MultiBinary(3) # np.array([1,0,1]).astype(np.int8)

        # angle_vecx angle_vecy vx vy ax ay + 8 dist
        low  = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 
                         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                         -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                           ]).astype(np.float32)
        
        high = np.array([ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  
                          1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0,
                          1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0,
                          ]).astype(np.float32)

        self.observation_space = gymnasium.spaces.Box(low, high)
        
        self.done = False
        self.frame_pic = None

        self.ship_1 = Ship(self.game.screen_width, self.game.screen_height, 1, 430, 730, \
                               SHIP_1_PIC, SHIP_1_PIC_THRUST, SHIP_1_PIC_SHIELD, SHIP_MAX_LIVES)

        # per level data
        self.map = self.game.getv("map", current_level=self.level)
        self.map_buffer = self.game.getv("map_buffer", current_level=self.level)
        self.map_buffer_mask = self.game.getv("map_buffer_mask", current_level=self.level)
        self.flipped_masks_map_buffer = self.game.getv("flipped_masks_map_buffer", current_level=self.level)
        self.platforms = self.game.getv("platforms", current_level=self.level)

    def close(self):
        # TODO pygame cleanup
        pass

    def record_if_needed(self):
        if self.record_play:
            with open(self.record_play, "wb") as f:
                pickle.dump(self.played_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            time.sleep(0.1)
            print("Frames=", self.frames)
            print("%s seconds" % int(self.frames/self.max_fps))
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

    def practice_loop(self):

        # Game Main Loop
        while True:

            # clear screen
            self.game.window.fill((0,0,0))

            # pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.record_if_needed()
                    sys.exit(0)

                elif event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:
                        self.record_if_needed()
                        sys.exit(0)
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                    elif event.key == pygame.K_DELETE:
                        self.ship_1.explod = True

                    self.ship_key_down(event.key, self.ship_1, SHIP_1_KEYS)
                
                elif event.type == pygame.KEYUP:
                    self.ship_key_up(event.key, self.ship_1, SHIP_1_KEYS)
                    
            if not self.paused:            
                # per level data
                self.map = self.game.getv("map", current_level=self.level)
                self.map_buffer = self.game.getv("map_buffer", current_level=self.level)
                self.map_buffer_mask = self.game.getv("map_buffer_mask", current_level=self.level)
                self.flipped_masks_map_buffer = self.game.getv("flipped_masks_map_buffer", current_level=self.level)
                self.platforms = self.game.getv("platforms", current_level=self.level)

                #
                self.map_buffer.blit(self.map, (0, 0))

                self.ship_1.step(self, action=None)

                # collision
                self.ship_1.collide_map(self.map_buffer, self.map_buffer_mask, self.platforms)
                # TODO collide_ship

                self.ship_1.plot_shots(self.map_buffer)
                # TODO collide_shots

                # blit ship in the map
                self.ship_1.draw(self.map_buffer)

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
                self.game.window.blit(self.map_buffer, (self.ship_1.view_left, self.ship_1.view_top), sub_area1)

                # sensors
                if self.sensor == "ray":
                    self.ship_1.ray_sensor(self, self.flipped_masks_map_buffer)

                if self.ship_1.explod:
                    self.ship_1.reset(self)
                    self.nb_dead += 1
                    print("Dead=%s" % str(self.nb_dead))
                    #self.level = randint(1, 5)

                self.screen_print_info()

                pygame.display.flip()

                self.frames += 1

            self.clock.tick(self.max_fps) # https://python-forum.io/thread-16692.html


    def screen_print_info(self):
        DCOL = (255,255,0)

        if self.debug_print:
            ship_pos = self.myfont.render('Pos: %s %s' % (self.ship_1.xpos, self.ship_1.ypos), False, DCOL)
            self.game.window.blit(ship_pos, (DEBUG_TEXT_XPOS + 5, DEBUG_TEXT_YPOS + 5))

            ship_va = self.myfont.render('vx=%.2f, vy=%.2f, ax=%.2f, ay=%.2f' % (self.ship_1.vx,self.ship_1.vy, self.ship_1.ax, self.ship_1.ay), False, DCOL)
            self.game.window.blit(ship_va, (DEBUG_TEXT_XPOS + 5, DEBUG_TEXT_YPOS + 5 + (FONT_SIZE + 4)*1))

            ship_angle = self.myfont.render('Angle: %s' % (self.ship_1.angle,), False, DCOL)
            self.game.window.blit(ship_angle, (DEBUG_TEXT_XPOS + 5, DEBUG_TEXT_YPOS + 5 + (FONT_SIZE + 4)*2))

            dt = self.myfont.render('Frames: %s' % (self.frames,), False, DCOL)
            self.game.window.blit(dt, (DEBUG_TEXT_XPOS + 5, DEBUG_TEXT_YPOS + 5 + (FONT_SIZE + 4)*3))

            fps = self.myfont.render('FPS: %.2f' % self.clock.get_fps(), False, DCOL)
            self.game.window.blit(fps, (DEBUG_TEXT_XPOS + 5, DEBUG_TEXT_YPOS + 5 + (FONT_SIZE + 4)*4))

            try:
                #step_dist = self.myfont.render('Step dist: %s' % self.step_dist, False, DCOL)
                #self.game.window.blit(step_dist, (DEBUG_TEXT_XPOS + 5, 130)) 

                reward = self.myfont.render('Reward: %s' % self.reward, False, DCOL)
                self.game.window.blit(reward, (DEBUG_TEXT_XPOS + 5, 155))  

                #zones = self.myfont.render('Zones: %s' % str(len(self.ship_1.visited_zones)), False, DCOL)
                #self.game.window.blit(zones, (DEBUG_TEXT_XPOS + 5, 180))  

            except:
                pass 

            #ship_lives = self.myfont.render('Lives: %s' % (self.ship_1.lives,), False, (255, 255, 255))
            #self.game.window.blit(ship_lives, (DEBUG_TEXT_XPOS + 5, 105))

    def reset(self, seed=None):
        self.frames = 0
        self.done = False
        self.paused = False
        self.ship_1.reset(self)

        #new_state = [angle_vecx, angle_vecy, vx_norm, vy_norm, ax_norm, ay_norm]
        #new_state.extend(wall_distances)

        new_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        if self.sensor == "ray":
            wall_distances = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            new_state.extend(wall_distances)
        elif self.sensor == "pic":
            new_state.extend([0]*32*32)

        return np.array(new_state, dtype=np.float32), {}

    # AI training only
    def step(self, action, max_frame=2000):

        # pause ?
        if self.paused:
            return None, None, None, {}
            
        # needed for ray sensor
        self.game.window.fill((0,0,0))

        done = False

        old_xpos = self.ship_1.xposprecise
        old_ypos = self.ship_1.yposprecise

        # Update ship xposprecise, yposprecise, xpos, ypos, collided etc
        self.ship_1.step(self, action)

        # https://www.baeldung.com/cs/normalizing-inputs-artificial-neural-network
        # https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
        # min-max: (((x - min) / (max - min)) * (end - start)) + start (typically start=0, end=1)

        if self.sensor == "ray":
            wall_distances = self.ship_1.ray_sensor(self, self.flipped_masks_map_buffer)

            # NORMALIZE
            for i, dist in enumerate(wall_distances):
                #wall_distances[i] = dist / RAY_MAX_LEN # [0, 1]
                wall_distances[i] = ((dist / RAY_MAX_LEN)*2) - 1 # [-1, 1]

            #print(wall_distances)

        # angle_norm = ((self.ship_1.angle / (360. - SHIP_ANGLESTEP)) * 2) - 1
        # we cannot use the angle as is as input for the NN because of the
        # jump 360 => 0 degrees, we have to use a direction vector instead
        angle_vecx = - math.sin( math.radians(self.ship_1.angle) )
        angle_vecy =   math.cos( math.radians(self.ship_1.angle) )

        # vx range = [-5.5, +5.5] (more or less with default phisical values, for "standard playing")
        # vy range = [-6.5, +8.5] (more or less with default phisical values, for "standard playing")
        # ax range = [-0.16, +0.16] (more or less with default phisical values, for "standard playing")
        # vx range = [-0.12, +0.20] (more or less with default phisical values, for "standard playing")

        vx_min = -9.0  ; vx_max = 9.0
        vy_min = -9.0  ; vy_max = 10.0
        ax_min = -0.24 ; ax_max = 0.25
        ay_min = -0.19 ; ay_max = 0.31

        vx_norm = (((self.ship_1.vx - vx_min) / (vx_max - vx_min)) * 2) - 1
        vy_norm = (((self.ship_1.vy - vy_min) / (vy_max - vy_min)) * 2) - 1
        ax_norm = (((self.ship_1.ax - ax_min) / (ax_max - ax_min)) * 2) - 1
        ay_norm = (((self.ship_1.ay - ay_min) / (ay_max - ay_min)) * 2) - 1

        new_state = [angle_vecx, angle_vecy, vx_norm, vy_norm, ax_norm, ay_norm]

        if self.sensor == "ray":
            new_state.extend(wall_distances)

        elif self.sensor == "pic":
            if self.frame_pic is not None:
                new_state.extend( ((self.frame_pic/255)*2)-1 )
                #new_state.extend( self.frame_pic/255 )
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

        truncated = self.frames > max_frame

        done = self.ship_1.explod
        done |= truncated
        done |= collision

        #d_end = math.sqrt((self.ship_1.init_xpos - self.ship_1.xpos)**2 + (self.ship_1.init_ypos - self.ship_1.ypos)**2)
        #d_end = np.linalg.norm(np.array((self.ship_1.init_xpos, self.ship_1.init_ypos)) - np.array((self.ship_1.xpos, self.ship_1.ypos)))

        self.reward = 0.0

        # ship has not moved ?
        if 0:
            self.step_dist = np.linalg.norm(np.array((old_xpos, old_ypos)) - np.array((self.ship_1.xposprecise, self.ship_1.yposprecise)))

            if self.step_dist < 0.05:
                self.reward = -self.step_dist
            else:
                self.reward = self.step_dist        

            #print("d=", d)

        if 0:
            if done:
                # visited zone
                self.number_visited_zone = len(self.ship_1.visited_zones)
                self.reward = self.number_visited_zone * 4

                if self.number_visited_zone == 1:
                    self.reward = 0

                # if ship exploded => bad reward
                if collision or self.ship_1.explod:
                    self.reward -= 10

                #self.level = randint(1, 5)

        if done:
            dist_to_goal = np.linalg.norm(np.array((self.ship_1.xpos_dest, self.ship_1.ypos_dest)) - np.array((self.ship_1.xposprecise, self.ship_1.yposprecise)))
            #print("dist_to_goal", dist_to_goal)
            self.reward = 1200 - dist_to_goal
            self.reward = max(0, self.reward)
            self.reward /= 50.
            #print("self.reward", self.reward)
            #print()

            if collision or self.ship_1.explod:
                self.reward -= 10

            #self.level = randint(1, 5)

        self.frames += 1

        if self.sensor == "ray":

            # angle vx vy ax ay + wall dists
            obs = np.array(new_state, dtype=np.float32)

            #print(obs)
            #print("reward=", self.reward)
            return obs, self.reward, done, truncated, {}

        elif self.sensor == "pic":
            #print(new_state)
            new_state = np.reshape(new_state, 32*32 + 5)
            return new_state, self.reward, done, truncated, {}

    # AI training only
    def render(self, max_fps=60, collision_check=True):

        # pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.record_if_needed()
                sys.exit(0)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.record_if_needed()
                    sys.exit(0)
                #elif event.key == pygame.K_p:
                #    self.paused = not self.paused
                elif event.key == pygame.K_DELETE:
                    self.ship_1.explod = True

        if not self.paused:
            # per level data
            self.map = self.game.getv("map", current_level=self.level)
            self.map_buffer = self.game.getv("map_buffer", current_level=self.level)
            self.map_buffer_mask = self.game.getv("map_buffer_mask", current_level=self.level)
            self.flipped_masks_map_buffer = self.game.getv("flipped_masks_map_buffer", current_level=self.level)
            self.platforms = self.game.getv("platforms", current_level=self.level)

            self.map_buffer.blit(self.map, (0, 0))

            #self.ship_1.step(self, action=None)

            # collision (when false we use the sensor to detect a collision)
            if collision_check:
                try:
                    self.ship_1.collide_map(self.map_buffer, self.map_buffer_mask, self.platforms)
                # ValueError: subsurface rectangle outside surface area
                except ValueError:
                    self.ship_1.explod = True

            # TODO collide_ship

            #self.ship_1.plot_shots(self.map_buffer)
            # TODO collide_shots
                                        
            # blit ship in the map
            self.ship_1.draw(self.map_buffer)

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
            self.game.window.blit(self.map_buffer, (self.ship_1.view_left, self.ship_1.view_top), sub_area1)

            # debug on screen
            self.screen_print_info()

            pygame.display.flip()

            self.clock.tick(max_fps)

            #return self.map_buffer.subsurface(sub_area1)

# -------------------------------------------------------------------------------------------------

class GameWindow():

    def __init__(self, screen_width=400, screen_height=400):

        pygame.display.set_caption('Mayhem')

        self.screen_width = screen_width
        self.screen_height = screen_height

        self.window = pygame.display.set_mode((self.screen_width, self.screen_height), flags=pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE | pygame.SCALED)

        # Background
        self.map_1 = pygame.image.load(MAP_1).convert() # .convert_alpha()
        #self.map.set_colorkey( (0, 0, 0) ) # used for the mask, black = background
        #self.map_rect = self.map.get_rect()
        #self.map_mask = pygame.mask.from_surface(self.map)
        #self.mask_map_fx = pygame.mask.from_surface(pygame.transform.flip(self.map, True, False))
        #self.mask_map_fy = pygame.mask.from_surface(pygame.transform.flip(self.map, False, True))
        #self.mask_map_fx_fy = pygame.mask.from_surface(pygame.transform.flip(self.map, True, True))
        #self.flipped_masks = [[self.map_mask, self.mask_map_fy], [self.mask_map_fx, self.mask_map_fx_fy]]

        self.map_buffer_1 = self.map_1.copy() # pygame.Surface((self.map.get_width(), self.map.get_height()))

        self.map_buffer_1.set_colorkey( (0, 0, 0) )
        self.map_buffer_mask_1 = pygame.mask.from_surface(self.map_buffer_1)

        self.mask_map_buffer_fx_1 = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer_1, True, False))
        self.mask_map_buffer_fy_1 = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer_1, False, True))
        self.mask_map_buffer_fx_fy_1 = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer_1, True, True))
        self.flipped_masks_map_buffer_1 = [[self.map_buffer_mask_1, self.mask_map_buffer_fy_1], [self.mask_map_buffer_fx_1, self.mask_map_buffer_fx_fy_1]]

        # map2
        self.map_2 = pygame.image.load(MAP_2).convert() # .convert_alpha()
        self.map_buffer_2 = self.map_2.copy() # pygame.Surface((self.map.get_width(), self.map.get_height()))

        self.map_buffer_2.set_colorkey( (0, 0, 0) )
        self.map_buffer_mask_2 = pygame.mask.from_surface(self.map_buffer_2)

        self.mask_map_buffer_fx_2 = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer_2, True, False))
        self.mask_map_buffer_fy_2 = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer_2, False, True))
        self.mask_map_buffer_fx_fy_2 = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer_2, True, True))
        self.flipped_masks_map_buffer_2 = [[self.map_buffer_mask_2, self.mask_map_buffer_fy_2], [self.mask_map_buffer_fx_2, self.mask_map_buffer_fx_fy_2]]

        # map3
        self.map_3 = pygame.image.load(MAP_3).convert() # .convert_alpha()
        self.map_buffer_3 = self.map_3.copy() # pygame.Surface((self.map.get_width(), self.map.get_height()))

        self.map_buffer_3.set_colorkey( (0, 0, 0) )
        self.map_buffer_mask_3 = pygame.mask.from_surface(self.map_buffer_3)

        self.mask_map_buffer_fx_3 = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer_3, True, False))
        self.mask_map_buffer_fy_3 = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer_3, False, True))
        self.mask_map_buffer_fx_fy_3 = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer_3, True, True))
        self.flipped_masks_map_buffer_3 = [[self.map_buffer_mask_3, self.mask_map_buffer_fy_3], [self.mask_map_buffer_fx_3, self.mask_map_buffer_fx_fy_3]]

        # map4
        self.map_4 = pygame.image.load(MAP_4).convert() # .convert_alpha()
        self.map_buffer_4 = self.map_4.copy() # pygame.Surface((self.map.get_width(), self.map.get_height()))

        self.map_buffer_4.set_colorkey( (0, 0, 0) )
        self.map_buffer_mask_4 = pygame.mask.from_surface(self.map_buffer_4)

        self.mask_map_buffer_fx_4 = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer_4, True, False))
        self.mask_map_buffer_fy_4 = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer_4, False, True))
        self.mask_map_buffer_fx_fy_4 = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer_4, True, True))
        self.flipped_masks_map_buffer_4 = [[self.map_buffer_mask_4, self.mask_map_buffer_fy_4], [self.mask_map_buffer_fx_4, self.mask_map_buffer_fx_fy_4]]

        # map5
        self.map_5 = pygame.image.load(MAP_5).convert() # .convert_alpha()
        self.map_buffer_5 = self.map_5.copy() # pygame.Surface((self.map.get_width(), self.map.get_height()))

        self.map_buffer_5.set_colorkey( (0, 0, 0) )
        self.map_buffer_mask_5 = pygame.mask.from_surface(self.map_buffer_5)

        self.mask_map_buffer_fx_5 = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer_5, True, False))
        self.mask_map_buffer_fy_5 = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer_5, False, True))
        self.mask_map_buffer_fx_fy_5 = pygame.mask.from_surface(pygame.transform.flip(self.map_buffer_5, True, True))
        self.flipped_masks_map_buffer_5 = [[self.map_buffer_mask_5, self.mask_map_buffer_fy_5], [self.mask_map_buffer_fx_5, self.mask_map_buffer_fx_fy_5]]

        # platforms
        self.platforms_1 = [ ( 464, 513, 333 ),
                            ( 60, 127, 1045 ),
                            ( 428, 497, 531 ),
                            ( 504, 568, 985 ),
                            ( 178, 241, 875 ),
                            ( 8, 37, 187 ),
                            ( 302, 351, 271 ),
                            ( 434, 521, 835 ),
                            ( 499, 586, 1165 ),
                            ( 68, 145, 1181 ) ]

        self.platforms_2 = [ [ 201, 259, 175 ],
                            [ 21, 92, 1087 ],
                            [ 552, 615, 513 ],
                            [ 468, 525, 915 ],
                            [ 546, 599, 327 ],
                            [ 8, 37, 187 ],
                            [ 660, 697, 447 ],
                            [ 350, 435, 621 ],
                            [ 596, 697, 1141 ] ]

        self.platforms_3 = [ [ 14, 65, 111 ],
                            [ 38, 93, 1121 ],
                            [ 713, 760, 231 ],
                            [ 473, 540, 617 ],
                            [ 565, 616, 459 ],
                            [ 343, 398, 207 ],
                            [ 316, 385, 805 ],
                            [ 492, 548, 987 ],
                            [ 66, 145, 1180 ] ]

        self.platforms_4 = [ [ 19, 69, 111 ],
                            [ 32, 84, 1121 ],
                            [ 705, 755, 231],
                            [ 487, 547, 617 ],
                            [ 556, 607, 459 ],
                            [ 344, 393, 207 ],
                            [ 326, 377, 805 ],
                            [ 502, 554, 987 ],
                            [ 66, 145, 1180 ] ]

        self.platforms_5 = [ [ 504, 568, 985 ],
                            [ 464, 513, 333 ],
                            [ 428, 497, 531],
                            [ 178, 241, 875 ],
                            [ 8, 37, 187 ],
                            [ 302, 351, 271 ],
                            [ 434, 521, 835 ],
                            [ 434, 521, 835 ],
                            [ 60, 127, 1045 ],
                            [ 348, 377, 1089 ],
                            [ 499, 586, 1165 ],
                            [ 68, 145, 1181 ] ]

    def getv(self, name, current_level=1):
        return getattr(self, "%s_%s" % (name, str(current_level)))

# -------------------------------------------------------------------------------------------------

def init_pygame():
    # overwrite gym dsp
    os.environ["SDL_AUDIODRIVER"] = "pulseaudio"

    #pygame.mixer.pre_init(frequency=22050)
    pygame.init()
    #pygame.display.init()

    pygame.mouse.set_visible(False)
    pygame.font.init()
    pygame.mixer.init() # frequency=22050

    #pygame.event.set_blocked((MOUSEMOTION, MOUSEBUTTONUP, MOUSEBUTTONDOWN))

# -------------------------------------------------------------------------------------------------

def run():
    parser = argparse.ArgumentParser()

    parser.add_argument('-width', '--width', help='', type=int, action="store", default=500)
    parser.add_argument('-height', '--height', help='', type=int, action="store", default=500)
    parser.add_argument('-fps', '--fps', help='', type=int, action="store", default=60)
    parser.add_argument('-dp', '--debug_print', help='', action="store", default=True)

    parser.add_argument('-m', '--motion', help='How the ship moves', action="store", default='gravity', choices=("thrust", "gravity"))
    parser.add_argument('-r', '--record_play', help='', action="store", default="")
    parser.add_argument('-pr', '--play_recorded', help='', action="store", default="")
    parser.add_argument('-s', '--sensor', help='', action="store", default="", choices=("ray", ""))

    result = parser.parse_args()
    args = dict(result._get_kwargs())

    print("Args", args)

    # window
    game_window = GameWindow(args["width"], args["height"])

    env = MayhemEnv(game_window, level=1, max_fps=args["fps"], debug_print=args["debug_print"], play_sound=1, motion=args["motion"],
                    sensor=args["sensor"], record_play=args["record_play"], play_recorded=args["play_recorded"])

    env.practice_loop()
        
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    init_pygame()
    run()

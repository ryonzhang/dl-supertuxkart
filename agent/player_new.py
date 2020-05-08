import numpy as np
import math
import queue
from torchvision import transforms as T

from . import models

player0_location = [0,0]
player1_location =[0,0]
global_step = 0
ball_size = 2.5
kart_width = 0.77
kart_length = 1.345
previous_puck_location = [-0.47587478160858154, 0.3693891763687134, 0.5443593263626099]
red_teammate0_location = [-6.739621162414551, 0.30152183771133423, -56.0368537902832]
red_teammate1_location = [-0.009999435395002365, 0.30152183771133423, -53.05702209472656]
blue_teammate0_location = [-6.5378875732421875, 0.30127325654029846, 55.04726028442383]
blue_teammate1_location = [-0.04991918057203293, 0.3006347417831421, 52.13950729370117]
# all_players = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok',
#                        'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
# goal_line_l = [-10.510000228881836, 0.07000000029802322, gate_y]
# goal_line_r = [10.460000038146973, 0.07000000029802322, gate_y]

class HockeyPlayer:
    """
       Your ice hockey player. You may do whatever you want here. There are three rules:
        1. no calls to the pystk library (your code will not run on the tournament system if you do)
        2. There needs to be a deep network somewhere in the loop
        3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)

        Try to minimize library dependencies, nothing that does not install through pip on linux.
    """

    """
       You may request to play with a different kart.
       Call `python3 -c "import pystk; pystk.init(pystk.GraphicsConfig.ld()); print(pystk.list_karts())"` to see all values.
    """
    kart = ""

    def __init__(self, team_id=0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        
        self.team_id = team_id
        self.kart = 'wilber'  # we would like to maintain using this kart type.
        self.previous_location = [0.0, 0.0, 0.0]
        self.teammate_location = [0.0, 0.0, 0.0]
        self.reverse_stuck_count = 0
        self.reverse_step = 0
        self.previous_10_location = queue.Queue(maxsize=9)
        self.screen_width = 400
        self.screen_height = 300
        self.model = models.load_model()
        
    
     def act(self, image, player_info):         #used for eval
#    def act(self, image, player_info, ball):   #used for training
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        ##########################################################################
        
        global player0_location 
        global player1_location 
        global global_step
        
        img_tensor = T.ToTensor()(image)
        proj = np.array(player_info.camera.projection).T
        view = np.array(player_info.camera.view).T
        player_id = player_info.kart.player_id
 
 
 
        det = self.model.detect(img_tensor, min_score=-2)
        ball_info = self.previous_location
        for tup in det:
            if tup[0] == 0 and tup[1] > -1.2:
                ball_info = self._to_world([tup[2], tup[3]], proj, view, 0.3695)
                break
#        ball_info = ball     #used for training
        
        ball_x = ball_info[0]
        ball_y = ball_info[2]

    
        kart_x = player_info.kart.location[0]
        kart_y = player_info.kart.location[2]
         
            
        if global_step == 0:
            if self.team_id == 0:
                if player_id == 0:
                    self.teammate_location = red_teammate1_location
                elif player_id == 1:
                    self.teammate_location = red_teammate0_location
            elif self.team_id == 1:
                if player_id == 0:
                    self.teammate_location = blue_teammate1_location
                elif player_id == 1:
                    self.teammate_location = blue_teammate0_location
        else:
            if player_id == 0:
                    self.teammate_location = player1_location
            elif player_id == 1:
                    self.teammate_location = player0_location
            if self.team_id == 0:
                if player_id == 0 and (abs(kart_x - red_teammate0_location[0]) < 1e-2) and (abs(kart_y - red_teammate0_location[2]) < 1e-2):
                    global_step = 0
                elif player_id == 1 and (abs(kart_x - red_teammate1_location[0]) < 1e-2) and (abs(kart_y - red_teammate1_location[2]) < 1e-2):
                    global_step = 0
            elif self.team_id == 1:
                if player_id == 0 and (abs(kart_x - blue_teammate0_location[0]) < 1e-2) and (abs(kart_y - blue_teammate0_location[2]) < 1e-2):
                    global_step = 0
                elif player_id == 1 and (abs(kart_x - blue_teammate1_location[0]) < 1e-2) and (abs(kart_y - blue_teammate1_location[2]) < 1e-2):
                    global_step = 0
                
        
        tm_x = self.teammate_location[0]
        tm_y = self.teammate_location[1]
        velocity_x = player_info.kart.velocity[0]
        velocity_y = player_info.kart.velocity[2]
        front_x = player_info.kart.front[0]
        front_y = player_info.kart.front[2]

        gate_y = 64.5 if (self.team_id in (0, 2)) else -64.5
        kart_v = math.sqrt(velocity_x ** 2 + velocity_y ** 2)
        ball_kart_distance = self.ball_kart_dist(ball_x, ball_y, kart_x, kart_y)
        ball_kart_distance_tm = self.ball_kart_dist(ball_x, ball_y, tm_x, tm_y)
        
        
        # default value of each control output.
        nitro_value = False
        brake_value = False
        drift_value = False
        rescue_value = False
        acceleration_value = 1
        global_step += 1
        
        
        # player1 actions
        if player_id == 1:   
        
            # block for reverse if hit the wall or stuck somewhere                  
            if global_step == 0:
                q.queue.clear()
            
            sign_kart = 0
            if abs(kart_y) > 64:
                sign_kart = np.sign(kart_x) * np.sign(kart_y) 
        
            if abs(front_x) > 45:
                sign_kart = np.sign(ball_y - kart_y) * np.sign(kart_x)
            
            if abs(front_y) > 63:
                sign_kart = -np.sign(ball_x - kart_x) * np.sign(kart_y)
                
            if self.previous_10_location.full():
                self.previous_10_location.get()
                self.previous_10_location.put(player_info.kart.location, block=False)
            else:
                self.previous_10_location.put(player_info.kart.location, block=False)
            if self.reverse_stuck_count == 0 and self.previous_10_location.qsize() == 9:
                if (math.sqrt(math.pow(self.previous_10_location.get()[0] - player_info.kart.location[0], 2)
                              + math.pow(self.previous_10_location.get()[2] - player_info.kart.location[2],
                                         2)) < 0.15):
                    acceleration_value = 0
                    brake_value = True
                    self.reverse_stuck_count += 1
                    return {'acceleration': acceleration_value, 'brake': brake_value, 'drift': False,
                            'nitro': False, 'rescue': False, 'steer': sign_kart}

            elif 0 < self.reverse_stuck_count <= 10:
                acceleration_value = 0
                brake_value = True
                self.reverse_stuck_count += 1
                if self.reverse_stuck_count > 10:
                    self.reverse_stuck_count = 0
                return {'acceleration': acceleration_value, 'brake': brake_value, 'drift': False,
                        'nitro': False, 'rescue': False, 'steer': sign_kart}


            # ----chase the puck----#
            ball_kart_distance = self.ball_kart_dist(ball_x, ball_y, front_x, front_y)
            angle_kart = self.angle_cal(kart_x, kart_y, front_x, front_y)
            angle_ball = self.angle_cal(front_x, front_y, ball_x, ball_y)                  
            steer_value = self.chasing(angle_kart, angle_ball)
            steer_value = 25 * steer_value
            

            # ----set the speed when chasing puck----#
            if (kart_v < 10) and (ball_kart_distance > 15):
                    nitro_value = True

            acceleration_value = 1

            if (steer_value >= 10 or steer_value <= -10):
                drift_value = True
                nitro_value = False
                brake_value = True
                acceleration_value = 0.25

            # ----when approaching ball, slow down---#
            if (ball_kart_distance < 12):
                if kart_v > 12:
                    acceleration_value = 0
                    brake_value = True
                else:
                    acceleration_value = 1

            # ----Strategy for offense---#
            dis_threshold = 8
            is_facing_opp_gl = front_y > kart_y if self.team_id in (0, 2) else front_y < kart_y

            if (ball_kart_distance < dis_threshold and is_facing_opp_gl):
                if (kart_v > 20):
                    acceleration_value = 0
#                     brake_value = True
                else:
                    acceleration_value = 1
                    nitro_value = True

                offset = (ball_size+kart_width)/3.5
#                 offset = (ball_size)/3
                angle_goal = self.angle_cal(ball_x, ball_y, 0, gate_y)
                chase_y = -offset*np.sin(angle_goal) + ball_y
                chase_x = -offset*np.cos(angle_goal) + ball_x

                if ((chase_y < front_y) and (gate_y > 0)) or ((chase_y > front_y) and (gate_y < 0)):
                    chase_y = front_y
                    chase_x = 1e-5 + front_x

                angle_goal = self.angle_cal(front_x, front_y, chase_x, chase_y)
                steer_value = self.chasing(angle_kart, angle_goal)
#     #             print('step:%d, angle:%0.3f' %(self.global_step, angle_goal))
#     #             print('step:%d, steer:%0.3f' %(self.global_step, steer_value))
#                 if (abs(ball_x) - 45.5) < 0.2:
#                     steer_value = 10*steer_value
#                 else:
# #                     steer_value = 25*steer_value
#                     steer_value = 12*(self.sigmoid(129.0/abs(ball_y - gate_y)-0.5))
                
                if (abs(angle_goal) < np.pi/4):
    #                 brake = True
                    steer_value = 15*steer_value
                    if abs(angle_goal) < np.pi/50:
                        steer_value = 5*steer_value/30
                else:
                    steer_value = 12*steer_value
                
                
#                 steer_value = abs(ball_y-gate_y)/129*5*steer_value
                
#                 gap = (ball_y - front_y) if self.team_id in (0, 2) else (front_y - ball_y)
                
               
#                 if (gap < 0.1):
#                     brake = True
#                     acceleration = 0.1
                    
#                 previous_gap = gap    
                nitro = False
                
                
                





    #         if (ball_kart_distance < dis_threshold) and (ball_x > -10.45) or (ball_x < 10.45) and abs(ball_y - gate_y)<1:
    #             angle_goal = self.angle_cal(ball_x, ball_y, 0, gate_y)

    #             offset = (ball_size+kart_width)/3
    #             sf = 1
    #             angle_goal = self.angle_cal(ball_x, ball_y, 0, gate_y)

    #             chase_y = -offset*np.sin(angle_goal) + ball_y
    #             chase_x = -sf*offset*np.cos(angle_goal) + ball_x

    #             if ((chase_y < front_y) and (gate_y > 0)) or ((chase_y > front_y) and (gate_y < 0)):
    #                 chase_y = front_y
    #                 chase_x = 1e-3 + front_x

    #             angle_goal = self.angle_cal(front_x, front_y, chase_x, chase_y)
    #             steer_value = self.chasing(angle_kart, angle_goal, kart_x, kart_y, 0, gate_y, front_x, front_y)
    #             steer_value = 50*steer_value

             # ----Strategy for defense---#
    #         dis_threshold2 = 10
            is_facing_own_gl = front_y < kart_y if self.team_id in (0, 2) else front_y > kart_y
            if (is_facing_own_gl):
    #             if (kart_v > 20):
    #                 acceleration_value = 0
    #                 brake = True
    #             else:
    #                 acceleration_value = 0.5
                if (kart_v > 18):
                    acceleration_value = 0.01
                    brake_value = True
                else:
                    acceleration_value = 1
                    nitro_value = False
                    brake_value = False

                ball_in_front = front_y < ball_y if self.team_id in (0, 2) else front_y > ball_y

                if (is_facing_own_gl) and (ball_in_front):
                    
                    if (ball_x < -10.45) or (ball_x > 10.45):

            #             angle_goal_n = self.angle_cal(ball_x, ball_y, -10.45, gate_y)
            #             angle_goal_p = self.angle_cal(ball_x, ball_y, 10.45, gate_y)
                        offset = (ball_size+kart_width)/3
                        if ball_x < 0:
                            angle_goal = self.angle_cal(ball_x, ball_y, -15, gate_y)
                        elif ball_y > 0:
                            angle_goal = self.angle_cal(ball_x, ball_y, 15, gate_y)
                        else:
                            angle_goal = self.angle_cal(ball_x, ball_y, -15, gate_y)
                        if gate_y > 0:
                            chase_y = -offset*np.sin(angle_goal) + ball_y
                            chase_x = -offset*np.cos(angle_goal) + ball_x
                        else:
                            chase_y = -offset*np.sin(angle_goal) + ball_y
                            chase_x = -offset*np.cos(angle_goal) + ball_x

                        angle_goal = self.angle_cal(front_x, front_y, chase_x, chase_y)

                        steer_value = self.chasing(angle_kart, angle_goal)
                        steer_value = 15*steer_value
                        drift_value = False
                    else:
                        if abs(ball_y - kart_y) > 4:
                            offset = (ball_size+kart_width)
                            chase_y = ball_y
                            chase_x = offset*np.sign(ball_x-kart_x)*np.sign(gate_y) + ball_x
                            angle_goal = self.angle_cal(front_x, front_y, chase_x, chase_y)
                            steer_value = self.chasing(angle_kart, angle_goal)
                            steer_value = 15*steer_value
                            drift_value = False
                            
                            # ----when approaching ball, slow down---#
                            if (ball_kart_distance < 12):
                                if kart_v > 7.5:
                                    acceleration_value = 0
                                    brake_value = True
                                else:
                                    acceleration_value = 0.25
                                    brake_value = False
    
                        else:
                            chase_y = ball_y+0.5*ball_size*np.sign(-gate_y)
                            chase_x = 45.5*np.sign(ball_x-kart_x)*np.sign(gate_y)
                            angle_goal = self.angle_cal(front_x, front_y, chase_x, chase_y)
                            steer_value = self.chasing(angle_kart, angle_goal)
                            steer_value = 15*steer_value
                            drift_value = True 
                            if (kart_v > 15):
                                acceleration_value = 0.01
                                brake_value = True
                            else:
                                acceleration_value = 1
                                nitro_value = False
                                brake_value = True
                   
                        if abs(kart_y + np.sign(gate_y)*63.5) < 0.2:
                            steer_value = 0
                    
    
        if player_id == 0:
            # block for reverse if hit the wall or stuck somewhere                  
            if global_step == 0:
                q.queue.clear()
            
            sign_kart = 0
            if abs(kart_y) > 64:
                sign_kart = np.sign(kart_x) * np.sign(kart_y) 
        
            if abs(front_x) > 45:
                sign_kart = np.sign(ball_y - kart_y) * np.sign(kart_x)
            
            if abs(front_y) > 63:
                sign_kart = -np.sign(ball_x - kart_x) * np.sign(kart_y)
                
            if self.previous_10_location.full():
                self.previous_10_location.get()
                self.previous_10_location.put(player_info.kart.location, block=False)
            else:
                self.previous_10_location.put(player_info.kart.location, block=False)
            if self.reverse_stuck_count == 0 and self.previous_10_location.qsize() == 9:
                if (math.sqrt(math.pow(self.previous_10_location.get()[0] - player_info.kart.location[0], 2)
                              + math.pow(self.previous_10_location.get()[2] - player_info.kart.location[2],
                                         2)) < 0.15):
                    acceleration_value = 0
                    brake_value = True
                    self.reverse_stuck_count += 1
                    return {'acceleration': acceleration_value, 'brake': brake_value, 'drift': False,
                            'nitro': False, 'rescue': False, 'steer': sign_kart}

            elif 0 < self.reverse_stuck_count <= 10:
                acceleration_value = 0
                brake_value = True
                self.reverse_stuck_count += 1
                if self.reverse_stuck_count > 10:
                    self.reverse_stuck_count = 0
                return {'acceleration': acceleration_value, 'brake': brake_value, 'drift': False,
                        'nitro': False, 'rescue': False, 'steer': sign_kart}


            # ----chase the puck----#
            ball_kart_distance = self.ball_kart_dist(ball_x, ball_y, front_x, front_y)
            angle_kart = self.angle_cal(kart_x, kart_y, front_x, front_y)
            angle_ball = self.angle_cal(front_x, front_y, ball_x, ball_y)                  
            steer_value = self.chasing(angle_kart, angle_ball)
            steer_value = 25 * steer_value
            

            # ----set the speed when chasing puck----#
            if (kart_v < 10) and (ball_kart_distance > 15):
                    nitro_value = True

            acceleration_value = 1

            if (steer_value >= 10 or steer_value <= -10):
                drift_value = True
                nitro_value = False
                brake_value = True
                acceleration_value = 0.25

            # ----when approaching ball, slow down---#
            if (ball_kart_distance < 12):
                if kart_v > 12:
                    acceleration_value = 0
                    brake_value = True
                else:
                    acceleration_value = 1

            # ----Strategy for offense---#
            dis_threshold = 8
            is_facing_opp_gl = front_y > kart_y if self.team_id in (0, 2) else front_y < kart_y

            if (ball_kart_distance < dis_threshold and is_facing_opp_gl):
                if (kart_v > 20):
                    acceleration_value = 0
#                     brake_value = True
                else:
                    acceleration_value = 1
                    nitro_value = True

                offset = (ball_size+kart_width)/3.5
#                 offset = (ball_size)/3
                angle_goal = self.angle_cal(ball_x, ball_y, 0, gate_y)
                chase_y = -offset*np.sin(angle_goal) + ball_y
                chase_x = -offset*np.cos(angle_goal) + ball_x

                if ((chase_y < front_y) and (gate_y > 0)) or ((chase_y > front_y) and (gate_y < 0)):
                    chase_y = front_y
                    chase_x = 1e-5 + front_x

                angle_goal = self.angle_cal(front_x, front_y, chase_x, chase_y)
                steer_value = self.chasing(angle_kart, angle_goal)
#     #             print('step:%d, angle:%0.3f' %(self.global_step, angle_goal))
#     #             print('step:%d, steer:%0.3f' %(self.global_step, steer_value))
#                 if (abs(ball_x) - 45.5) < 0.2:
#                     steer_value = 10*steer_value
#                 else:
# #                     steer_value = 25*steer_value
#                     steer_value = 12*(self.sigmoid(129.0/abs(ball_y - gate_y)-0.5))
                
                if (abs(angle_goal) < np.pi/4):
    #                 brake = True
                    steer_value = 15*steer_value
                    if abs(angle_goal) < np.pi/50:
                        steer_value = 5*steer_value/30
                else:
                    steer_value = 12*steer_value
                
                
#                 steer_value = abs(ball_y-gate_y)/129*5*steer_value
                
#                 gap = (ball_y - front_y) if self.team_id in (0, 2) else (front_y - ball_y)
                
               
#                 if (gap < 0.1):
#                     brake = True
#                     acceleration = 0.1
                    
#                 previous_gap = gap    
                nitro = False
                
                
                





    #         if (ball_kart_distance < dis_threshold) and (ball_x > -10.45) or (ball_x < 10.45) and abs(ball_y - gate_y)<1:
    #             angle_goal = self.angle_cal(ball_x, ball_y, 0, gate_y)

    #             offset = (ball_size+kart_width)/3
    #             sf = 1
    #             angle_goal = self.angle_cal(ball_x, ball_y, 0, gate_y)

    #             chase_y = -offset*np.sin(angle_goal) + ball_y
    #             chase_x = -sf*offset*np.cos(angle_goal) + ball_x

    #             if ((chase_y < front_y) and (gate_y > 0)) or ((chase_y > front_y) and (gate_y < 0)):
    #                 chase_y = front_y
    #                 chase_x = 1e-3 + front_x

    #             angle_goal = self.angle_cal(front_x, front_y, chase_x, chase_y)
    #             steer_value = self.chasing(angle_kart, angle_goal, kart_x, kart_y, 0, gate_y, front_x, front_y)
    #             steer_value = 50*steer_value

             # ----Strategy for defense---#
    #         dis_threshold2 = 10
            is_facing_own_gl = front_y < kart_y if self.team_id in (0, 2) else front_y > kart_y
            if (is_facing_own_gl):
    #             if (kart_v > 20):
    #                 acceleration_value = 0
    #                 brake = True
    #             else:
    #                 acceleration_value = 0.5
                if (kart_v > 18):
                    acceleration_value = 0.01
                    brake_value = True
                else:
                    acceleration_value = 1
                    nitro_value = False
                    brake_value = False

                ball_in_front = front_y < ball_y if self.team_id in (0, 2) else front_y > ball_y

                if (is_facing_own_gl) and (ball_in_front):
                    
                    if (ball_x < -10.45) or (ball_x > 10.45):

            #             angle_goal_n = self.angle_cal(ball_x, ball_y, -10.45, gate_y)
            #             angle_goal_p = self.angle_cal(ball_x, ball_y, 10.45, gate_y)
                        offset = (ball_size+kart_width)/3
                        if ball_x < 0:
                            angle_goal = self.angle_cal(ball_x, ball_y, -15, gate_y)
                        elif ball_y > 0:
                            angle_goal = self.angle_cal(ball_x, ball_y, 15, gate_y)
                        else:
                            angle_goal = self.angle_cal(ball_x, ball_y, -15, gate_y)
                        if gate_y > 0:
                            chase_y = -offset*np.sin(angle_goal) + ball_y
                            chase_x = -offset*np.cos(angle_goal) + ball_x
                        else:
                            chase_y = -offset*np.sin(angle_goal) + ball_y
                            chase_x = -offset*np.cos(angle_goal) + ball_x

                        angle_goal = self.angle_cal(front_x, front_y, chase_x, chase_y)

                        steer_value = self.chasing(angle_kart, angle_goal)
                        steer_value = 15*steer_value
                        drift_value = False
                    else:
                        if abs(ball_y - kart_y) > 4:
                            offset = (ball_size+kart_width)
                            chase_y = ball_y
                            chase_x = offset*np.sign(ball_x-kart_x)*np.sign(gate_y) + ball_x
                            angle_goal = self.angle_cal(front_x, front_y, chase_x, chase_y)
                            steer_value = self.chasing(angle_kart, angle_goal)
                            steer_value = 15*steer_value
                            drift_value = False
                            
                            # ----when approaching ball, slow down---#
                            if (ball_kart_distance < 12):
                                if kart_v > 7.5:
                                    acceleration_value = 0
                                    brake_value = True
                                else:
                                    acceleration_value = 0.25
                                    brake_value = False
    
                        else:
                            chase_y = ball_y+0.5*ball_size*np.sign(-gate_y)
                            chase_x = 45.5*np.sign(ball_x-kart_x)*np.sign(gate_y)
                            angle_goal = self.angle_cal(front_x, front_y, chase_x, chase_y)
                            steer_value = self.chasing(angle_kart, angle_goal)
                            steer_value = 15*steer_value
                            drift_value = True 
                            if (kart_v > 15):
                                acceleration_value = 0.01
                                brake_value = True
                            else:
                                acceleration_value = 1
                                nitro_value = False
                                brake_value = True
                   
                        if abs(kart_y + np.sign(gate_y)*63.5) < 0.2:
                            steer_value = 0
                    
        
        
        
        
        
        
#         if player_id == 0:
#             # block for reverse if hit the wall or stuck somewhere                  
#             if global_step == 0:
#                 q.queue.clear()
            
#             sign_kart = 0
#             if abs(kart_y) > 64:
#                 sign_kart = np.sign(kart_x) * np.sign(kart_y) 
        
#             if abs(front_x) > 45:
#                 sign_kart = np.sign(ball_y - kart_y) * np.sign(kart_x)
            
#             if abs(front_y) > 63:
#                 sign_kart = -np.sign(ball_x - kart_x) * np.sign(kart_y)
                
#             if self.previous_10_location.full():
#                 self.previous_10_location.get()
#                 self.previous_10_location.put(player_info.kart.location, block=False)
#             else:
#                 self.previous_10_location.put(player_info.kart.location, block=False)
#             if self.reverse_stuck_count == 0 and self.previous_10_location.qsize() == 9:
#                 if (math.sqrt(math.pow(self.previous_10_location.get()[0] - player_info.kart.location[0], 2)
#                               + math.pow(self.previous_10_location.get()[2] - player_info.kart.location[2],
#                                          2)) < 0.15) and (abs(front_x) > 20 or abs(front_y) > 63):
#                     acceleration_value = 0
#                     brake_value = True
#                     self.reverse_stuck_count += 1
#                     return {'acceleration': acceleration_value, 'brake': brake_value, 'drift': False,
#                             'nitro': False, 'rescue': False, 'steer': sign_kart}

#             elif 0 < self.reverse_stuck_count <= 10:
#                 acceleration_value = 0
#                 brake_value = True
#                 self.reverse_stuck_count += 1
#                 if self.reverse_stuck_count > 10:
#                     self.reverse_stuck_count = 0
#                 return {'acceleration': acceleration_value, 'brake': brake_value, 'drift': False,
#                         'nitro': False, 'rescue': False, 'steer': sign_kart}
            
            
            
#             if ball_kart_distance_tm > 6 and (abs(ball_y + gate_y) < 30):
#                 # ----chase the puck----#
#                 ball_kart_distance = self.ball_kart_dist(ball_x, ball_y, front_x, front_y)
#                 angle_kart = self.angle_cal(kart_x, kart_y, front_x, front_y)
#                 angle_ball = self.angle_cal(front_x, front_y, ball_x, ball_y)
#                 steer_value = self.chasing(angle_kart, angle_ball)
#                 steer_value = 25 * steer_value

#                 # ----set the speed when chasing puck----#
#                 if (kart_v < 10) and (ball_kart_distance > 15):
#                         nitro_value = True

#                 acceleration_value = 1

#                 if (steer_value >= 10 or steer_value <= -10):
#                     drift_value = True
#                     nitro_value = False
#                     brake_value = True
#                     acceleration_value = 0.25


#                 # ----when approaching ball, slow down---#
#                 if (ball_kart_distance < 12):
#                     if kart_v > 7.5:
#                         acceleration_value = 0
#                         brake_value = True
#                     else:
#                         acceleration_value = 1
#                         brake_value = False
                
                
#                 # ----Strategy for offense---#
#                 dis_threshold = 8
#                 is_facing_opp_gl = front_y > kart_y if self.team_id in (0, 2) else front_y < kart_y

#                 if (ball_kart_distance < dis_threshold and is_facing_opp_gl):
#                     if (kart_v > 20):
#                         acceleration_value = 0
#     #                     brake_value = True
#                     else:
#                         acceleration_value = 1
#                         nitro_value = True

#                     offset = (ball_size+kart_width)/3
#                     angle_goal = self.angle_cal(ball_x, ball_y, 0, gate_y)
#                     chase_y = -offset*np.sin(angle_goal) + ball_y
#                     chase_x = -offset*np.cos(angle_goal) + ball_x

#                     if ((chase_y < front_y) and (gate_y > 0)) or ((chase_y > front_y) and (gate_y < 0)):
#                         chase_y = front_y
#                         chase_x = 1e-3 + front_x

#                     angle_goal = self.angle_cal(front_x, front_y, chase_x, chase_y)
#                     steer_value = self.chasing(angle_kart, angle_goal)
#         #             print('step:%d, angle:%0.3f' %(self.global_step, angle_goal))
#         #             print('step:%d, steer:%0.3f' %(self.global_step, steer_value))
                 
#                     if ((abs(ball_x) - 45.5) < 0.3):
#                         steer_value = 10*steer_value
            
#                     if (abs(angle_goal) < np.pi/4):
#         #                 brake = True
#                         steer_value = 30*steer_value
#                         if abs(angle_goal) < np.pi/50:
#                             steer_value = 5*steer_value/30
#                     else:    
#                         steer_value = 12*steer_value
#                     nitro = False

             
#                  # ----Strategy for defense---#
#                 is_facing_own_gl = front_y < kart_y if self.team_id in (0, 2) else front_y > kart_y
#                 if (is_facing_own_gl):
#         #             if (kart_v > 20):
#         #                 acceleration_value = 0
#         #                 brake = True
#         #             else:
#         #                 acceleration_value = 0.5
#                     if (kart_v > 18):
#                         acceleration_value = 0.01
#                         brake_value = True
#                     else:
#                         acceleration_value = 1
#                         nitro_value = False
#                         brake_value = False

#                     ball_in_front = front_y < ball_y if self.team_id in (0, 2) else front_y > ball_y

#                     if (is_facing_own_gl) and (ball_in_front):
#         #                 if (ball_x > -10.45) or (ball_x < 10.45):

#                 #             angle_goal_n = self.angle_cal(ball_x, ball_y, -10.45, gate_y)
#                 #             angle_goal_p = self.angle_cal(ball_x, ball_y, 10.45, gate_y)
#                             offset = (ball_size+kart_width)/3
#                             if ball_x < 0:
#                                 angle_goal = self.angle_cal(ball_x, ball_y, -15, gate_y)
#                             elif ball_y > 0:
#                                 angle_goal = self.angle_cal(ball_x, ball_y, 15, gate_y)
#                             else:
#                                 angle_goal = self.angle_cal(ball_x, ball_y, -15, gate_y)

#                             if gate_y > 0:
#                                 chase_y = -offset*np.sin(angle_goal) + ball_y
#                                 chase_x = -offset*np.cos(angle_goal) + ball_x
#                             else:
#                                 chase_y = -offset*np.sin(angle_goal) + ball_y
#                                 chase_x = -offset*np.cos(angle_goal) + ball_x

#                             angle_goal = self.angle_cal(front_x, front_y, chase_x, chase_y)

#                             steer_value = self.chasing(angle_kart, angle_goal)
#                             steer_value = 15*steer_value
#                             drift_value = False
                
#             else:
#                 chase_x = 0
#                 chase_y = -np.sign(gate_y)*63.0

#                 # ----chase the goal line----#
#                 ball_kart_distance = self.ball_kart_dist(chase_x, chase_y, front_x, front_y)
#                 angle_kart = self.angle_cal(kart_x, kart_y, front_x, front_y)
#                 angle_ball = self.angle_cal(front_x, front_y, chase_x, chase_y)
#                 steer_value = self.chasing(angle_kart, angle_ball)
#                 steer_value = 25 * steer_value

#                 # ----set the speed when chasing goal line----#
#                 if (kart_v < 10) and (ball_kart_distance > 15):
#                         nitro_value = True

#                 acceleration_value = 1

#                 if (steer_value >= 10 or steer_value <= -10):
#                     drift_value = True
#                     nitro_value = False
#                     brake_value = True
#                     acceleration_value = 0.25


#                 # ----when approaching goal line, slow down---#
#                 if (ball_kart_distance < 12):
#                     if kart_v > 0:
#                         acceleration_value = 0
#                         brake_value = True
#                     else:
#                         acceleration_value = 0
#                         brake_value = False
                
#                 is_facing_opp_gl = front_y > kart_y if self.team_id in (0, 2) else front_y < kart_y
                
#                 if (ball_kart_distance < 5):
#                     if kart_v > 0:
#                         acceleration_value = 0
#                         brake_value = True
#                     else:
#                         acceleration_value = 0
#                         brake_value = False
                
#                 if is_facing_opp_gl:
#                     acceleration_value = 0
#                     brake_value = False
                
                
#        self.previous_location = [kart_x, kart_y]
        self.previous_kart_v = kart_v
        self.previous_location = ball_info
        
        if player_id == 0:
            player0_location = [kart_x, kart_y]
        else:
            player1_location = [kart_x, kart_y]
        
                
        action = {'acceleration': acceleration_value, 'brake': brake_value, 'drift': drift_value, 'nitro': nitro_value,
                  'rescue': False, 'steer': steer_value}


        return action
    
    
    
    def ball_kart_dist(self, ball_x, ball_y, kart_x, kart_y):
        x_delta = ball_x - kart_x
        y_delta = ball_y - kart_y
        return math.sqrt(x_delta ** 2 + y_delta ** 2)


    def angle_cal(self, x_ori, y_ori, x_des, y_des):
        x_delta = x_des - x_ori
        y_delta = y_des - y_ori
        angle_temp = np.arctan(y_delta / (x_delta + 0.00001))
        
        if x_delta >= 0:
            return angle_temp
        elif y_delta >= 0:
            return np.pi + angle_temp
        else:
            return -np.pi + angle_temp
    
    
    def chasing(self, angle_kart, angle_object_to_kart):
        
        if angle_kart >= 0:
            if angle_object_to_kart < angle_kart and angle_kart - np.pi < angle_object_to_kart:
                steer_value = (angle_kart-angle_object_to_kart)/np.pi
            elif angle_object_to_kart > angle_kart:
                steer_value = -(-angle_kart+angle_object_to_kart)/np.pi
            else:
                steer_value = -(2*np.pi+angle_object_to_kart-angle_kart)/np.pi
        else:
            if angle_object_to_kart > angle_kart and angle_kart + np.pi > angle_object_to_kart:
                steer_value = (angle_kart-angle_object_to_kart)/np.pi
            elif angle_object_to_kart < angle_kart:
                steer_value = -(-angle_kart+angle_object_to_kart)/np.pi
            else:
                steer_value = -(-2*np.pi+angle_object_to_kart-angle_kart)/np.pi

        return steer_value

    
    def _to_world(self, aim_point_image, proj, view, height=0):
        x, y, W, H = *aim_point_image, self.screen_width, self.screen_height
        pv_inv = np.linalg.pinv(proj @ view)
        xy, d = pv_inv.dot([float(x) / (W / 2) - 1, 1 - float(y) / (H / 2), 0, 1]), pv_inv[:, 2]
        x0, x1 = xy[:-1] / xy[-1], (xy + d)[:-1] / (xy + d)[-1]
        t = (height - x0[1]) / (x1[1] - x0[1])
        return t * x1 + (1 - t) * x0

    
    def _to_image(self, x, proj, view):
        W, H = self.screen_width, self.screen_height
        p = proj @ view @ np.array(list(x) + [1])
        return np.array([W / 2 * (p[0] / p[-1] + 1), H / 2 * (1 - p[1] / p[-1])])
                                      
                                      
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

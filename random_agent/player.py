import numpy as np
import math


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
    
    
    def __init__(self, player_id = 0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        all_players = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
        self.kart = all_players[np.random.choice(len(all_players))]
        self.previous_action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        self.previous_location = [0.0,0.0,0.0]
        self.previous_reverse = False
        #self.threshold = 2.25
        
    
    def act(self, socceri, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        #-----accelerate if distance between the ball and kart is far----#
        nitro_value = False
        brake_value = False
        acceleration_value = 1
        kart_x =player_info.kart.location[0]
        kart_y =player_info.kart.location[2]
        velocity_x = player_info.kart.velocity[0]
        velocity_y = player_info.kart.velocity[2]
        front_x = player_info.kart.front[0]
        front_y = player_info.kart.front[2]
        ball_x =socceri.ball.location[0]
        ball_y =socceri.ball.location[2]
        
        #calculate the distance from ball to kart
        #x_delta = ball_x-kart_x
        #y_delta = ball_y-kart_y
        x_delta = ball_x-front_x
        y_delta = ball_y-front_y
        ball_kart_distance = math.sqrt(x_delta**2 + y_delta**2)
        x_direction = front_x - kart_x
        y_direction = front_y - kart_y
        #angle_ball_temp = np.arctan(ball_y / (ball_x+0.00001))
        angle_ball = np.arctan(y_delta / (x_delta+0.00001))
        angle_kart_temp = np.arctan(y_direction / (x_direction+0.0001))
        
        #----set the speed----#
        kart_v = math.sqrt(velocity_x**2 + velocity_y**2)
        if(kart_v<10 and ball_kart_distance>7):
            if(player_info.kart.id % 2 ==0):
                acceleration_value = 1
                nitro_value = True
            else:
                acceleration_value = 0.75
        elif(ball_kart_distance>20):
            acceleration_value = 1
            nitro_value = True
        else:
            acceleration_value = 0.5
            
        #-----set the steering-----#
        
        if(x_direction>=0 and y_direction>=0):
            angle_kart = angle_kart_temp
        elif(x_direction<0 and y_direction>=0):
            angle_kart = np.pi + angle_kart_temp
        elif(x_direction<0 and y_direction<0):
            angle_kart = -np.pi + angle_kart_temp
        elif(x_direction>=0 and y_direction<0):
            angle_kart = angle_kart_temp
        #print("angle of kart:",angle_kart)
        
        delta_angle = - angle_ball + angle_kart
        
        if(x_delta<=0 and y_delta>0):
            if(delta_angle>np.pi):
                steer_value = np.pi-(2*np.pi-(delta_angle))
            else:
                steer_value = -(np.pi-abs(delta_angle))
            #print("just steer_value_t C:",steer_value)
        elif(x_delta>0 and y_delta>0):
            if(delta_angle<-np.pi):
                steer_value = 2*np.pi + delta_angle
            else:
                steer_value = delta_angle
            #print("just steer_value_t B:",steer_value)
        elif(x_delta>0 and y_delta<=0):
            if(delta_angle>np.pi):
                steer_value = -(2*np.pi - delta_angle)
            else:
                steer_value = delta_angle
            #print("just steer_value_t A:",steer_value)
        elif(x_delta<=0 and y_delta<=0):
            if(delta_angle<-np.pi):
                steer_value = -(np.pi-(2*np.pi+(delta_angle)))
            else:
                steer_value = -(np.pi-abs(delta_angle))
            #print("just steer_value_t C:",steer_value)
  
        
        steer_value = 8.5*steer_value
        if(steer_value>0.5):
            nitro_value = False
            
        #----when approaching ball, slow down---#
        if(ball_kart_distance<7 and ball_kart_distance>3):
            brake_value = True
            acceleration_value = 0.2
            #steer_value = -1
        #when touching the ball, shoot for the gate or reverse for angle#
        
        if(ball_kart_distance<2.25):
            #self.threshold = 2.25
            if(player_info.kart.id in (0,2)):
                gate_y = 64.5
            else:
                gate_y = -64.5
            #if(kart_x<0):
            if(kart_x<0 and angle_kart > 0 and angle_kart <=np.pi/2):
                acceleration_value = 0.75
            elif(kart_x<0 and angle_kart> np.pi/2 and angle_kart <np.pi):
                steer_value = -0.5*kart_x/abs(gate_y - kart_y)
                acceleration_value = 0.5
            #if(kart_x>=0):
            elif(kart_x>=0 and angle_kart > 0 and angle_kart <=np.pi/2):
                steer_value = -0.5*kart_x/abs(gate_y - kart_y)
                acceleration_value = 0.5
            elif(kart_x>=0 and angle_kart >np.pi/2 and angle_kart <np.pi):
                acceleration_value = 0.75
            else:
                brake_value = True
                acceleration_value = 0
                steer_value = 1
                #self.threshold = 10
            
        moving_distance = np.array(self.previous_location) - np.array(player_info.kart.location)
        #print(moving_distance, kart_v)
        
        if(moving_distance[0]**2+moving_distance[2]**2<0.0001):
            #print("hi i am moving back:")
            brake_value = True
            acceleration_value = 0
            steer_value = -1
            self.previous_reverse = True
        else:
            self.previous_reverse = False
        if(self.previous_reverse == True):
            acceleration_value = 0
            brake_value = True
        self.previous_location = player_info.kart.location

        action = {'acceleration': acceleration_value, 'brake': brake_value, 'drift': False, 'nitro': nitro_value, 'rescue': False, 'steer': steer_value}

        return action


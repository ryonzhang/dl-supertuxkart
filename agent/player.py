# import numpy as np
# import math
# import queue
# from torchvision import transforms as T
#
# from . import models
#
#
# class HockeyPlayer:
#     """
#        Your ice hockey player. You may do whatever you want here. There are three rules:
#         1. no calls to the pystk library (your code will not run on the tournament system if you do)
#         2. There needs to be a deep network somewhere in the loop
#         3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)
#
#         Try to minimize library dependencies, nothing that does not install through pip on linux.
#     """
#
#     """
#        You may request to play with a different kart.
#        Call `python3 -c "import pystk; pystk.init(pystk.GraphicsConfig.ld()); print(pystk.list_karts())"` to see all values.
#     """
#     kart = ""
#
#     def __init__(self, player_id=0):
#         """
#         Set up a soccer player.
#         The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
#         """
#         all_players = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok',
#                        'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
#
#         self.kart = 'xue'  # we would like to maintain using this kart type.
#         self.previous_action = {'acceleration': 1, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False,
#                                 'steer': 0}
#         self.previous_location = [0.0, 0.0,
#                                   0.0]
#         self.team_mate_location = [0.0, 0.0, 0.0]
#         self.global_step = 0
#         self.reverse_stuck_count = 0
#         self.reverse_step = 0
#         self.previous_10_location = queue.Queue(maxsize=9)
#         self.previous_kart_v = 0
#
#         # Variables being used by deep learning model
#         self.screen_width = 400
#         self.screen_height = 300
#
#         self.red_teammate1_location = [-6.739621162414551, 0.30152183771133423, -56.0368537902832]
#         self.red_teammate2_location = [-0.009999435395002365, 0.30152183771133423, -53.05702209472656]
#         self.red_opponent1_location = [-6.5378875732421875, 0.30127325654029846, 55.04726028442383]
#         self.red_opponent2_location = [-0.04991918057203293, 0.3006347417831421, 52.13950729370117]
#
#         self.blue_opponent1_location = [-6.739621162414551, 0.30152183771133423, -56.0368537902832]
#         self.blue_opponent2_location = [-0.009999435395002365, 0.30152183771133423, -53.05702209472656]
#         self.blue_teammate1_location = [-6.5378875732421875, 0.30127325654029846, 55.04726028442383]
#         self.blue_teammate2_location = [-0.04991918057203293, 0.3006347417831421, 52.13950729370117]
#
#         self.previous_puck_location = [-0.47587478160858154, 0.3693891763687134, 0.5443593263626099]
#
#     def _to_world(self, aim_point_image, proj, view, height=0):
#         x, y, W, H = *aim_point_image, self.screen_width, self.screen_height
#         pv_inv = np.linalg.pinv(proj @ view)
#         xy, d = pv_inv.dot([float(x) / (W / 2) - 1, 1 - float(y) / (H / 2), 0, 1]), pv_inv[:, 2]
#         x0, x1 = xy[:-1] / xy[-1], (xy + d)[:-1] / (xy + d)[-1]
#         t = (height - x0[1]) / (x1[1] - x0[1])
#         return t * x1 + (1 - t) * x0
#
#     def _to_image(self, x, proj, view):
#         W, H = self.screen_width, self.screen_height
#         p = proj @ view @ np.array(list(x) + [1])
#         return np.array([W / 2 * (p[0] / p[-1] + 1), H / 2 * (1 - p[1] / p[-1])])
#
#     def act(self, image, player_info):
#         """
#         Set the action given the current image
#         :param image: numpy array of shape (300, 400, 3)
#         :param player_info: pystk.Player object for the current kart.
#         return: Dict describing the action
#         """
#         ##########################################################################
#         # Deep network module
#
#         # At the end we will have the following variables
#         # teammate_location - Location of our team mate
#         # player_location - location of the current player
#         # player_front - location of the current player front
#         # puck_location - location of the puck
#         # item_location_list - list of pick up items on the field. Can be null
#         # opponent_kart_location_list - list of opponent kart locations. Can be null. If null we can chase the items.
#         # if items are also null then we can chase the ball
#         # From the model
#         # TK Changing this for testing. In real time when playing against team mates we need to check this
#         # player_id = player_info.kart.player_id
#         # player_location = player_info.kart.location
#         # player_front = player_info.kart.front
#         # player_color = 'red' if player_id in (0, 2) else 'blue'
#
#         # PIL.Image.fromarray(image).save('/Users/ssras/GoogleDrive/Masters/DeepLearning/workspace/ut-deeplearning/final/det1_ann/image_test.png')
#
#         img_tensor = T.ToTensor()(image)
#         model = models.load_model()
#         # Camera Projection
#         proj = np.array(player_info.camera.projection).T
#         view = np.array(player_info.camera.view).T
#
#         # Getting the detections
#         det = model.detect(img_tensor, min_score=-2)
#         det_map = {'puck': [], 'kart': [], 'pickup': []}
#         for tup in det:
#             if tup[0] == 0 and tup[1] > -1.2:
#                 if len(det_map['puck']) == 0:
#                     det_map['puck'] = self._to_world(np.array([tup[2], tup[3]]), proj, view, 0.3695)
#             # elif tup[0] == 1 and tup[1] >= -2:
#             #     if len(det_map['kart']) < 4:
#             #         det_map['kart'].append(self._to_world((tup[2], tup[3]), proj, view, 0.2887))
#             #
#             # elif tup[0] == 2 and tup[1] >= 0:
#             #     det_map['pickup'].append(self._to_world([tup[2], tup[3]], proj, view, 0.010277748107910156))
#
#         # # Getting the puck_location, team mate karts, opponent karts and items on the field
#
#         puck_location = det_map['puck'] if len(det_map['puck']) == 1 else self.previous_puck_location
#         self.previous_puck_location = puck_location
#
#         # item_location_list = det_map['pickup']
#         #
#         # if player_color == 'red':
#         #     if player_id == 0:
#         #         teammate1_location, self.red_teammate1_location = player_location, player_location
#         #         teammate2_location, teammate_location = self.red_teammate2_location, self.red_teammate2_location
#         #     else:
#         #         teammate1_location, teammate_location = self.red_teammate1_location, self.red_teammate1_location
#         #         teammate2_location, self.red_teammate2_location = player_location, player_location
#         # else:
#         #     if player_id == 1:
#         #         teammate1_location, self.blue_teammate1_location = player_location, player_location
#         #         teammate2_location, teammate_location = self.blue_teammate2_location
#         #     else:
#         #         teammate1_location, teammate_location = self.blue_teammate1_location, self.blue_teammate1_location
#         #         teammate2_location, self.blue_teammate2_location = player_location, player_location
#         #
#         # # Identifying opponent and teammate cards from deep network predictions
#         # opponent_kart_location_list = []
#         # for kart_loc in det_map['kart']:
#         #     if (np.linalg.norm(np.subtract(teammate1_location, kart_loc)) > 5
#         #             and np.linalg.norm(np.subtract(teammate2_location, kart_loc) > 5)):
#         #         opponent_kart_location_list.append(kart_loc)
#
#         ############################### End of deepnetwork module ##############################################
#
#         # default value of each control output.
#         nitro_value = False
#         brake_value = False
#         drift_value = False
#         rescue_value = False
#         acceleration_value = 1
#         self.global_step += 1
#
#         kart_x = player_info.kart.location[0]
#         kart_y = player_info.kart.location[2]
#         velocity_x = player_info.kart.velocity[0]
#         velocity_y = player_info.kart.velocity[2]
#         front_x = player_info.kart.front[0]
#         front_y = player_info.kart.front[2]
#         # ball_x = socceri.ball.location[0]
#         # ball_y = socceri.ball.location[2]
#         ball_x = puck_location[0]
#         ball_y = puck_location[1]
#
#         gate_y = 63.5 if (player_info.kart.player_id in (0, 2)) else -63.5
#         # ssras
#         goal_line_l = [-10.510000228881836, 0.07000000029802322, 64.5]
#         goal_line_r = [10.460000038146973, 0.07000000029802322, 64.5]
#         x_delta = ball_x - front_x
#         y_delta = ball_y - front_y
#         ball_kart_distance = math.sqrt(x_delta ** 2 + y_delta ** 2)
#         x_direction = front_x - kart_x
#         y_direction = front_y - kart_y
#         angle_ball = np.arctan(y_delta / (x_delta + 0.00001))  # this value is from [-pi/2, pi/2]
#         angle_kart_temp = np.arctan(y_direction / (x_direction + 0.0001))
#         velocity = math.sqrt(velocity_x ** 2 + velocity_y ** 2)
#         kart_v = math.sqrt(velocity_x ** 2 + velocity_y ** 2)
#         sign_kart = np.sign(x_delta) * np.sign(y_delta) * np.sign(front_x)
#
#         # block for reverse if hit the wall or stuck somewhere
#         if self.previous_10_location.full():
#             self.previous_10_location.get()
#             self.previous_10_location.put(player_info.kart.location, block=False)
#         else:
#             self.previous_10_location.put(player_info.kart.location, block=False)
#         if self.reverse_stuck_count == 0 and self.previous_10_location.qsize() == 9:
#             if (math.sqrt(math.pow(self.previous_10_location.get()[0] - player_info.kart.location[0], 2)
#                           + math.pow(self.previous_10_location.get()[2] - player_info.kart.location[2],
#                                      2)) < 0.15):
#                 acceleration_value = 0
#                 brake_value = True
#                 self.reverse_stuck_count += 1
#                 return {'acceleration': acceleration_value, 'brake': brake_value, 'drift': False,
#                         'nitro': False, 'rescue': False, 'steer': -1 * sign_kart}
#
#         elif 0 < self.reverse_stuck_count <= 10:
#             acceleration_value = 0
#             brake_value = True
#             self.reverse_stuck_count += 1
#             if self.reverse_stuck_count > 10:
#                 self.reverse_stuck_count = 0
#             return {'acceleration': acceleration_value, 'brake': brake_value, 'drift': False,
#                     'nitro': False, 'rescue': False, 'steer': -1 * sign_kart}
#
#         # ----set the speed----#
#
#         if (kart_v < 8 and ball_kart_distance > 7):
#             if (player_info.kart.player_id % 2 == 0):
#                 acceleration_value = 0.8
#                 nitro_value = True
#             else:
#                 acceleration_value = 0.6
#         else:
#             acceleration_value = 0.4
#
#         # -----calculate the angle of the kart, it is front (-pi,pi) -----#
#
#         if (x_direction >= 0 and y_direction >= 0):
#             angle_kart = angle_kart_temp
#         elif (x_direction < 0 and y_direction >= 0):
#             angle_kart = np.pi + angle_kart_temp
#         elif (x_direction < 0 and y_direction < 0):
#             angle_kart = -np.pi + angle_kart_temp
#         elif (x_direction >= 0 and y_direction < 0):
#             angle_kart = angle_kart_temp
#         # print("angle of kart:",angle_kart)
#
#         # -----here is the code of calculate steering in general to chase the puck----#
#         steer_value = self.chasing(angle_kart, angle_ball, kart_x, kart_y, ball_x, ball_y, front_x, front_y)
#         steer_value = 7.5 * steer_value
#         # steer_value = steer_value / abs(np.pi*y_delta)
#         if (steer_value >= 3 or steer_value <= -3):
#             # drift_value = True
#             nitro_value = False
#             brake_value = True
#             acceleration_value = 0.5
#
#         if (steer_value > 0.75):
#             nitro_value = False
#
#         # ----when approaching ball, slow down---#
#         if (ball_kart_distance < 10 and kart_v < 15):
#             # brake_value = True
#             acceleration_value = 0.25
#
#         elif (ball_kart_distance < 10 and kart_v > 15):
#             acceleration_value = 0
#
#         dis_threshold = (2.5 + 2.52) / 2 - 0.4
#         is_facing_opp_gl = front_y > kart_y if player_info.kart.player_id in (0, 2) else front_y < kart_y
#
#         if (ball_kart_distance < dis_threshold and is_facing_opp_gl):
#
#             if (kart_v > 10):
#                 acceleration_value = 0
#             else:
#                 acceleration_value = 0.25
#             if (ball_x >= -10.5 and ball_x <= 10.5):
#                 front_x_1 = front_x - ball_x
#                 x = ball_x
#             else:
#                 front_x_1 = front_x
#                 x = 0
#             angle_kart_gl = np.arctan((gate_y - front_y) / (front_x_1 + 0.0001))
#             # ----when I have the puck i should steer to the gl----#
#             steer_value = self.chasing(angle_kart, angle_kart_gl, kart_x, kart_y, x, gate_y, front_x, front_y)
#
#         # -----below is the code for steer to the goaline or in the case to our own gate, push away the puck---#
#         if (player_info.kart.player_id in (0, 2)):
#             gate_y = 63.5
#             ##check if the ball, kart and front are aligned.
#             angle_front_gate_nx = np.arctan((gate_y - front_y) / ((front_x + 10.5) + 0.00001))
#             angle_front_gate_px = np.arctan((gate_y - front_y) / ((front_x - 10.5) + 0.00001))
#             angle_front_gate_nx_own = np.arctan((-gate_y - front_y) / ((front_x + 10.5) + 0.00001))
#             angle_front_gate_px_own = np.arctan((-gate_y - front_y) / ((front_x - 10.5) + 0.00001))
#             # facing your opponent goal line
#
#             # facing your own gate
#             if (angle_front_gate_nx_own >= 0):
#                 angle_front_gate_nx_own = -np.pi + angle_front_gate_nx
#             if (angle_front_gate_px_own >= 0):
#                 angle_front_gate_px_own = -np.pi + angle_front_gate_px
#             if (x_delta >= 0):
#                 angle_ball_change_own = -np.pi - angle_ball
#             else:
#                 angle_ball_change_own = angle_ball
#             if (front_y < kart_y and (
#                     angle_ball_change_own <= angle_front_gate_px_own and angle_ball >= angle_front_gate_nx_own) and ball_y < front_y):
#                 steer_value = steer_value - np.sign(steer_value) * 0.3
#                 if (kart_v > 4):
#                     brake_value = True
#                     acceleration_value = 0.1
#         else:
#             gate_y = -63.5
#
#             angle_front_gate_nx = np.arctan((gate_y - front_y) / ((front_x + 10.5) + 0.00001))
#             angle_front_gate_px = np.arctan((gate_y - front_y) / ((front_x - 10.5) + 0.00001))
#             angle_front_gate_nx_own = np.arctan((-gate_y - front_y) / ((front_x + 10.5) + 0.00001))
#             angle_front_gate_px_own = np.arctan((-gate_y - front_y) / ((front_x - 10.5) + 0.00001))
#
#             if (angle_front_gate_nx >= 0):
#                 angle_front_gate_nx = -np.pi / 2 + angle_front_gate_nx
#             if (angle_front_gate_px >= 0):
#                 angle_front_gate_px = -np.pi / 2 + angle_front_gate_px
#
#             # facing your opponent goal line
#             # facing your own gate
#             if (angle_front_gate_nx_own <= 0):
#                 angle_front_gate_nx_own = np.pi - angle_front_gate_nx
#             if (angle_front_gate_px_own <= 0):
#                 angle_front_gate_px_own = np.pi - angle_front_gate_px
#
#             if (x_delta >= 0):
#                 angle_ball_change_own = angle_ball
#             else:
#                 angle_ball_change_own = np.pi - angle_ball
#             if (front_y > kart_y and (
#                     angle_ball_change_own >= angle_front_gate_px_own
#                     and angle_ball <= angle_front_gate_nx_own) and ball_y > front_y):
#
#                 steer_value = steer_value - np.sign(steer_value) * 0.3
#                 if (kart_v > 4):
#                     brake_value = True
#                     acceleration_value = 0.1
#             # ____________________#
#
#         self.previous_location = player_info.kart.location
#         nitro_value = False
#         self.previous_kart_v = kart_v
#         action = {'acceleration': acceleration_value, 'brake': brake_value, 'drift': False, 'nitro': False,
#                   'rescue': False, 'steer': steer_value}
#
#         self.team_mate_location = player_info.kart.location
#
#         return action
#
#     def chasing(self, angle_kart, angle_object_to_kart, kart_x, kart_y, object_x, object_y, front_x, front_y):
#         delta_angle = angle_kart - angle_object_to_kart
#         if (delta_angle > np.pi):
#             steer_value = np.pi - (2 * np.pi - (delta_angle))
#         else:
#             steer_value = -(np.pi - abs(delta_angle))
#
#         x_delta = object_x - front_x
#         y_delta = object_y - front_y
#         if (x_delta <= 0 and y_delta > 0):
#             if (delta_angle > np.pi):
#                 steer_value = np.pi - (2 * np.pi - (delta_angle))
#             else:
#                 steer_value = -(np.pi - abs(delta_angle))
#                 # steer_value = 2*np.pi-(np.pi-(delta_angle))
#             # print("just steer_value_t D:",steer_value)
#         elif (x_delta > 0 and y_delta > 0):
#             if (delta_angle < -np.pi):
#                 steer_value = 2 * np.pi + delta_angle
#             else:
#                 steer_value = delta_angle
#             # print("just steer_value_t B:",steer_value)
#         elif (x_delta > 0 and y_delta <= 0):
#             if (delta_angle > np.pi):
#                 steer_value = -(2 * np.pi - delta_angle)
#             else:
#                 steer_value = delta_angle
#             # print("just steer_value_t A:",steer_value)
#         elif (x_delta <= 0 and y_delta <= 0):
#             if (delta_angle < -np.pi):
#                 steer_value = -(np.pi - (2 * np.pi + (delta_angle)))
#             else:
#                 # steer_value = -(np.pi-abs(delta_angle))
#                 steer_value = 2 * np.pi - (np.pi - (delta_angle))
#         if (steer_value > np.pi):
#             steer_value = -(2 * np.pi - steer_value)
#         elif (steer_value < -np.pi):
#             steer_value = 2 * np.pi + steer_value
#
#         return steer_value
import numpy as np
import math
import queue
from torchvision import transforms as T


# from . import models
from agent import models


class Locatable:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Locatable(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Locatable(self.x + other.x, self.y + other.y)

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    # This calculates the angle between two locatables
    def __xor__(self, other):
        angle_self = self.orientation()
        angle_other = other.orientation()
        if angle_self >= 0:
            if angle_self > angle_other > angle_self - np.pi or angle_other > angle_self:
                return (angle_self - angle_other) / np.pi
            else:
                return -(2 * np.pi + angle_other - angle_self) / np.pi
        else:
            if angle_self < angle_other < angle_self + np.pi or angle_other < angle_self:
                return (angle_self - angle_other) / np.pi
            else:
                return (2 * np.pi - angle_other + angle_self) / np.pi

    def orientation(self):
        angle = np.arctan(self.y / (self.x + 0.00001))
        if self.x >= 0:
            return angle
        elif self.y >= 0:
            return np.pi + angle
        else:
            return -np.pi + angle


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

    def __init__(self, team_id=0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        self.team_id = team_id
        self.kart = 'wilber'

        # hyper-parameter
        self.QUEUE_CAPACITY = 5

        self.global_step = 0
        self.reverse_stuck_count = 0
        self.previous_locations = queue.Queue(maxsize=self.QUEUE_CAPACITY)
        self.previous_location = [0,0,0]

        self.screen_width = 400
        self.screen_height = 300

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

    def update_previous_locations(self, kart_location):
        if self.previous_locations.full():
            self.previous_locations.get()
        self.previous_locations.put(kart_location, block=False)

    def kart_got_stuck(self, kart_location):
        STUCK_DISPLACEMENT_THRESHOLD = 0.15
        if self.reverse_stuck_count == 0 and self.previous_locations.qsize() == self.QUEUE_CAPACITY:
            if (self.previous_locations.get() - kart_location).length() < STUCK_DISPLACEMENT_THRESHOLD:
                self.reverse_stuck_count += 1
                return True
        if 0 < self.reverse_stuck_count < self.QUEUE_CAPACITY:
            self.reverse_stuck_count += 1
            return True
        if self.reverse_stuck_count >= self.QUEUE_CAPACITY:
            self.reverse_stuck_count = 0
            return True
        return False

    def stuck_response(self, kart_sign):
        return {'acceleration': 0, 'brake': True, 'drift': False,
                'nitro': False, 'rescue': False, 'steer': -1 * kart_sign}

    def act(self, image, player_info, ball):  # used for training
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        ##########################################################################

        # General Metrics for the playground
        BALL_SIZE = 2.5
        KART_WIDTH = 0.77
        TEAM_NUMBER = 2
        GATE_LOCATION = Locatable(0, 64.5 if self.team_id % TEAM_NUMBER == 0 else -64.5)
        OFFSET = (BALL_SIZE + KART_WIDTH) / 3
        RUNNING_LANE_OFFSET = 15
        ACCELERATION_DURING_DRIFT = 0.25

        STEER_MULTIPLIER_GENERAL = 30
        STEER_MULTIPLIER_MILD = 25
        STEER_MULTIPLIER_GENTLE = 10
        STEER_MULTIPLIER_SLOW = 5
        STEER_MULTIPLIER_REDUCED = 12
        STEER_MULTIPLIER_GRAND = 50

        THRESHOLD_FOR_VELOCITY_TO_QUALIFY_NITRO = 10
        THRESHOLD_FOR_DISTANCE_TO_QUALIFY_NITRO = 15
        THRESHOLD_FOR_STEER_VALUE_TO_QUALIFY_DRIFT = 10
        THRESHOLD_FOR_DISTANCE_TO_QUALIFY_SLOW_DOWN = 12
        THRESHOLD_FOR_VELOCITY_TO_QUALIFY_SLOW_DOWN = 7.5
        THRESHOLD_FOR_DISTANCE_TO_QUALIFY_ATTACK = 8
        THRESHOLD_FOR_VELOCITY_TO_QUALIFY_ATTACK = 20
        THRESHOLD_FOR_VELOCITY_TO_QUALIFY_DEFENCE = 30
        THRESHOLD_FOR_ANGLE_TO_QUALIFY_GENERAL_STEER = np.pi / 4
        THRESHOLD_FOR_ANGLE_TO_QUALIFY_SLOW_STEER = np.pi / 50


        nitro_value = False
        brake_value = False
        drift_value = False
        acceleration_value = 1
        self.global_step += 1

        img_tensor = T.ToTensor()(image)
        model = models.load_model()
        # Camera Projection
        proj = np.array(player_info.camera.projection).T
        view = np.array(player_info.camera.view).T

        det = model.detect(img_tensor, min_score=-0.5,max_det=1)
        ball_info = self.previous_location
        for tup in det:
            ball_info = self._to_world([tup[2], tup[3]], proj, view, 0.3695)
            self.previous_location = ball_info
            break


        # # Getting the puck_location, team mate karts, opponent karts and items on the field

        print('info calculated:' + str(det))
        print('model calculated:'+str(ball_info))
        print('real:'+str(ball))


        kart_location = Locatable(player_info.kart.location[0], player_info.kart.location[2])
        kart_velocity = Locatable(player_info.kart.velocity[0], player_info.kart.velocity[2])
        kart_frontier_location = Locatable(player_info.kart.front[0], player_info.kart.front[2])
        ball_location = Locatable(ball_info[0], ball_info[2])
        ball_distance = ball_location - kart_frontier_location

        #         ball_x = puck_location[0]  #used for eval
        #         ball_y = puck_location[1]  #used for eval

        ##?
        sign_kart = np.sign(ball_distance.x) * np.sign(ball_distance.y) * np.sign(kart_frontier_location.x)

        #         print("step: %d, speed: %0.2f, dist: % 0.2f" % (self.global_step, kart_v, ball_kart_distance))

        # block for reverse if hit the wall or stuck somewhere
        self.update_previous_locations(kart_location)

        if self.kart_got_stuck(kart_location):
            return self.stuck_response(sign_kart)

        # ----chase the puck----#
        steer_value = STEER_MULTIPLIER_GRAND * ((kart_frontier_location - kart_location) ^ (ball_location - kart_frontier_location))

        # ----speed up when chasing puck----#
        if kart_velocity.length() < THRESHOLD_FOR_VELOCITY_TO_QUALIFY_NITRO and ball_distance.length() > THRESHOLD_FOR_DISTANCE_TO_QUALIFY_NITRO:
            nitro_value = True

        # ----set up drift----#
        if (steer_value >= THRESHOLD_FOR_STEER_VALUE_TO_QUALIFY_DRIFT or steer_value <= -THRESHOLD_FOR_STEER_VALUE_TO_QUALIFY_DRIFT):
            drift_value = True
            nitro_value = False
            brake_value = True
            acceleration_value = ACCELERATION_DURING_DRIFT

        # ----slow down upon approaching puck---#
        if ball_distance.length() < THRESHOLD_FOR_DISTANCE_TO_QUALIFY_SLOW_DOWN and kart_velocity.length() > THRESHOLD_FOR_VELOCITY_TO_QUALIFY_SLOW_DOWN:
            acceleration_value = 0

        # ----Strategy for attack---#
        is_facing_goal = (kart_frontier_location - kart_location).y > 0 if self.team_id % TEAM_NUMBER == 0 else (kart_frontier_location - kart_location).y < 0
        if ball_distance.length() < THRESHOLD_FOR_DISTANCE_TO_QUALIFY_ATTACK and is_facing_goal:
            acceleration_value = 1 if kart_velocity.length() < THRESHOLD_FOR_VELOCITY_TO_QUALIFY_ATTACK else 0
            goal_position = GATE_LOCATION - ball_location
            chase_location = ball_location - Locatable(OFFSET * np.cos(goal_position.orientation()),
                                                       OFFSET * np.sin(goal_position.orientation()))
            if (chase_location.y < kart_frontier_location.y and GATE_LOCATION.y > 0) or (
                    chase_location.y > kart_frontier_location.y and GATE_LOCATION.y < 0):
                chase_location.y = kart_frontier_location.y
            goal_vector = chase_location - kart_frontier_location
            steer_value = (kart_frontier_location - kart_location) ^ goal_vector

            if abs(goal_vector.orientation()) < THRESHOLD_FOR_ANGLE_TO_QUALIFY_GENERAL_STEER:
                steer_value = STEER_MULTIPLIER_SLOW * steer_value if abs(
                    goal_vector.orientation()) < THRESHOLD_FOR_ANGLE_TO_QUALIFY_SLOW_STEER else steer_value * STEER_MULTIPLIER_GENERAL
            else:
                steer_value = STEER_MULTIPLIER_REDUCED * steer_value

        if ball_distance.length() < THRESHOLD_FOR_DISTANCE_TO_QUALIFY_ATTACK and ball_location.x > -10.45 or ball_location.x < 10.45 and abs(
                (ball_location - GATE_LOCATION).y) < 5:
            goal_vector = GATE_LOCATION - ball_location
            chase_vector = ball_location - Locatable(OFFSET * np.cos(goal_vector.orientation()),
                                                     OFFSET * np.sin(goal_vector.orientation()))
            steer_value = STEER_MULTIPLIER_GRAND * ((kart_frontier_location - kart_location) ^ (chase_vector - kart_frontier_location))

        # ----Strategy for defense---#
        if not is_facing_goal:
            acceleration_value = 1 if kart_velocity.length() < THRESHOLD_FOR_VELOCITY_TO_QUALIFY_DEFENCE else 0
            is_ball_ahead = (ball_location - kart_frontier_location).y > 0 if self.team_id % TEAM_NUMBER == 0 else (ball_location - kart_frontier_location).y < 0
            if is_ball_ahead:
                goal_vector = Locatable(RUNNING_LANE_OFFSET if ball_location.x > 0 else -RUNNING_LANE_OFFSET,GATE_LOCATION.y) - ball_location
                chase_vector = ball_location - Locatable(OFFSET * np.cos(goal_vector.orientation()), OFFSET * np.sin(goal_vector.orientation()))
                steer_value = STEER_MULTIPLIER_GENTLE * ((kart_frontier_location - kart_location) ^ (chase_vector - kart_frontier_location))
                drift_value = False

        return {'acceleration': acceleration_value, 'brake': brake_value, 'drift': drift_value, 'nitro': nitro_value,
                'rescue': False, 'steer': steer_value}

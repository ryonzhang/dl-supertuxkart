import pystk
import numpy as np


class Player:
    def __init__(self, player, team=0):
        self.player = player
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(controller=pystk.PlayerConfig.Controller.PLAYER_CONTROL, kart=self.player.kart, team=self.team)
    
    def __call__(self, image, player_info,ball):
        return self.player.act(image, player_info,ball)

def quat_rot(r, v, inverse=False):
    inv = 1 - 2 * float(inverse)
    return np.array([(1 - 2 * (r[(i + 1) % 3] ** 2 + r[(i + 2) % 3] ** 2)) * v[i] +
                     2 * (r[i] * r[(i + 1) % 3] - r[(i + 2) % 3] * r[3] * inv) * v[(i + 1) % 3] +
                     2 * (r[i] * r[(i + 2) % 3] + r[(i + 1) % 3] * r[3] * inv) * v[(i + 2) % 3] for i in range(3)])



class Tournament:
    _singleton = None

    def __init__(self, players, screen_width=400, screen_height=300, track='icy_soccer_field'):
        assert Tournament._singleton is None, "Cannot create more than one Tournament object"
        Tournament._singleton = self

        self.graphics_config = pystk.GraphicsConfig.hd()
        self.graphics_config.screen_width = screen_width
        self.graphics_config.screen_height = screen_height
        pystk.init(self.graphics_config)

        self.race_config = pystk.RaceConfig(num_kart=len(players), difficulty = 0, track=track, mode=pystk.RaceConfig.RaceMode.SOCCER)
        #self.race_config = pystk.RaceConfig(num_kart=len(players), track=track, mode=pystk.RaceConfig.RaceMode.SOCCER)
        self.race_config.players.pop()
        
        self.active_players = []
        print('ryon:')
        for p in players:
            if p is not None:
                self.race_config.players.append(p.config)
                self.active_players.append(p)
        
        self.k = pystk.Race(self.race_config)

        self.k.start()
        self.k.step()


    @staticmethod
    def _to_kart(x, kart):
        return quat_rot(kart.rotation, x - kart.location, True)

    def _to_image(self, x, proj, view):
        W, H = self.graphics_config.screen_width, self.graphics_config.screen_height
        p = proj @ view @ np.array(list(x) + [1])
        return np.array([W / 2 * (p[0] / p[-1] + 1), H / 2 * (1 - p[1] / p[-1])])


    def play(self, save=None, max_frames=50):
        state = pystk.WorldState()
        import PIL.Image
        import os
        prev_score = [0,0]
        for t in range(max_frames):
            #print('\rframe %d' % t, end='\r')

            state.update()

            #print some info
            test = True
            if test:
                #print('soccer ball location', state.soccer.ball.location)
                #print('soccer ball location', state.soccer.ball.location[1])
                #print('soccer ball location', state.soccer.ball.location[2])
                #print('kart 1 location', state.karts[0].location)
                #print('kart 2 location', state.karts[1].location)
                #print('kart 3 location', state.karts[2].location)
                #print('kart 4 location', state.karts[3].location)
                #print('score', state.soccer.score)
                if prev_score != state.soccer.score:
                    print("Scored: ", state.soccer.score, " at frame ", t)
                    prev_score = state.soccer.score
                """
                goal = state.soccer.goal_line[0][0]
                print('Goal Poistion: ', goal[0],",", goal[1], ",",goal[2])
                goal1 = state.soccer.goal_line[0][1]
                print('Goal Poistion: ', goal1[0],",", goal1[1], ",",goal1[2])
                """
                #for kart in state.karts:
                    #print(kart.location)
                #print('goal line', state.soccer.goal_line[0][0])
                #print('goal line', state.soccer.goal_line[0][0][0])
                #print('goal line', state.soccer.goal_line[0][0][1])
                #print('goal line', state.soccer.goal_line[0][0][2])

                #print("Soccer location: ", state.soccer.ball.location)
                #aim_point_car = self._to_kart(np.array(state.soccer.ball.location), state.karts[0])
                #print("Soccer location to car: ", aim_point_car )
    

            list_actions = []
            for i, p in enumerate(self.active_players):
                player = state.players[i]

                image = np.array(self.k.render_data[i].image)
                
                action = pystk.Action()
                #print('soccer ball location', state.soccer.ball.location)
                #player_action = p( state.soccer, player)
                player_action = p(image, player,state.soccer.ball.location)
                for a in player_action:
                    setattr(action, a, player_action[a])
                
                list_actions.append(action)

                proj = np.array(player.camera.projection).T
                view = np.array(player.camera.view).T
                # print(state.players)


                    #Generate labels for training sets
                aim_point_world_soccer = state.soccer.ball.location
                aps = self._to_image(aim_point_world_soccer, proj, view)
                #print(state.soccer.goal_line )

                list = [[state.soccer.ball.size / 2, 0.18, state.soccer.ball.size / 2],
                        [state.soccer.ball.size / 2, 0.18, -state.soccer.ball.size / 2],
                        [state.soccer.ball.size / 2, -0.18, state.soccer.ball.size / 2],
                        [state.soccer.ball.size / 2, -0.18, -state.soccer.ball.size / 2],
                        [-state.soccer.ball.size / 2, 0.18, state.soccer.ball.size / 2],
                        [-state.soccer.ball.size / 2, 0.18, -state.soccer.ball.size / 2],
                        [-state.soccer.ball.size / 2, -0.18, state.soccer.ball.size / 2],
                        [-state.soccer.ball.size / 2, -0.18, -state.soccer.ball.size / 2]]

                width1 = 400
                width2 = 0
                height1 = 300
                height2 = 0
                for item in list:
                    aps_flex = self._to_image(
                        [aim_point_world_soccer[0] + item[0], aim_point_world_soccer[1] + item[1],
                         aim_point_world_soccer[2] + item[2]], proj, view)
                    if width1 >aps_flex[0]:
                        width1 = aps_flex[0]
                    if width2 < aps_flex[0]:
                        width2 = aps_flex[0]
                    if height1 > aps_flex[1]:
                        height1=aps_flex[1]
                    if height2 <aps_flex[1]:
                        height2 = aps_flex[1]

                player_x=player.kart.location[0]
                player_y=player.kart.location[2]
                front_x = player.kart.front[0]
                front_y = player.kart.front[2]
                ball_x = aim_point_world_soccer[0]
                ball_y = aim_point_world_soccer[2]

                vector_self = [front_x-player_x,front_y-player_y]
                vector_ball = [ball_x - player_x, ball_y - player_y]

                dot_prod = vector_self[0]*vector_ball[0]+vector_self[1]*vector_ball[1]

                if dot_prod >0 and 0 <= aps[0] < self.graphics_config.screen_width and 0 <= aps[1] < self.graphics_config.screen_height and -30<player.kart.location[0]<30 and -55<player.kart.location[2]<55:
                    if (width2-width1)*(height2-height1)>40:
                        # print([[width1, height1, width2, height2]])
                        PIL.Image.fromarray(image).save(os.path.join('dense_data/train/', 'player%02d_%05d.png' % (i, t)))
                        np.savez(os.path.join('dense_data/train/', 'player%02d_%05d' % (i, t)) + '.npz', puck=[[width1,height1,width2,height2]])


                        '''
                        # with open(os.path.join(save, 'player%02d_%05d' % (i, t)) + '.csv', 'w') as f:
                        #     f.write('%0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f' % (
                        #         state.soccer.ball.location[0],state.soccer.ball.location[1], state.soccer.ball.location[2],
                        #         state.soccer.goal_line[0][0][0], state.soccer.goal_line[0][0][1], state.soccer.goal_line[0][0][2]
                        #     ))
                        '''
                        '''
                        f.write('%0.3f %0.3f %0.3f %0.3f %0.3f %0.3f' %tuple(state.soccer.ball.location), tuple(state.soccer.goal_line[0][0])))
                        '''
                    

            s = self.k.step(list_actions)
            if not s:  # Game over
                break

        if save is not None:
            import subprocess
            for i, p in enumerate(self.active_players):
                dest = os.path.join(save, 'player%02d' % i)
                output = save + '_player%02d.mp4' % i
                subprocess.call(['ffmpeg', '-y', '-framerate', '10', '-i', dest + '_%05d.png', output])
        if hasattr(state, 'soccer'):
            return state.soccer.score
        return state.soccer_score

    def close(self):
        self.k.stop()
        del self.k

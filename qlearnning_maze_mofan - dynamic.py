#### 搭建可视化环境 ####
import numpy as np
import pandas as pd
import time
import sys
import math
import random
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 9  # grid height
MAZE_W = 9  # grid width
class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 4
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        # create origin
        origin = np.array([20, 20])
        # hell
        hell1_center = origin + np.array([UNIT , 0])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT * 4, 0])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')
        
        # hell
        hell3_center = origin + np.array([UNIT * 8, 0])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')

        #hell
        hell4_center = origin + np.array([UNIT * 6, UNIT])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 15, hell4_center[1] - 15,
            hell4_center[0] + 15, hell4_center[1] + 15,
            fill='black')

        # hell
        hell5_center = origin + np.array([0, UNIT * 2])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - 15, hell5_center[1] - 15,
            hell5_center[0] + 15, hell5_center[1] + 15,
            fill='black')
        
        # hell
        hell6_center = origin + np.array([UNIT * 3, UNIT * 2])
        self.hell6 = self.canvas.create_rectangle(
            hell6_center[0] - 15, hell6_center[1] - 15,
            hell6_center[0] + 15, hell6_center[1] + 15,
            fill='black')

        # hell
        hell7_center = origin + np.array([UNIT * 2, UNIT * 3])
        self.hell7 = self.canvas.create_rectangle(
            hell7_center[0] - 15, hell7_center[1] - 15,
            hell7_center[0] + 15, hell7_center[1] + 15,
            fill='black')
        # hell
        hell8_center = origin + np.array([UNIT * 5, UNIT * 3])
        self.hell8 = self.canvas.create_rectangle(
            hell8_center[0] - 15, hell8_center[1] - 15,
            hell8_center[0] + 15, hell8_center[1] + 15,
            fill='black')
        # hell
        hell9_center = origin + np.array([UNIT * 7, UNIT * 3])
        self.hell9 = self.canvas.create_rectangle(
            hell9_center[0] - 15, hell9_center[1] - 15,
            hell9_center[0] + 15, hell9_center[1] + 15,
            fill='black')
        # hell
        hell10_center = origin + np.array([UNIT * 8, UNIT * 3])
        self.hell10 = self.canvas.create_rectangle(
            hell10_center[0] - 15, hell10_center[1] - 15,
            hell10_center[0] + 15, hell10_center[1] + 15,
            fill='black')
        # hell
        hell11_center = origin + np.array([0, UNIT * 4])
        self.hell11 = self.canvas.create_rectangle(
            hell11_center[0] - 15, hell11_center[1] - 15,
            hell11_center[0] + 15, hell11_center[1] + 15,
            fill='black')
        # hell
        hell12_center = origin + np.array([UNIT * 3, UNIT * 4])
        self.hell12 = self.canvas.create_rectangle(
            hell12_center[0] - 15, hell12_center[1] - 15,
            hell12_center[0] + 15, hell12_center[1] + 15,
            fill='black')
        # hell
        hell13_center = origin + np.array([UNIT, UNIT * 6])
        self.hell13 = self.canvas.create_rectangle(
            hell13_center[0] - 15, hell13_center[1] - 15,
            hell13_center[0] + 15, hell13_center[1] + 15,
            fill='black')
        # hell
        hell14_center = origin + np.array([UNIT * 3, UNIT * 7])
        self.hell14 = self.canvas.create_rectangle(
            hell14_center[0] - 15, hell14_center[1] - 15,
            hell14_center[0] + 15, hell14_center[1] + 15,
            fill='black')
        # hell
        hell15_center = origin + np.array([0, UNIT * 8])
        self.hell15 = self.canvas.create_rectangle(
            hell15_center[0] - 15, hell15_center[1] - 15,
            hell15_center[0] + 15, hell15_center[1] + 15,
            fill='black')
        # hell
        hell16_center = origin + np.array([UNIT * 5, UNIT * 5])
        self.hell16 = self.canvas.create_rectangle(
            hell16_center[0] - 15, hell16_center[1] - 15,
            hell16_center[0] + 15, hell16_center[1] + 15,
            fill='black')
        # hell
        hell17_center = origin + np.array([UNIT * 5, UNIT * 6])
        self.hell17 = self.canvas.create_rectangle(
            hell17_center[0] - 15, hell17_center[1] - 15,
            hell17_center[0] + 15, hell17_center[1] + 15,
            fill='black')
        # hell
        hell18_center = origin + np.array([UNIT * 5, UNIT * 7])
        self.hell18 = self.canvas.create_rectangle(
            hell18_center[0] - 15, hell18_center[1] - 15,
            hell18_center[0] + 15, hell18_center[1] + 15,
            fill='black')
        # hell
        hell19_center = origin + np.array([UNIT * 6, UNIT * 5])
        self.hell19 = self.canvas.create_rectangle(
            hell19_center[0] - 15, hell19_center[1] - 15,
            hell19_center[0] + 15, hell19_center[1] + 15,
            fill='black')
        # hell
        hell20_center = origin + np.array([UNIT * 7, UNIT * 5])
        self.hell20 = self.canvas.create_rectangle(
            hell20_center[0] - 15, hell20_center[1] - 15,
            hell20_center[0] + 15, hell20_center[1] + 15,
            fill='black')
        # hell
        hell21_center = origin + np.array([UNIT * 7, UNIT * 6])
        self.hell21 = self.canvas.create_rectangle(
            hell21_center[0] - 15, hell21_center[1] - 15,
            hell21_center[0] + 15, hell21_center[1] + 15,
            fill='black')
        # hell
        hell22_center = origin + np.array([UNIT * 7, UNIT * 7])
        self.hell22 = self.canvas.create_rectangle(
            hell22_center[0] - 15, hell22_center[1] - 15,
            hell22_center[0] + 15, hell22_center[1] + 15,
            fill='black')



        # create oval
        oval_center = origin + UNIT * 6
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')
        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # pack all
        self.canvas.pack()
    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)     # list [起始点四个角点的坐标]

    def move_grid(self):
        dx = random.randint(-40, 40)  # 随机生成x方向的移动量
        dy = random.randint(-40, 40)  # 随机生成y方向的移动量
        x1, y1, x2, y2 = self.canvas.coords(self.hell13)  # 获取当前矩形格子的坐标
        new_x1 = x1 + dx
        new_y1 = y1 + dy
        new_x2 = x2 + dx
        new_y2 = y2 + dy

        # 检查新的坐标是否超出边界或与目标格子重叠
        if self.is_valid_move(new_x1, new_y1, new_x2, new_y2):
            self.canvas.move(self.hell13, dx, dy)  # 移动矩形格子

        self.root.after(500, self.move_grid)  # 间隔一定时间后再次移动格子

    def move_grid1(self):
        dx = random.randint(-40, 40)  # 随机生成x方向的移动量
        dy = random.randint(-40, 40)  # 随机生成y方向的移动量
        x1, y1, x2, y2 = self.canvas.coords(self.hell16)  # 获取当前矩形格子的坐标
        new_x1 = x1 + dx
        new_y1 = y1 + dy
        new_x2 = x2 + dx
        new_y2 = y2 + dy

        # 检查新的坐标是否超出边界或与目标格子重叠
        if self.is_valid_move(new_x1, new_y1, new_x2, new_y2):
            self.canvas.move(self.hell16, dx, dy)  # 移动矩形格子

        self.root.after(500, self.move_grid1)  # 间隔一定时间后再次移动格子

    def is_valid_move(self, x1, y1, x2, y2):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        overlapping_items = self.canvas.find_overlapping(x1, y1, x2, y2)

        if x1 < 0 or y1 < 0 or x2 > 360 or y2 > 360:
            return False
        elif self.oval in overlapping_items:
            return False
        else:
            return True

    def step(self, action,episode):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        self.path_line = None
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        # 绘制路径线
        if episode == 499:
            x1, y1, x2, y2 = self.canvas.coords(self.rect)
            if self.path_line is None:
                self.path_line = self.canvas.create_line(x1, y1, x2, y2, fill='red')
            else:
                self.canvas.coords(self.path_line, x1, y1, x2, y2)

        s_ = self.canvas.coords(self.rect)  # next state
        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 2
            done = True
            s_ = [1,1,1,1]      # 一个特殊的列表，代表Terminal,只是起到标志性的作用，所以该列表也可以任意设置，最好不要和正常状态列表一样
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2), self.canvas.coords(self.hell3),
                    self.canvas.coords(self.hell4),self.canvas.coords(self.hell5),self.canvas.coords(self.hell6),
                    self.canvas.coords(self.hell7),self.canvas.coords(self.hell8),self.canvas.coords(self.hell9),
                    self.canvas.coords(self.hell10),self.canvas.coords(self.hell11),self.canvas.coords(self.hell12),
                    self.canvas.coords(self.hell13),self.canvas.coords(self.hell14),self.canvas.coords(self.hell15),
                    self.canvas.coords(self.hell16),self.canvas.coords(self.hell17),self.canvas.coords(self.hell18),
                    self.canvas.coords(self.hell19),self.canvas.coords(self.hell20),self.canvas.coords(self.hell21),
                    self.canvas.coords(self.hell22)]:
            reward = -1
            done = True
            s_ = [1,1,1,1]      # 一个特殊的列表，代表Terminal
        else:
            reward = 0
            done = False
        return s_, reward, done     # s_是一个list，里面包含四个角点的坐标，reward是int，done用以结束当前episode
    def render(self):
        time.sleep(0.1)
        self.update()
def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break
# if __name__ == '__main__':
#     env = Maze()
#     env.after(100, update)
#     env.mainloop()


#### 强化学习主体代码 ####
class QLearningTable:
    num = 0
    epsilion_step = 0.05
    def __init__(self, actions, n_features, learning_rate=0.1, e_greedy=0.5, memory_size=5000):
        self.actions = actions  # a list
        self.n_features = n_features
        self.lr = learning_rate
        # self.gamma = reward_decay
        # self.gamma_step = discount_rate_step
        self.epsilon = e_greedy
        self.memory_size = memory_size
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)     # q_table初始化,q_table是逐步扩充的，每探索到一个新的状态，就会增加一行

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))  # memory是一个二维数组


    def store_transition(self, s, a, r, s_, gamma):   # 传进来的s和s_是列表
        self.num += 1
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        error = self.error(str(s), a, r, str(s_), gamma)
        transition = np.hstack((s, [a, r], s_,[error]))   # 将多个列表组合成一个列表[s,a,r,s_,error] 共11个元素




        # replace the old memory with new memory
        if self.memory_counter < self.memory_size:
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition     # 当memory没有满时，正常顺序填充
        else:
            # index = self.memory_counter % self.memory_size
            self.memory[self.memory_size-1, :] = transition  # 当memory满时，每次固定替换最后的元素
        #if (self.num > 200) and (self.num % 5 == 0):   # 这样可以不必每次存储时都排序，浪费时间，只要在每次学习前排序就好。但是当memory满时，进行替换时，并不能保证memory里的顺序是排好的
        self.memory = self.memory[np.lexsort(-self.memory.T)]   #按照最后一个数字（误差）排序

        self.memory_counter += 1
        # print("----------memory--------")
        # print(self.memory)
        # print("----------memory--------")

    def choose_action(self, observation, episode):  # 传入的是字符串：'[....]'
        self.check_state_exist(observation)

        if episode % 50 ==0:
            self.epsilon += self.epsilion_step
            self.epsilon = min(self.epsilon, 0.99)

        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    # 返回误差用以排序
    def error(self,s, a, r, s_, gamma):      # s和s_是字符串 '[....]'
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != [1,1,1,1]:
            q_target = r + gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal

        # error = q_target - q_predict
        error = abs(q_predict - q_target)
        return error

    # 根据元素顺序进行采样,越靠后的，概率越低
    def sample_whole_row_from_2d_array(self,array):
        rows, cols = np.array(array).shape
        probabilities = 1 / np.arange(1, rows + 1)
        probabilities /= np.sum(probabilities)

        sampled_row = np.random.choice(np.arange(rows), p=probabilities)
        sampled_whole_row = array[sampled_row].copy()
        prob = probabilities[sampled_row].copy()

        #print('****************',probabilities)

        return sampled_whole_row,prob

    def learn(self, gamma, lr_start, beta):


        # sample batch memory from all memory
        #if self.memory_counter > self.memory_size:
            #sample_index = np.random.choice(self.memory_size, size=1)########self.batch_size
        sampled_elements, prob = self.sample_whole_row_from_2d_array(self.memory)
        sampled_elements = list(sampled_elements)       # 数组变列表，是为了让从memory提取出来的元素与q_table中的状态列表类型一致，'[....]',避免计算Q值索引q_table时出现错误
        #else:
            #sample_index = np.random.choice(self.memory_counter, size=1)#######self.batch_size
        #batch_memory = self.memory[sample_index, :]

        self.lr = lr_start / math.pow((self.memory_size * prob), beta)   # 随着采样概率的增加，学习率降低

        s = str(sampled_elements[:4])
        a = sampled_elements[4]
        r = sampled_elements[5]
        s_ = str(sampled_elements[6:10])


        self.check_state_exist(s_)

        q_predict = self.q_table.loc[s, a]
        if s_ != [1,1,1,1]:
            q_target = r + gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
        #self.q_table.loc[s, a] += 1 - (q_target * self.lr)  # update



    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
        print(self.q_table)
        #print(type(self.q_table))


#### 智能体学习实现 ####
def update():
    step = 0
    gamma_step = 0.05
    gamma = 0
    lr_start = 0.5
    beta = 0.3

    # env.move_grid()
    # env.move_grid1()
    for episode in range(500):
        # initial observation
        observation = env.reset()

        reward_sum = 0
        while True:
            # fresh env
            env.render()


            # RL choose action based on observation
            action = RL.choose_action(str(observation), episode)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action,episode)
            reward_sum += reward

            RL.store_transition(observation, action, reward, observation_, gamma)

            #if (step > 50) and (step % 10 == 0):
            if (step > 20):
                # RL learn from this transition
                #RL.learn(str(observation), action, reward, str(observation_), 1)
                RL.learn(gamma, lr_start, beta)
            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                break
            step += 1
            # if step % 50 ==0:
            #     print("----------step-------------")
            #     print(step)
            #     print("-------------step---------")
            #     time.sleep(3)

        # 增加折扣率
        if episode % 25 ==0:    # 最好随着episode的成倍增加而成倍增加
            gamma += gamma_step
            gamma = min(gamma, 0.9)


        if episode % 11 ==0:
            print("------------------")
            print(episode,reward_sum)
            print("------------------")

    # end of game
    print('game over')
    #env.destroy()
if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)), n_features=env.n_features)
    env.after(100, update)
    env.mainloop()







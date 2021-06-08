import numpy as np
import copy
import time
import matplotlib.pyplot as plt

up = 0
down = 1
left = 2
right = 3
all_actions=[up, down, left, right]

def act(current_state, action):
    if action == up:
        return [max(current_state[0] - 1, 0), current_state[1]]
    elif action == down:
        return [min(current_state[0] + 1, 3), current_state[1]]
    elif action == left:
        return [current_state[0], max(current_state[1] - 1, 0)]
    elif action == right:
        return [current_state[0], min(current_state[1] + 1, 11)]

def epsilon_greedy_choosing_action(Q_table, state, epsilon):
        if np.random.binomial(1, epsilon) == 1: # choose an action randomly
            action = np.random.choice(all_actions)
            while(state == act(state, action)):
                action = np.random.choice(all_actions)
            return action
        else: # choose the best action
            best_actions_list = []
            max_Q = max(Q_table[state[0]][state[1]])
            for action in all_actions:
                if abs(Q_table[state[0]][state[1]][action] - max_Q) < 1e-5:
                    best_actions_list.append(action)
            if len(best_actions_list) == 0:
                print('a')
            return np.random.choice(best_actions_list)

def draw(Q_table):
    policy = np.array(['x']*4*12).reshape(4,12)
    for i in range(4):
        for j in range(12):
            bestAction = np.array(Q_table[i][j]).argmax()
            if bestAction == up:
                policy[i][j] = "↑"
            elif bestAction == down:
                policy[i][j] = "↓"
            elif bestAction == right:
                policy[i][j] = '→'
            elif bestAction == left:
                policy[i][j] = "←"
    policy[3][11]="O"
    for i in range(len(policy)):
        for j in policy[i]:
            print(j, end='  ')
        print()
    print()

episode_num = -1
rounds = -1

def Qlearning(beginning_state, learning_rate = 0.1, gamma = 1.0, epsilon = 0.1):
    Q_table = np.array([0.0]*4*12*4).reshape(4,12,4)
    y = np.array([0.0]*episode_num) # for plt
    for count in range(rounds):
        #print(str(count+1)+' / '+str(rounds))
        for episode in range(episode_num):
            episode_reward = 0.0
            current_state = beginning_state
            death = False
            while current_state != [3,11]: # until terminal
                action = epsilon_greedy_choosing_action(Q_table, current_state, epsilon) # A
                new_state = act(current_state, action) # S'. Take action A, observe R, S'
                reward = -1.0 # R
                if new_state[0] == 3 and (new_state[1]>=1 and new_state[1]<=10): # fall into cliff
                    reward = -100.0
                    death = True
                episode_reward += reward
                something = np.max(Q_table[new_state[0]][new_state[1]])
                #                                            Q(S,A) +=       alpha  *  (R + gamma * max() - Q(S,A))
                Q_table[current_state[0]][current_state[1]][action] += learning_rate * (reward + gamma * something - Q_table[current_state[0]][current_state[1]][action])
                current_state = new_state
                if death:
                    break
            y[episode] += episode_reward
    y /= rounds
    draw(Q_table)
    return y

def Sarsa(beginning_state, learning_rate = 0.1, gamma = 1, epsilon = 0.1):
    Q_table = np.array([0.0]*4*12*4).reshape(4,12,4)
    y = np.array([0.0]*episode_num)
    for count in range(rounds):
        #print(str(count+1)+' / '+str(rounds))
        for episode in range(episode_num):
            total_reward = 0.0
            current_state = beginning_state
            action = epsilon_greedy_choosing_action(Q_table, current_state, epsilon)
            death = False
            while current_state != [3,11]:
                new_state = act(current_state, action)
                reward = -1.0
                if new_state[0] == 3 and (new_state[1]>=1 and new_state[1]<=10): # fall into cliff
                    reward = -100.0
                    death = True
                total_reward += reward
                #print(new_state)
                #something = np.max(Q_table[new_state[0]][new_state[1]])
                next_action = epsilon_greedy_choosing_action(Q_table, new_state, epsilon)
                something = Q_table[new_state[0]][new_state[1]][next_action]
                Q_table[current_state[0]][current_state[1]][action] += learning_rate * (reward + gamma * something - Q_table[current_state[0]][current_state[1]][action])
                current_state = new_state
                action = next_action
                if death:
                    break
            y[episode] += total_reward
    y /= rounds
    draw(Q_table)
    return y

def n_step_sarsa(beginning_state, n, learning_rate = 0.5, gamma = 1, epsilon = 0.1):
    Q_table = np.array([0.0]*4*12*4).reshape(4,12,4)
    for count in range(rounds):
        #print(str(count+1)+' / '+str(rounds))
        for episode in range(episode_num):
            current_state = beginning_state
            action = epsilon_greedy_choosing_action(Q_table, current_state, epsilon)
            T = np.infty
            t = 0
            reward_list=[-1]
            state_list=[current_state]
            action_list=[action]
            while True:
                if t < T:
                    new_state = act(state_list[-1], action_list[-1])
                    reward = -1.0
                    if new_state[0] == 3 and (new_state[1]>=1 and new_state[1]<=10): # fall into cliff
                        reward = -100.0
                    reward_list.append(reward)
                    state_list.append(new_state)
                    if (new_state[0] == 3 and new_state[1] == 11): # terminal
                        T = t + 1
                    else:
                        action = epsilon_greedy_choosing_action(Q_table, state_list[-1], epsilon)
                        action_list.append(action)
                tau = t - n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min(tau+n, T)+1):
                        G += gamma ** (i - tau + 1) * reward_list[ i - 1 ]
                    if tau + n < T:
                        G += gamma ** n * Q_table[state_list[tau + n][0]][state_list[tau + n][1]][action_list[tau + n]]
                    Q_table[state_list[tau][0]][state_list[tau][1]][action_list[tau]] += learning_rate * ( G - Q_table[state_list[tau][0]][state_list[tau][1]][action_list[tau]] )
                if tau == T - 1:
                    break
                t += 1
    draw(Q_table)
    return

def sarsa_lambda(beginning_state, Lambda, learning_rate = 0.05, gamma = 1, epsilon = 0.1):
    Q_table = np.array([-1.0]*4*12*4).reshape(4,12,4)
    for count in range(rounds):
        print(str(count+1)+' / '+str(rounds))
        for episode in range(episode_num):
            Z = np.array([0.0]*4*12*4).reshape(4,12,4)
            current_state = beginning_state
            action = epsilon_greedy_choosing_action(Q_table, current_state, epsilon)
            death = False
            while current_state != [3,11]:
                new_state = act(current_state, action)
                reward = -1
                if new_state[0] == 3 and (new_state[1]>=1 and new_state[1]<=10): # fall into cliff
                    reward = -100
                    death = True
                next_action = epsilon_greedy_choosing_action(Q_table, new_state, epsilon)
                delta = reward + gamma * Q_table[new_state[0]][new_state[1]][next_action] - Q_table[current_state[0]][current_state[1]][action]
                Z[current_state[0]][current_state[1]][action] += 1
                Q_table += learning_rate * delta * Z
                Z = gamma * Lambda * Z
                current_state = new_state
                action = next_action
                if death:
                    break
    draw(Q_table)
    return

if __name__ == '__main__':
    current_state = [3,0]

    episode_num = 500
    rounds = 1000

    #mode = 'Qlearning_Sarsa_Comparison'
    #mode = 'Nstep_Sarsa'
    mode = 'Sarsa_Lambda'

    if mode == 'Qlearning_Sarsa_Comparison':
        Q_learning = Qlearning(current_state, learning_rate = 0.1, gamma = 1.0, epsilon = 0.1)
        sarsa      =   Sarsa  (current_state, learning_rate = 0.1, gamma = 1.0, epsilon = 0.1)
        x = [i for i in range(episode_num)]
        plt.plot(x, Q_learning, color = 'r', label = 'Q-learning')
        plt.plot(x, sarsa, color = 'b', label = 'Sarsa')
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.legend()
        plt.show()
    elif mode == 'Nstep_Sarsa':
        n_step_sarsa(current_state, n=5, learning_rate = 0.1, gamma = 1.0, epsilon = 0.1)
    elif mode == 'Sarsa_Lambda':
        sarsa_lambda(current_state, Lambda = 1.0, learning_rate = 0.1, gamma = 1.0, epsilon = 0.1)
    else:
        print("Wrong mode")
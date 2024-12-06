import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES = 50000
LEARNING_RATE = .01
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":
    env_name = "CliffWalking-v0"
    env = gym.envs.make(env_name)
    env.reset(seed=1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with SARSA

        # Epsilon-greedy policy
        def epsilon_greedy_action(state, epsilon):
            if np.random.rand() < epsilon:
                return env.action_space.sample()
            else:
                actions = [Q_table[(state, a)] for a in range(env.action_space.n)]
                return np.argmax(actions)
        
        # Choose initial action from the initial state
        action = epsilon_greedy_action(obs, EPSILON)

        while not done:
            # Perform the action
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            # Choose the next action using the current epsilon-greedy policy
            next_action = epsilon_greedy_action(next_obs, EPSILON)

            # SARSA update
            current_q = Q_table[(obs, action)]
            next_q = Q_table[(next_obs, next_action)]
            Q_table[(obs, action)] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_q - current_q)

            obs = next_obs
            action = next_action

        # Decay epsilon more aggressively and ensure a lower min epsilon
        EPSILON = max(EPSILON * EPSILON_DECAY, 0.001)
            
        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open(f'Q_TABLE_SARSA.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################
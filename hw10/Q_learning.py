import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES = 100000
LEARNING_RATE = .001
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .9999


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
        # TODO: Replace the following with Q-Learning

        # while (not done):
        #     action = env.action_space.sample() # currently only performs a random action.
        #     obs,reward,terminated,truncated,info = env.step(action)
        #     episode_reward += reward # update episode reward
            
        #     done = terminated or truncated
        act_space = env.action_space.n
        
        while not done:
            # Construct a list of Q-values for all actions in current state
            action_values = [Q_table[(obs, a)] for a in range(act_space)]

            # Epsilon-greedy action selection
            if np.random.rand() < EPSILON:
                action = np.random.randint(act_space)
            else:
                action = np.argmax(action_values)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Q-learning update
            next_action_values = [Q_table[(next_obs, a)] for a in range(act_space)]
            best_next_action = np.argmax(next_action_values)
            td_target = reward + DISCOUNT_FACTOR * next_action_values[best_next_action] * (1 - done)
            Q_table[(obs, action)] += LEARNING_RATE * (td_target - Q_table[(obs, action)])
            
            obs = next_obs
            episode_reward += reward

        # Decay epsilon after each episode
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
    model_file = open(f'Q_TABLE_QLearning.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################
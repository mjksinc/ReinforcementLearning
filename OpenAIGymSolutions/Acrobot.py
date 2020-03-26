import numpy as np
import gym
import matplotlib.pyplot as plt
from datetime import datetime

# Import and initialize Acrobot Environment
env = gym.make('Acrobot-v1')
env.reset()

timeStamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):

    disc_array = np.array([10, 10, 10, 10, 1, 1])

    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)*disc_array
    num_states = np.round(num_states, 0).astype(int) + 1
    
    # Initialize Q table
    Q = np.random.uniform(low = -1, high = 1, 
                          size = (num_states[0], num_states[1], num_states[2], num_states[3], num_states[4], num_states[5], 
                                  env.action_space.n))
    
    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []
    
    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)/episodes
    
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        
        # Discretize state
        state_adj = (state - env.observation_space.low)*disc_array
        state_adj = np.round(state_adj, 0).astype(int)
    
        while done != True:   
            # Render environment for last five episodes
            if i >= (episodes - 20):
                env.render()
                
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3], state_adj[4], state_adj[5]]) 
            else:
                action = np.random.randint(0, env.action_space.n)
                
            # Get next state and reward
            state2, reward, done, info = env.step(action) 
            
            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)*disc_array
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            #Allow for terminal states
            if done and state2[0] >= 0.5:
                Q[state2_adj[0], state2_adj[1], state2_adj[2], state2_adj[3], state2_adj[4], state2_adj[5], action] = reward
                
            # Adjust Q value for current state
            else:
                delta = learning*(reward + discount*np.max(Q[state2_adj[0], state2_adj[1], state2_adj[2], state2_adj[3], state2_adj[4], state2_adj[5]]) - 
                                 Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3], state_adj[4], state_adj[5],action])
                Q[state_adj[0], state_adj[1], state_adj[2], state_adj[3], state_adj[4], state_adj[5],action] += delta
                                     
            # Update variables
            tot_reward += reward
            state_adj = state2_adj
        
        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction
        
        # Track rewards
        reward_list.append(tot_reward)
        
        if (i+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            
        if (i+1) % 100 == 0:    
            print('Episode {} Average Reward: {}'.format(i+1, ave_reward))
            
    env.close()
    
    return ave_reward_list

# Run Q-learning algorithm

LR_init = 0.2
DR_init = 0.9
EG_init = 0.95
Ep_init = 50000

rewards = QLearning(env, LR_init, DR_init, EG_init, 0, Ep_init)

# Plot Rewards
fig, ax = plt.subplots()

plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')

textstr = '\n'.join((f'LR = {LR_init}',f'DR = {DR_init}',f'EG ={ EG_init}'))
# these are matplotlib.patch.Patch properties   
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
ax.text(0.05, 0.95,textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

plt.savefig(f'Acrobot_rewards-{timeStamp}.jpg')
#plt.show()
plt.close()

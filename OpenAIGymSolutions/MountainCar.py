import numpy as np
import gym
import matplotlib.pyplot as plt
from datetime import datetime

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low)*\
                    np.array([10, 100]) # Get min/Max state space, multiple by decimal places to discretise
    num_states = np.round(num_states, 0).astype(int) + 1 #casts as integer, and adds one to account for total
    
    # Initialize Q table
    Q = np.random.uniform(low = -1, high = 1, 
                          size = (num_states[0], num_states[1], 
                                  env.action_space.n)) # initialise with random values, create based on two state arrays and the action array (3)
    
    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []
    
    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)/episodes #reduction rate with constant epsilon
    
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        
        # Discretize state
        state_adj = (state - env.observation_space.low)*np.array([10, 100]) #inital state
        state_adj = np.round(state_adj, 0).astype(int) 
    
        while done != True:   
            # Render environment for last five episodes
            if i >= (episodes - 20):
                True
                # env.render()
                
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]])
                # action is the max argument from Q table 
            else:
                action = np.random.randint(0, env.action_space.n)
            ## will I take the expeted action, or will I a random action? (Explore)

            # Get next state and reward
            #take action and get the next step
            state2, reward, done, info = env.step(action) 
            
            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            #Allow for terminal states
            if done and state2[0] >= 0.5: #is the state complete?
                Q[state_adj[0], state_adj[1], action] = reward #if yes,  set this position to rewarsa
                
            # Adjust Q value for current state
            else:
                delta = learning*(reward + 
                                 discount*np.max(Q[state2_adj[0], 
                                                   state2_adj[1]]) - 
                                 Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta
                                     
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

timeStamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

# Run Q-learning algorithm

LR_init = 0.2
DR_init = 0.9
EG_init = 0.8
Ep_init = 200

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

plt.savefig(f'MountainCar_rewards-{timeStamp}.jpg')
#plt.show()
plt.close()


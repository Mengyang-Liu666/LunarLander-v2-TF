import numpy as np

# For more detailed explainations, see "custom_control_API.md" in the folder.

class TDControl:
    # disc: a discretizer is required to solve this problem with tabular methods
    def __init__(self, state_shape=[4,4,4,4,4,4,2,2], action_num=4, algorithm="Q-learning",
                gamma=0.9, epsilon=None, alpha=None, disc=None):
        self.state_shape = state_shape
        self.action_num = action_num
        # ====Initialize Q(s,a)====
        self.Qtable = np.zeros(tuple(self.state_shape)+(self.action_num,))
        self.algorithm = algorithm
        self.gamma = gamma

        if (isinstance(epsilon, (int, float))):
        # Constant epsilon
            self.epsilon = lambda x : epsilon
        else: 
            self.epsilon = epsilon
        
        if (isinstance(alpha, (int, float))):
        # Constant alpha
            self.alpha = lambda x : alpha
        else: 
            self.alpha = alpha

        self.disc = disc
    
    # Generate the optimal policy
    def generate_policy(self):
        policy = np.zeros(tuple(self.state_shape))
        it = np.nditer(policy, flags=['multi_index'])
        while not it.finished:
            policy[it.multi_index] = np.argmax(self.Qtable[it.multi_index])
            it.iternext()
        return policy

    # No exploration, predict on discretized states
    def predict_d(self, obs):
        return np.argmax(self.Qtable[tuple(obs)])

    # Encapsulated method
    def predict(self, obs):
        return self.predict_d(self.disc(obs))
    
    # Epsilon greedy
    def egreedy(self, obs, epi=0):
        if np.random.random() > self.epsilon(epi):
              return self.predict_d(obs)
        else:
              return np.random.randint(0, self.action_num)
    
    # Need a custom step to observe the discretized states
    def step(self, env, action):
        obs, reward, done, info = env.step(action)
        return self.disc(obs), reward, done, info

    # Training
        # epoch_size: a report of the current average total reward 
        # is generated after such episodes.
    def train(self, env, max_episode_number=150000, epoch_size = 1000, tol = 0, tol_epoch = 12, verbose=False):
        # Below for report
        epoch_step_list = []
        epoch_reward_list = []
        epoch_policy_change_list = []
        epoch_step = 0
        epoch_reward = 0
        prev_reward = 0
        # Make a policy backup for comparison.
        prev_policy = self.generate_policy()
        # Set up a boolean matrix to capture modified states.
        # Each entry represents a state. True if the state is modified.
        full_modified_S = np.zeros(tuple(self.state_shape), dtype=bool)

        # ====Loop for each episode====
        for epi in range(max_episode_number):
            # Continued in SARSA_1_episode() or Qlearning_1_episode()
            if self.algorithm=="Q-learning":
                # Q-learning
                step_num, episode_reward, modified_S = self.Qlearning_1_episode(env, epi=0)
            else:
                # SARSA
                step_num, episode_reward, modified_S = self.SARSA_1_episode(env, epi=0)
                
            epoch_step += step_num
            epoch_reward += episode_reward
            for idx in modified_S:
                if not full_modified_S[tuple(idx)]:
                    full_modified_S[tuple(idx)] = True
            
            # End of each epoch
            if ((epi+1) % epoch_size) == 0:
                epoch_step = epoch_step / epoch_size
                epoch_step_list.append(epoch_step)
                epoch_reward = epoch_reward / epoch_size
                epoch_reward_list.append(epoch_reward)

                # Count the changed policies
                changed_policies = 0
                it = np.nditer(full_modified_S, flags=['multi_index'])
                while not it.finished:
                    # Handle duplicates
                    if full_modified_S[it.multi_index]:
                        prev_p = prev_policy[it.multi_index]
                        new_p = int(np.argmax(self.Qtable[it.multi_index]))
                        if not prev_p == new_p:
                            changed_policies += 1
                    it.iternext()
                epoch_policy_change_list.append(changed_policies)

                # Print the log
                if verbose:
                    print("episode: %d, avg episode steps: %.4f, avg episode reward: %.4f, abs reward diff: %.4f, changed policies: %d" 
                          % (epi+1, epoch_step, epoch_reward, abs(epoch_reward-prev_reward), changed_policies)) 

                # Set for the records in the next epoch
                prev_reward = epoch_reward
                epoch_step = 0
                epoch_reward = 0
                prev_policy = self.generate_policy()
                full_modified_S = np.zeros(tuple(self.state_shape), dtype=bool)
                
                # Termination condition test
                epoch_num = len(epoch_policy_change_list)
                if epoch_num >= tol_epoch + 1:
                    terminate = True
                    for i in range(tol_epoch):
                        if epoch_policy_change_list[epoch_num-i-1] > tol:
                            terminate = False
                            break
                    if terminate:
                        break
        print("Done")
        return epoch_step_list, epoch_reward_list, epoch_policy_change_list
    
    # SARSA algorithm with "Loop for each episode"       
        # epi: number of current episode (for epsilon-greedy and alpha) 
    def SARSA_1_episode(self, env, epi=0):
        step_num = 0
        episode_reward = 0
        # For tracking policy changes, set cannot hash lists
        modified_S = []
        # ====Initialize S====
        S = self.disc(env.reset())
        done = False
        # ====Choose A from S using policy derived from Q (epsilon-greedy)====
        A = self.egreedy(S, epi)
        # ====Loop for each step of episode:====
        while not done:
            # ====Take action A, observe R,S'====
            S_new, R, done, info= self.step(env, A)
            # ====Choose A' from S' using policy derived from Q (epsilon-greedy)====
            A_new = self.egreedy(S_new, epi)
            # ====(Follow the SARSA update rule)====
            TD_error =  R + self.gamma * self.Qtable[tuple(S_new)+(A_new,)] - self.Qtable[tuple(S)+(A,)]
            self.Qtable[tuple(S)+(A,)] = self.Qtable[tuple(S)+(A,)] + self.alpha(epi) * TD_error
            # Add S to the set for tracking, since this value is modified.
            modified_S.append(S)
            # ====S <- S', A <- A'====
            S = S_new
            A = A_new
            # Count step numbers and collect reward
            step_num += 1
            episode_reward += R
            
        return step_num, episode_reward, modified_S
    
    # Q-learning algorithm with "Loop for each episode"   
        # epi: number of current episode (for epsilon-greedy and alpha) 
    def Qlearning_1_episode(self, env, epi=0):
        step_num = 0
        episode_reward = 0
        # For tracking policy changes, set cannot hash lists
        modified_S = []
        # ====Initialize S====
        S = self.disc(env.reset())
        done = False
        # ====Loop for each step of episode:====
        while not done:
            # ====Choose A from S using policy derived from Q (epsilon-greedy)====
            A = self.egreedy(S, epi)
            # ====Take action A, observe R,S'====
            S_new, R, done, info = self.step(env, A)
            # ====(Follow the Q-learning update rule)====
            TD_error =  R + self.gamma * max(self.Qtable[tuple(S_new)]) - self.Qtable[tuple(S)+(A,)]
            self.Qtable[tuple(S)+(A,)] = self.Qtable[tuple(S)+(A,)] + self.alpha(epi) * TD_error
            # Add S to the set for tracking, since this value is modified.
            modified_S.append(S)
            # ====S <- S'====
            S = S_new
            # Count step numbers and collect reward
            step_num += 1
            episode_reward += R
            
        return step_num, episode_reward, modified_S
## Table of Contents:
*   [`TDControl` class](#class)
    *   [Arguments](#arguments)
    *   [Attributes](#attributes)
*   [Encapsulated methods:](#E_methods)
    *   [`train`](#train)
    *   [`predict`](#predict)
*   [Miscellaneous](#Misc)

<br>

## `TDControl` class: <a class="anchor" id="class"></a>
       td.TDControl(
            state_shape=[4,4,4,4,4,4,2,2], action_num=4
            algorithm="Q-learning",gamma=0.9, epsilon=None,
            alpha=None, disc=None
        )
*   ### Arguments: <a class="anchor" id="arguments"></a>
    *   **state_shape**: a list of numbers of unique values for each feature of a state, used to construct Q-table.
        *   Default selection is used specifically for Lunar Lander that uses 4 bins for the first 6 continuous features.
    *   **action_num**: number of actions in each state, used to construct Q-table.
    *   **algorithm**: name of algorithm to use. Only 2 algorithms are implemented: SARSA and Q-learning. Default is Q-learning. Any input otherthan `"Q-learning"` will leads to use SARSA.
    *   **gamma**: the discounting factor $\gamma$ used in bootstrapping for finite estimate of return.
    *   **epsilon**: the $\varepsilon$(exploration) to use in $\varepsilon$-greedy. Two types of input are accepted:
        *   If a contant is inputed, constant $\varepsilon$=`epsilon` is used.
        *   If a function is input, $\varepsilon$=`epsilon(t)` is used, where `t` is the current episode number.
    *   **alpha**: the learning rate $\alpha$ to use in updating. Two types of input are accepted:
        *   If a contant is inputed, constant $\alpha$=`alpha` is used.
        *   If a function is input, $\alpha$=`alpha(t)` is used, where `t` is the current episode number.
    *   **disc**: the discretizer to use:
        *   Expecting `disc` that takes in a vanilla(can be continuous) state from the environment and outputs a discretized state.

<br>

*   Attributes: <a class="anchor" id="attributes"></a>
    *   **Qtable**: the learned Q-table. Initialized with all zeros.
    *   **state_shape**: the state shape of the environment.
    *   **action_num**: number of actions.
    *   **algorithm**: name of the algorithm used.
    *   **gamma**: discounting factor $\gamma$ used.
    *   **epsilon**: function that takes in an episode number and outputs the $\varepsilon$ used in $\epsilon$-greedy.
    *   **alpha**: function that takes in an episode number and outputs the learning rate $\alpha$.
    *   **disc**: the discretizer used.

<br>

## Encapsulated Methods: <a class="anchor" id="E_methods"></a>
*   ### `train` Method: <a class="anchor" id="train"></a>
           td.train(env, max_episode_number=150000, epoch_size = 1000, 
                tol = 0.05, tol_epoch = 6, verbose=False)
    *   Train with the selected `algorithm`.
    *   Training will stop if any one of the two criteria is satisfied:
        *   The number of episodes used exceeds `max_episode_number`.
        *   The number of changes in the policy for the recent `tol_epoch` epochs are all smaller than or equal to `tol`.
    *   Returns a tuple of `(epoch_step_list, epoch_reward_list)`:
        *   `epoch_step_list`: a list of average steps of each episodes in each epoch.
        *   `epoch_reward_list`: a list of average total reward of each epsiodes in each epoch.
        *   `epoch_policy_change_list`: a list of changes in policies from Q-table in each epoch.
            *   Comparisons are done between before the epoch and after the epoch.
    
    <br>

    *   **env**: the environment to use.
    *   **max_episode_number**: the maximum numbers of episodes to train.
    *   **epoch_size**: the number of episodes to count as an epoch.
    *   **tol**: the tolerance of number of changes in the policy in each epoch.
    *   **tol_epoch**: the number of epochs to last for stop training, if the number of changes in the policy are all smaller than or equal to `tol`.
    *   **verbose**: if `True`, the episode number, average steps, average total reward, the absolute change in average total rewards and the changes in the policy are reported at the end of each training epoch.

<br>

*   ### `predict` Method: <a class="anchor" id="predict"></a>
           td.predict(obs)
    *   Predict the action based on the given state (can be continuous), with the internal Q-table.
    *   Returns an integer of the action to execute.

    <br>

    *   **obs**: the given state to predict.

<br>

## Miscellaneous: <a class="anchor" id="Misc"></a>
*   `custom_tdcontrol.py` depends on `numpy`.
*   Other methods are also implemented to support the encapulated methods:
    *   `SARSA_1_episode`: learning following SARSA algorithm in 1 episode.
    *   `Qlearning_1_episode`: learning following Q-learning algorithm in 1 episode.
    *   `predict_d`: predict the action based on a discretized state.
    *   `egreedy`: $\varepsilon$-greedy method to choose actions, taking in a discretized state.
    *   `step`: step the environment by 1 step and apply discretizer for return.
    *   `generate_policy`: generates the optimal policy from the Q-table. Used for compare the policy changes and called once for each epoch.

from operator import itemgetter # getting max q from a dict
import random
import numpy
import matplotlib.pyplot as plt

## Use the simple mdp to formalise the simple problem of using
## different ways to access a program.

class mdp():

    def __init__(self, success_prob):

        # All actions that the agent may execute. Note that generally
        # we wish all actions to be available at all states. It is up
        # to the agent to figure out (using reinforcement learning)
        # what actions are useful and what are not in any given state.
        self.actions = ["move mouse to icon",
                        "move mouse to menu",
                        "move mouse to menu item",
                        "click mouse",
                        "press key"]

        # Q table will be populated with state-action-value triplets.
        # It is a dictionary, with state as a key, and a new
        # dictionary as its value. This new dictionary contains
        # actions as keys, and state-action values as values.
        self.q = {}

        # The state is a dictionary that can be used to represent the
        # environment. The dictionary format allows a human-readable
        # representation of the different components of the full
        # state. See reset() to get a glimpse of the state space.
        self.state = {}
        
        # RL parameters.
        self.alpha = 0.1 # learning rate
        self.epsilon = 0.1 # explore vs exploit
        self.gamma = 0.9 # future reward discount  
        
        self.success_prob = success_prob
        print(f"Successful prob of press key: {self.success_prob}")

        # Initialise the environment.
        self.reset()

    # In order to learn, the agent needs to do the task multiple
    # times. This function resets the environment to its starting
    # stage. Note that the Q table is not re-initialised: this allows
    # the agent to carry information from previous task iterations.
    def reset(self):
        self.state = {
            "task": "start",
            "mouse": "not moved",    
            "key": "not pressed",
            "menu": "closed"}

        # Start with no previous state, no current state, no previous
        # action, and no current action.
        self.previous_state = None
        self.current_state = None
        self.previous_action = None
        self.action = None

    # The logic of the environment. Given an action and a state,
    # change the state.
    def update_environment(self):

        if self.action == "move mouse to icon":        
            self.state["mouse"] = "on icon"
        if self.action == "move mouse to menu":
            self.state["mouse"] = "on menu"
        if self.action == "move mouse to menu item" and self.state["menu"] == "open":
            self.state["mouse"] = "on menu item"

        if self.action == "click mouse":
            if self.state["mouse"] == "on menu":
                self.state["menu"] = "open"
            if self.state["mouse"] == "on icon":
                self.state["task"] = "done"
            if self.state["mouse"] == "on menu item":
                self.state["task"] = "done"
        
        if self.action == "press key":
            if random.random() < self.success_prob:
                self.state["task"] = "done"
            else:
                self.state["key"] = "pressed"

    # Calculate the reward that the agent gets, given a state. Punish
    # for time consuming actions, reward for getting the task done.
    def calculate_reward(self):
        self.reward = 0
        if self.action == "click mouse":
            self.reward += -1
        if self.action == "move mouse to icon":
            self.reward += -2
        if self.action == "move mouse to menu" or self.action == "move mouse to menu item":
            self.reward += -2
        if self.state["task"] == "done":
            self.reward += 5
        if self.action == "press key":
            self.reward += -2

    # Epsilon greedy action selection. Choose the action with best
    # utility in the present state, except for a certain probability
    # (epsilon), choose a completely random action (which may also be
    # the best action).
    def choose_action_epsilon_greedy(self):
        if random.random() < self.epsilon:
            self.action = random.choice(self.actions)
            return "randomly" # for output (debug) purposes
        else:
            self.action = max(self.q[self.current_state].items(), key = itemgetter(1))[0]
            return "greedily"

    # For output (debug) purposes, cleanly print the state dictionary.
    def print_state(self):
        print("Current state:")
        for s in self.state:
            print("   ", s, ":", self.state[s])
            print()

    # For output (debug) purposes, cleanly print the nested Q
    # dictionary. Optionally, only print a given state's
    # action-values.
    def print_q(self, state = None):
        return round(self.q[repr(state)]['press key'], 2)
#         for s in self.q:
#             print(s)
#             if state == None or repr(state) == s:
#                 for a in self.actions:
#                     print("   ", a, ":", round(self.q[s][a],2))
            
    # Update the q-table. Q learning can only be called after an
    # action has been taken, the environment transitioned, and the
    # reward observed. The easiest to do this is at the start of the
    # new iteration, because it lets us get the current max Q -value,
    # which can be then chained to the previous state-action Q-value,
    # which is the one that is to be updated.
    def update_q_learning(self):
        # Only learn if there is a previous action. If this is a start
        # of a new episode after a self.reset(), cannot learn yet.
        if self.previous_action != None:
            previous_q = self.q[self.previous_state][self.previous_action]
            next_q = max(self.q[self.current_state].items(), key = itemgetter(1))[1]
            self.q[self.previous_state][self.previous_action] = \
                previous_q + self.alpha * (self.reward + self.gamma * next_q - previous_q)

    # Because Q-learning function updates Q-values only after seeing
    # the next state, it is not useful for training the last state of
    # an episode. For this purpose, before resetting the environment,
    # call td-update, which works like Q-learning but does not
    # consider the next state (which does not exist).
    def update_q_td(self):
        previous_q = self.q[self.current_state][self.action]
        self.q[self.current_state][self.action] = \
                previous_q + self.alpha * (self.reward - previous_q)
    

    # Do one iteration of the model.
    #
    # print_progress: print debug information
    #
    # force_action: instead of epsilon greedy action selection, force an action
    def iterate_model(self, print_progress = False, force_action = None):
        self.previous_state = self.current_state
        self.previous_action = self.action
        # Make the current state to string with repr(), so that the
        # current state can be accessed as a dictionary keyword. Note
        # that self.state and self.current_state are both the same
        # state, but in different data structures (dict vs string).
        self.current_state = repr(self.state)
        # Add the current state to the q table if it is not there yet.
        # Initialise all state-action pair values to 0.        
        if self.current_state not in self.q:
            self.q[self.current_state] = {}
            # Add all actions as possible pairs if this new state.
            for a in self.actions:
                self.q[self.current_state][a] = 0.0

        # Update the Q table based on previous state and previous
        # action, and the best (optimal) action from the current
        # state. If this is the first state (no previous state
        # exists), the function does not do anything.
        self.update_q_learning()

        # Choose action, store the explore or exploit as a string for
        # outputting (debug). If the action is forced, take that action.
        if force_action:
            self.action = force_action
            how = "forced"
        else:
            how = self.choose_action_epsilon_greedy()

        if print_progress: print("Took action <", self.action, "> ", how, sep = '')
        # Based on the action and the current state, update the environment.
        self.update_environment()
        # Based on the new state after the update, observe the reward.
        # Note that learning only happens after the fact: either in
        # the beginning of next iteration, when the next state is
        # known (necessary for Q-learning), or if this is the last
        # state before reset, at the end of this iteration.
        self.calculate_reward()
        if print_progress: print("Reward =", self.reward)

        # If the state is end state, update using TD learning, because
        # Q learning only happens when there is a previous state,
        # which reset removes.
        if self.state["task"] == "done":
            if print_progress:
                print("Task done!")
                self.print_state()
            self.update_q_td()
            # Reset the environment. The Q-table is retained as
            # memory, but the task starts again.
            self.reset()

# Create the agent.
agent = mdp(0.9)
print("Training the model...")
# Learn the model multiple times. This takes some seconds to run.
# numbers = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
q_press_key = []
for i in range(0, 1500):
    agent.iterate_model()
    q_press_key.append(agent.print_q({'task': 'start', 'mouse': 'not moved', 'key': 'not pressed', 'menu': 'closed'}))

fig = plt.figure()
ax = fig.add_subplot(111)
# print(range(0,10))
ax.plot(range(0, 1500), q_press_key)
plt.xlabel('the number of iterations')
plt.ylabel('the Q-value of press key')
plt.title('How Q-value changes as iterations increase with probability 0.9')
plt.show()
# Print the learned Q-values?
# agent.print_q({'task': 'start', 'mouse': 'not moved', 'key': 'not pressed', 'menu': 'closed'})

# Make one iteration. Useful if you wish to step the agent to see how it behaves.
agent.epsilon = 0 # just exploit to see the optimal behaviour and no
                  # exploration; remember to reset to > 0 if you wish
                  # the agent to explore
print("---- Testing the agent -----")
# agent.reset()
# agent.print_state()
# agent.print_q(agent.state)
# agent.iterate_model(print_progress = True)
# agent.print_state()
# agent.print_q(agent.state)
# agent.iterate_model(print_progress = True)


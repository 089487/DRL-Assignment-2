import copy
import random
import math
import numpy as np
from approximator import NTupleApproximator

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4)]

    def fully_expanded(self):
		# A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self,approximator,iterations=500, exploration_constant=1.41, rollout_depth=10):
        self.iterations = iterations
        self.c = exploration_constant  # Balances exploration and exploitation
        self.rollout_depth = rollout_depth
        self.approximator = approximator
    def create_env_from_state(self, state, score):
        """
        Creates a deep copy of the environment with a given board state and score.
        """
        new_env = copy.deepcopy(self.env)
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent_visits)/child_visits) to select the child
        """for x in node.children.items():
            print(x[1].total_reward/x[1].visits,self.c * np.sqrt(np.log(node.visits)/x[1].visits))"""
        #max_reward = max([x.total_reward/x.visits for x in node.children.values()])
        #min_reward = min([x.total_reward/x.visits for x in node.children.values()])
        max_key = max(node.children.items(), key=lambda x: x[1].total_reward/x[1].visits + self.c * np.sqrt(np.log(node.visits)/x[1].visits))[0]
        return node.children[max_key]
    def get_best_action(self,sim_env,legal_moves=None):
        val_best = float('-inf')
        best_action = None
        if not legal_moves:
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
        for a in legal_moves:
            val_a = 0
            for _ in range(4):
                temp_env = copy.deepcopy(sim_env)
                new_board, score, _done, _ = temp_env.step(a)
                
                if _done:
                    score -= 10000
                val_a += self.approximator.value(new_board) + score
            if val_a > val_best:
                val_best = val_a
                best_action = a
        return best_action
    def evaluate(self,sim_env,action,iteration=4):
        val = 0
        for _ in range(iteration):
            temp_env = copy.deepcopy(sim_env)
            new_board, score, _done, _ = temp_env.step(action)
            if _done:
                score -= 10000
            val += self.approximator.value(new_board) + score
        return val/iteration
    def rollout(self, sim_env, action):
        # TODO: Perform a random rollout from the current state up to the specified depth.
        """rollout_reward = 0
        done = sim_env.is_game_over()
        while depth and not done:
            action = self.get_best_action(sim_env)
            _board,reward,done,_ = sim_env.step(action)
            depth -= 1
        #print(sim_env.score,value_0 - self.approximator.value(sim_env.board))"""
        return self.evaluate(sim_env,action)

    def backpropagate(self, node, reward):
        # TODO: Propagate the reward up the tree, updating visit counts and total rewards.
        while node :
            node.visits += 1
            node.total_reward += reward
            node = node.parent


    def run_simulation(self, root,env):
        node = root
        sim_env = copy.deepcopy(env)
        # TODO: Selection: Traverse the tree until reaching a non-fully expanded node.
        done = sim_env.is_game_over()
        l2 = [i for i in node.untried_actions if sim_env.is_move_legal(i)]
        while not done and (node.fully_expanded() or not l2):
            node = self.select_child(node)
            _,_,done,_ = sim_env.step(node.action)
            l2 = [i for i in node.untried_actions if sim_env.is_move_legal(i)]

        # TODO: Expansion: if the node has untried actions, expand one.
        # expand node self.board, self.score, done, {} = env.step()
        # Rollout: Simulate a random game from the expanded node.
        # Backpropagation: Update the tree with the rollout reward.
        l = [i for i in node.untried_actions if sim_env.is_move_legal(i)]
        if l:
            action = self.get_best_action(sim_env,l)
            node.untried_actions.remove(action)
            node.children[action] = TD_MCTS_Node(node,action)
            node = node.children[action]
            rollout_reward = self.rollout(sim_env,action)
        else:
            #assert(sim_env.is_game_over())
            rollout_reward = sim_env.score - 10000
        self.backpropagate(node, rollout_reward)
        return

    def best_action_distribution(self, root):
        '''
        Computes the visit count distribution for each action at the root node.
        '''
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution


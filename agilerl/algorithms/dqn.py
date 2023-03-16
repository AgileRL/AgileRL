import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agilerl.networks.evolvable_mlp import EvolvableMLP

class DQN():
    """The DQN algorithm class. DQN paper: https://arxiv.org/abs/1312.5602

    :param n_states: State observation dimension
    :type n_states: int
    :param n_actions: Action dimension
    :type n_actions: int
    :param one_hot: One-hot encoding, used with discrete observation spaces
    :type one_hot: bool
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param h_size: Network hidden layers size, defaults to [64,64]
    :type h_size: List[int], optional
    :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
    :type batch_size: int, optional
    :param lr: Learning rate for optimizer, defaults to 1e-4
    :type lr: float, optional
    :param learn_step: Learning frequency, defaults to 5
    :type learn_step: int, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param tau: For soft update of target network parameters, defaults to 1e-3
    :type tau: float, optional
    :param mutation: Most recent mutation to agent, defaults to None
    :type mutation: str, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    """
    def __init__(self, n_states, n_actions, one_hot, index=0, h_size=[64,64], batch_size=64, lr=1e-4, learn_step=5, gamma=0.99, tau=1e-3, mutation=None, device='cpu'):
        self.algo = 'DQN'
        self.n_states = n_states
        self.n_actions = n_actions
        self.one_hot = one_hot
        self.h_size = h_size
        self.batch_size = batch_size
        self.lr = lr
        self.learn_step = learn_step
        self.gamma = gamma
        self.tau = tau
        self.mut = mutation
        self.device = device

        self.index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]

        # model
        self.net_eval = EvolvableMLP(num_inputs=n_states, num_outputs=n_actions, hidden_size=h_size, device=self.device).to(device)
        self.net_target = EvolvableMLP(num_inputs=n_states, num_outputs=n_actions, hidden_size=h_size, device=self.device).to(device)

        self.optimizer = optim.Adam(self.net_eval.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def getAction(self, state, epsilon=0):
        """Returns the next action to take in the environment. Epsilon is the probability of taking a random action, used for exploration.
        For epsilon-greedy behaviour, set epsilon to 0.

        :param state: State observation, or multiple observations in a batch
        :type state: float or List[float]
        :param epsilon: Probablilty of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        """
        state = torch.from_numpy(state).float().to(self.device)

        if self.one_hot:
            state = nn.functional.one_hot(state.long(), num_classes=self.n_states).float().squeeze()
        
        if len(state.size())<2:
            state = state.unsqueeze(0)

        self.net_eval.eval()
        with torch.no_grad():
            action_values = self.net_eval(state)
        self.net_eval.train()

        # epsilon-greedy
        if random.random() < epsilon:
            action = np.random.randint(0, self.n_actions, size=state.size()[0])
        else:
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)

        return action

    def learn(self, experiences):
        """Updates agent network parameters to learn from experiences.

        :param experiences: List of batched states, actions, rewards, next_states, dones in that order.
        :type state: List[torch.Tensor[float]]
        """
        states, actions, rewards, next_states, dones = experiences

        if self.one_hot:
            states = nn.functional.one_hot(states.long(), num_classes=self.n_states).float().squeeze()
            next_states = nn.functional.one_hot(next_states.long(), num_classes=self.n_states).float().squeeze()

        q_target = self.net_target(next_states).detach().max(axis=1)[0].unsqueeze(1)
        y_j = rewards + self.gamma * q_target * (1 - dones)          # target, if terminal then y_j = rewards
        q_eval = self.net_eval(states).gather(1, actions.long())

        # loss backprop
        loss = self.criterion(q_eval, y_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update target network
        self.softUpdate()

    def softUpdate(self):
        """Soft updates target network.
        """
        for eval_param, target_param in zip(self.net_eval.parameters(), self.net_target.parameters()):
            target_param.data.copy_(self.tau*eval_param.data + (1.0-self.tau)*target_param.data)

    def test(self, env, max_steps=500, loop=3):
        """Returns mean test score of agent in environment with epsilon-greedy policy.

        :param env: The environment to be tested in
        :type env: Gym-style environment
        :param max_steps: Maximum number of testing steps, defaults to 500
        :type max_steps: int, optional
        :param loop: Number of testing loops/epsiodes to complete. The returned score is the mean over these tests. Defaults to 3
        :type loop: int, optional
        """
        with torch.no_grad():
            rewards = []
            for i in range(loop):
                state = env.reset()[0]
                score = 0
                for idx_step in range(max_steps):
                    action = self.getAction(state, epsilon=0)
                    state, reward, done, _, _ = env.step(action)
                    score += reward
                rewards.append(score)
        mean_fit = np.mean(rewards)
        self.fitness.append(mean_fit)
        return mean_fit

    def clone(self, index=None):
        """Returns cloned agent identical to self.

        :param index: Index to keep track of agent for tournament selection and mutation, defaults to None
        :type index: int, optional
        """
        if index is None:
            index = self.index

        clone = type(self)(n_states=self.n_states,
                            n_actions=self.n_actions,
                            one_hot=self.one_hot,
                            index=index, 
                            h_size = self.h_size, 
                            batch_size=self.batch_size,
                            lr=self.lr,
                            gamma=self.gamma,
                            tau=self.tau,
                            device=self.device,
                           )
                           
        clone.net_eval = self.net_eval.clone().to(self.device)
        clone.net_target = self.net_target.clone().to(self.device)
        clone.optimizer = optim.Adam(clone.net_eval.parameters(), lr=clone.lr)
        clone.fitness = copy.deepcopy(self.fitness)
        clone.steps = copy.deepcopy(self.steps)
        clone.scores = copy.deepcopy(self.scores)

        return clone
    
    def saveCheckpoint(self, path):
        """Saves a checkpoint of agent properties and network weights to path.

        :param path: Location to save checkpoint at
        :type path: string
        """
        torch.save({
                    'net_eval_init_dict': self.net_eval.init_dict,
                    'net_eval_state_dict': self.net_eval.state_dict(),
                    'net_target_init_dict': self.net_target.init_dict,
                    'net_target_state_dict': self.net_target.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'batch_size': self.batch_size,
                    'lr': self.lr,
                    'gamma': self.gamma,
                    'tau': self.tau,
                    'mutation': self.mut,
                    'index': self.index, 
                    'scores': self.scores,
                    'fitness': self.fitness,
                    'steps': self.steps,
                    }, path)
        
    def loadCheckpoint(self, path):
        """Loads saved agent properties and network weights from checkpoint.

        :param path: Location to load checkpoint from
        :type path: string
        """
        checkpoint = torch.load(path)
        self.net_eval = EvolvableMLP(**checkpoint['net_eval_init_dict'])
        self.net_eval.load_state_dict(checkpoint['net_eval_state_dict'])
        self.net_target = EvolvableMLP(**checkpoint['net_target_init_dict'])
        self.net_target.load_state_dict(checkpoint['net_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.batch_size = checkpoint['batch_size']
        self.lr = checkpoint['lr']
        self.gamma = checkpoint['gamma']
        self.tau = checkpoint['tau']
        self.mut = checkpoint['mutation']
        self.index = checkpoint['index']
        self.scores = checkpoint['scores']
        self.fitness = checkpoint['fitness']
        self.steps = checkpoint['steps']
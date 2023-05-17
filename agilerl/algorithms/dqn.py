import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.networks.evolvable_cnn import EvolvableCNN


class DQN():
    """The DQN algorithm class. DQN paper: https://arxiv.org/abs/1312.5602

    :param state_dim: State observation dimension
    :type state_dim: int
    :param action_dim: Action dimension
    :type action_dim: int
    :param one_hot: One-hot encoding, used with discrete observation spaces
    :type one_hot: bool
    :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
    :type index: int, optional
    :param net_config: Network configuration, defaults to mlp with hidden size [64,64]
    :type net_config: dict, optional
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
    :param double: Use double Q-learning, defaults to False
    :type double: bool, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        one_hot,
        index=0,
        net_config={
            'arch': 'mlp',
            'h_size': [
            64,
            64]},
            batch_size=64,
            lr=1e-4,
            learn_step=5,
            gamma=0.99,
            tau=1e-3,
            mutation=None,
            double=False,
            device='cpu'):
        self.algo = 'DQN'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.one_hot = one_hot
        self.net_config = net_config
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

        self.double = double

        # model
        if self.net_config['arch'] == 'mlp':      # Multi-layer Perceptron
            self.actor = EvolvableMLP(
                num_inputs=state_dim[0],
                num_outputs=action_dim,
                hidden_size=self.net_config['h_size'],
                device=self.device).to(
                self.device)
            self.actor_target = EvolvableMLP(
                num_inputs=state_dim[0],
                num_outputs=action_dim,
                hidden_size=self.net_config['h_size'],
                device=self.device).to(
                self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())

        elif self.net_config['arch'] == 'cnn':    # Convolutional Neural Network
            self.actor = EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config['c_size'],
                kernal_size=self.net_config['k_size'],
                stride_size=self.net_config['s_size'],
                hidden_size=self.net_config['h_size'],
                device=self.device).to(
                self.device)
            self.actor_target = EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config['c_size'],
                kernal_size=self.net_config['k_size'],
                stride_size=self.net_config['s_size'],
                hidden_size=self.net_config['h_size'],
                device=self.device).to(
                self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())

        self.optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
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
            state = nn.functional.one_hot(
                state.long(), num_classes=self.state_dim[0]).float().squeeze()

        if len(state.size()) < 2:
            state = state.unsqueeze(0)

        # epsilon-greedy
        if random.random() < epsilon:
            action = np.random.randint(0, self.action_dim, size=state.size()[0])
        else:
            self.actor.eval()
            with torch.no_grad():
                action_values = self.actor(state)
            self.actor.train()

            action = np.argmax(action_values.cpu().data.numpy(), axis=1)

        return action

    def learn(self, experiences):
        """Updates agent network parameters to learn from experiences.

        :param experiences: List of batched states, actions, rewards, next_states, dones in that order.
        :type state: List[torch.Tensor[float]]
        """
        states, actions, rewards, next_states, dones = experiences

        if self.one_hot:
            states = nn.functional.one_hot(
                states.long(), num_classes=self.state_dim[0]).float().squeeze()
            next_states = nn.functional.one_hot(
                next_states.long(), num_classes=self.state_dim[0]).float().squeeze()

        if self.double: # Double Q-learning
            q_idx = self.actor_target(next_states).argmax(dim=1).unsqueeze(1)
            q_target = self.actor(next_states).gather(dim=1, index=q_idx).detach()
        else:
            q_target = self.actor_target(next_states).detach().max(axis=1)[0].unsqueeze(1)

        # target, if terminal then y_j = rewards
        y_j = rewards + self.gamma * q_target * (1 - dones)
        q_eval = self.actor(states).gather(1, actions.long())

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
        for eval_param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)

    def test(self, env, swap_channels=False, max_steps=500, loop=3):
        """Returns mean test score of agent in environment with epsilon-greedy policy.

        :param env: The environment to be tested in
        :type env: Gym-style environment
        :param swap_channels: Swap image channels dimension from last to first [H, W, C] -> [C, H, W], defaults to False
        :type swap_channels: bool, optional
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
                    if swap_channels:
                        state = np.moveaxis(state, [3], [1])
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

        clone = type(self)(state_dim=self.state_dim,
                           action_dim=self.action_dim,
                           one_hot=self.one_hot,
                           index=index,
                           net_config=self.net_config,
                           batch_size=self.batch_size,
                           lr=self.lr,
                           learn_step=self.learn_step,
                           gamma=self.gamma,
                           tau=self.tau,
                           mutation=self.mut,
                           device=self.device,
                           )

        clone.actor = self.actor.clone().to(self.device)
        clone.actor_target = self.actor_target.clone().to(self.device)
        clone.optimizer = optim.Adam(clone.actor.parameters(), lr=clone.lr)
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
            'actor_init_dict': self.actor.init_dict,
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_init_dict': self.actor_target.init_dict,
            'actor_target_state_dict': self.actor_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'net_config': self.net_config,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'learn_step': self.learn_step,
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
        self.net_config = checkpoint['net_config']
        if self.net_config['arch'] == 'mlp':
            self.actor = EvolvableMLP(**checkpoint['actor_init_dict'])
            self.actor_target = EvolvableMLP(
                **checkpoint['actor_target_init_dict'])
        elif self.net_config['arch'] == 'cnn':
            self.actor = EvolvableCNN(**checkpoint['actor_init_dict'])
            self.actor_target = EvolvableCNN(
                **checkpoint['actor_target_init_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(
            checkpoint['actor_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.batch_size = checkpoint['batch_size']
        self.lr = checkpoint['lr']
        self.learn_step = checkpoint['learn_step']
        self.gamma = checkpoint['gamma']
        self.tau = checkpoint['tau']
        self.mut = checkpoint['mutation']
        self.index = checkpoint['index']
        self.scores = checkpoint['scores']
        self.fitness = checkpoint['fitness']
        self.steps = checkpoint['steps']

import random
import copy
import dill
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.networks.evolvable_cnn import EvolvableCNN


class MADDPG():
    """The MADDPG algorithm class. MADDPG paper: https://arxiv.org/abs/1706.02275

    :param state_dim: State observation dimension
    :type state_dim: int
    :param action_dim: Action dimension
    :type action_dim: int
    :param one_hot: One-hot encoding, used with discrete observation spaces
    :type one_hot: bool
    :param n_agents: Number of agents
    :type n_agents: int
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
    :param policy_freq: Frequency of target network updates compared to policy network, defaults to 2
    :type policy_freq: int, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: Hugging Face accelerate.Accelerator(), optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    """

    def __init__(self, state_dim, action_dim, one_hot, n_agents, index=0, 
                 net_config={'arch': 'mlp', 'h_size': [64,64]}, batch_size=64, lr=1e-4, 
                 learn_step=5, gamma=0.99, tau=1e-3, mutation=None, policy_freq=2, 
                 device='cpu', accelerator=None, wrap=True):
        self.algo = 'MADDPG'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.one_hot = one_hot
        self.n_agents = n_agents
        self.net_config = net_config
        self.batch_size = batch_size
        self.lr = lr
        self.learn_step = learn_step
        self.gamma = gamma
        self.tau = tau
        self.mut = mutation
        self.policy_freq = policy_freq
        self.device = device
        self.accelerator = accelerator

        self.index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]

        # model
        if self.net_config['arch'] == 'mlp':      # Multi-layer Perceptron
            self.actors = [EvolvableMLP(
                num_inputs=state_dim[0],
                num_outputs=action_dim,
                hidden_size=self.net_config['h_size'],
                output_activation='tanh',
                device=self.device,
                accelerator=self.accelerator) for i in range(self.n_agents)]
            self.actor_targets = copy.deepcopy(self.actors)

            self.critics = [EvolvableMLP(
                num_inputs=state_dim[0] + action_dim,
                num_outputs=1,
                hidden_size=self.net_config['h_size'],
                device=self.device,
                accelerator=self.accelerator) for i in range(self.n_agents)]
            self.critic_targets = copy.deepcopy(self.critics)

        elif self.net_config['arch'] == 'cnn':    # Convolutional Neural Network
            self.actors = [EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config['c_size'],
                kernal_size=self.net_config['k_size'],
                stride_size=self.net_config['s_size'],
                hidden_size=self.net_config['h_size'],
                normalize=self.net_config['normalize'],
                mlp_activation='tanh',
                device=self.device,
                accelerator=self.accelerator) for i in range(self.n_agents)]
            self.actor_targets = copy.deepcopy(self.actors)

            self.critics = [EvolvableCNN(
                input_shape=state_dim,
                num_actions=action_dim,
                channel_size=self.net_config['c_size'],
                kernal_size=self.net_config['k_size'],
                stride_size=self.net_config['s_size'],
                hidden_size=self.net_config['h_size'],
                normalize=self.net_config['normalize'],
                mlp_activation='tanh',
                critic=True,
                device=self.device,
                accelerator=self.accelerator) for i in range(self.n_agents)]
            self.critic_targets = copy.deepcopy(self.critics)

        self.actor_optimizer_types = [optim.Adam(actor.parameters(), lr=self.lr) for actor in self.actors]
        self.critic_optimizer_types = [optim.Adam(critic.parameters(), lr=self.lr) for critic in self.critics]

        if self.accelerator is not None:
            self.actor_optimizers = self.actor_optimizer_types
            self.critic_optimizers = self.critic_optimizer_types
            if wrap:
                self.wrap_models()          
        else:
            self.actors = [actor.to(self.device) for actor in self.actors]
            self.actor_targets = [actor_target.to(self.device) for actor_target in self.actor_targets]
            self.critics = [critic.to(self.device) for critic in self.critics]
            self.critic_targets = [critic_target.to(self.device) for critic_target in self.critic_targets]
            self.actor_optimizers = self.actor_optimizer_types
            self.critic_optimizers = self.critic_optimizer_types

        self.criterion = nn.MSELoss()

    def getAction(self, states, epsilon=0):
        """Returns the next action to take in the environment. 
        Epsilon is the probability of taking a random action, used for exploration.
        For epsilon-greedy behaviour, set epsilon to 0.

        :param state: Environment observations: [n_agents x state_dim]
        :type state: numpy.Array
        :param epsilon: Probablilty of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        """
        states = torch.from_numpy(states).float()
        if self.accelerator is None:
            states = states.to(self.device)

        if self.one_hot:
            states = nn.functional.one_hot(
                states.long(), num_classes=self.state_dim[0]).float().squeeze()

        if len(states.size()) < 2:
            states = states.unsqueeze(0)       

        actions = [] 
        for state, actor in zip(states, self.actors):
            # epsilon-greedy
            if random.random() < epsilon:
                action = (np.random.rand(state.size()[0], self.action_dim).astype('float32')-0.5)*2
            else:
                actor.eval()
                with torch.no_grad():
                    action_values = actor(state)
                actor.train()

                action = action_values.cpu().data.numpy()
            actions.append(action)

        return actions
    
    def _squeeze_exp(self, experiences):
        """Remove first dim created by dataloader.
        
        :param experiences: List of batched states, actions, rewards, next_states, dones in that order.
        :type state: List[torch.Tensor[float]]
        """
        st, ac, re, ne, do = experiences
        return st.squeeze(0), ac.squeeze(0), re.squeeze(0), ne.squeeze(0), do.squeeze(0)

    def learn(self, memory, noise_clip=0.5, policy_noise=0.2):
        """Updates agent network parameters to learn from experiences.

        :param memory: Experience Replay Buffer
        :type memory: object
        :param noise_clip: Maximum noise limit to apply to actions, defaults to 0.5
        :type noise_clip: float, optional
        :param policy_noise: Standard deviation of noise applied to policy, defaults to 0.2
        :type policy_noise: float, optional
        """
        for actor, actor_target, critic, critic_target, actor_optimizer, critic_optimizer in zip(self.actors, 
                                                                                                 self.actor_targets,
                                                                                                 self.critics, 
                                                                                                 self.critic_targets,
                                                                                                 self.actor_optimizers, 
                                                                                                 self.critic_optimizers):
            # [batch_size x n_agents x dim]
            states, actions, rewards, next_states, dones = memory.sample(self.batch_size)

            if self.one_hot:
                states = nn.functional.one_hot(
                    states.long(), num_classes=self.state_dim[0]).float().squeeze()
                next_states = nn.functional.one_hot(
                    next_states.long(), num_classes=self.state_dim[0]).float().squeeze()

            if self.net_config['arch'] == 'mlp':
                input_combined = torch.cat([states, actions], 1)
                q_value = critic(input_combined)
            elif self.net_config['arch'] == 'cnn':
                q_value = critic(states, actions)

            next_actions = [self.actor_targets[i](next_states[:,i,:]) for i in range(self.n_agents)]
            next_actions = torch.stack(next_actions)
            # next_actions = (next_actions.transpose(0, 1).contiguous())
            noise = actions.data.normal_(0, policy_noise)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_actions = (next_actions + noise)

            if self.net_config['arch'] == 'mlp':
                next_input_combined = torch.cat([next_states, next_actions], 1)
                q_value_next_state = critic_target(next_input_combined)
            elif self.net_config['arch'] == 'cnn':
                q_value_next_state = critic_target(next_states, next_actions)

            y_j = rewards + ((1-dones)*self.gamma * q_value_next_state).detach()

            critic_loss = self.criterion(q_value, y_j)

            # critic loss backprop
            critic_optimizer.zero_grad()
            if self.accelerator is not None:
                self.accelerator.backward(critic_loss)
            else:
                critic_loss.backward()
            critic_optimizer.step()

            # update actor and targets every policy_freq episodes
            if len(self.scores) % self.policy_freq == 0:
                if self.net_config['arch'] == 'mlp':
                    input_combined = torch.cat(
                        [states, actor.forward(states)], 1)
                    actor_loss = -critic(input_combined).mean()
                elif self.net_config['arch'] == 'cnn':
                    actor_loss = - \
                        critic(states, actor.forward(states)).mean()

                # actor loss backprop
                actor_optimizer.zero_grad()
                if self.accelerator is not None:
                    self.accelerator.backward(actor_loss)
                else:
                    actor_loss.backward()
                actor_optimizer.step()

        if len(self.scores) % self.policy_freq == 0:
            for actor, actor_target, critic, critic_target in zip(self.actors, self.actor_targets, 
                                                                  self.critics, self.critic_targets):
                self.softUpdate(actor, actor_target)
                self.softUpdate(critic, critic_target)

    def softUpdate(self, net, target):
        """Soft updates target network.
        """
        for eval_param, target_param in zip(net.parameters(), target.parameters()):
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
        :param loop: Number of testing loops/epsiodes to complete. The returned score is the mean. Defaults to 3
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
                    state, reward, done, trunc, info = env.step(action)
                    score += reward
                rewards.append(score)
        mean_fit = np.mean(rewards)
        self.fitness.append(mean_fit)
        return mean_fit

    def clone(self, index=None, wrap=True):
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
                           policy_freq=self.policy_freq,
                           device=self.device,
                           accelerator=self.accelerator,
                           wrap=wrap)
        
        if self.accelerator is not None:
            self.unwrap_models()
        actor = self.actor.clone()
        actor_target = self.actor_target.clone()
        critic = self.critic.clone()
        critic_target = self.critic_target.clone()
        actor_optimizer = optim.Adam(actor.parameters(), lr=clone.lr)
        critic_optimizer = optim.Adam(critic.parameters(), lr=clone.lr)
        clone.actor_optimizer_type = actor_optimizer
        clone.critic_optimizer_type = critic_optimizer

        if self.accelerator is not None:
            if wrap:
                clone.actor, clone.actor_target, clone.critic, clone.critic_target, \
                clone.actor_optimizer, clone.critic_optimizer = self.accelerator.prepare(actor,
                                                                                actor_target,
                                                                                critic,
                                                                                critic_target,
                                                                                actor_optimizer,
                                                                                critic_optimizer)
            else:
                clone.actor, clone.actor_target, clone.critic, clone.critic_target, \
                clone.actor_optimizer, clone.critic_optimizer = actor, actor_target, critic, \
                critic_target, actor_optimizer, critic_optimizer
        else:
            clone.actor = actor.to(self.device)
            clone.actor_target = actor_target.to(self.device)
            clone.critic = critic.to(self.device)
            clone.critic_target = critic_target.to(self.device)
            clone.actor_optimizer = actor_optimizer
            clone.critic_optimizer = critic_optimizer

        clone.fitness = copy.deepcopy(self.fitness)
        clone.steps = copy.deepcopy(self.steps)
        clone.scores = copy.deepcopy(self.scores)

        return clone

    def wrap_models(self):
        if self.accelerator is not None:
            self.actor, self.actor_target, self.critic, self.critic_target, \
            self.actor_optimizer, self.critic_optimizer = self.accelerator.prepare(self.actor,
                                                                            self.actor_target,
                                                                            self.critic,
                                                                            self.critic_target,
                                                                            self.actor_optimizer_type,
                                                                            self.critic_optimizer_type)
    
    def unwrap_models(self):
        if self.accelerator is not None:
            self.actor = self.accelerator.unwrap_model(self.actor)
            self.actor_target = self.accelerator.unwrap_model(self.actor_target)
            self.critic = self.accelerator.unwrap_model(self.critic)
            self.critic_target = self.accelerator.unwrap_model(self.critic_target)
            self.actor_optimizer = self.accelerator.unwrap_model(self.actor_optimizer)
            self.critic_optimizer = self.accelerator.unwrap_model(self.critic_optimizer)

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
            'critic_init_dict': self.critic.init_dict,
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_init_dict': self.critic_target.init_dict,
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
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
        }, path, pickle_module=dill)

    def loadCheckpoint(self, path):
        """Loads saved agent properties and network weights from checkpoint.

        :param path: Location to load checkpoint from
        :type path: string
        """
        checkpoint = torch.load(path, pickle_module=dill)
        self.net_config = checkpoint['net_config']
        if self.net_config['arch'] == 'mlp':
            self.actor = EvolvableMLP(**checkpoint['actor_init_dict'])
            self.actor_target = EvolvableMLP(**checkpoint['actor_target_init_dict'])
            self.critic = EvolvableMLP(**checkpoint['critic_init_dict'])
            self.critic_target = EvolvableMLP(**checkpoint['critic_target_init_dict'])
        elif self.net_config['arch'] == 'cnn':
            self.actor = EvolvableCNN(**checkpoint['actor_init_dict'])
            self.actor_target = EvolvableCNN(**checkpoint['actor_target_init_dict'])
            self.critic = EvolvableCNN(**checkpoint['critic_init_dict'])
            self.critic_target = EvolvableCNN(**checkpoint['critic_target_init_dict'])
        self.lr = checkpoint['lr']
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.batch_size = checkpoint['batch_size']
        self.learn_step = checkpoint['learn_step']
        self.gamma = checkpoint['gamma']
        self.tau = checkpoint['tau']
        self.mut = checkpoint['mutation']
        self.index = checkpoint['index']
        self.scores = checkpoint['scores']
        self.fitness = checkpoint['fitness']
        self.steps = checkpoint['steps']

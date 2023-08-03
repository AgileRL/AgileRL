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

    :param state_dims: State observation dimensions for each agent
    :type state_dims: List[tuple]
    :param action_dims: Action dimensions for each agent
    :type action_dims: List[int]
    :param one_hot: One-hot encoding, used with discrete observation spaces
    :type one_hot: bool
    :param n_agents: Number of agents
    :type n_agents: int
    :param agent_ids: Agent ID for each agent
    :type agent_ids: List[str]
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
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: Hugging Face accelerate.Accelerator(), optional
    :param wrap: Wrap models for distributed training upon creation, defaults to True
    :type wrap: bool, optional
    """

    def __init__(self, state_dims, action_dims, one_hot, n_agents, agent_ids, max_action, 
                 min_action, expl_noise=0.1, index=0, discrete_actions=False, 
                 net_config={'arch': 'mlp', 'h_size': [64,64]}, batch_size=64, lr=0.01,
                 learn_step=5, gamma=0.99, tau=1e-3, mutation=None, device='cpu', accelerator=None, wrap=True):
        self.algo = 'MADDPG'
        self.state_dims = state_dims
        self.total_state_dims = sum(state_dim[0] for state_dim in self.state_dims)
        self.action_dims = action_dims
        self.one_hot = one_hot
        self.n_agents = n_agents
        self.agent_ids = agent_ids
        self.net_config = net_config
        self.batch_size = batch_size
        self.lr = lr
        self.learn_step = learn_step
        self.gamma = gamma
        self.tau = tau
        self.mut = mutation
        self.device = device
        self.accelerator = accelerator
        self.index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]

        self.max_action = max_action
        self.expl_noise = expl_noise
        self.min_action = min_action
        self.discrete_actions = discrete_actions
        self.policy_freq=2

        # model
        if self.net_config['arch'] == 'mlp':      # Multi-layer Perceptron
            self.actors = [EvolvableMLP(
                num_inputs=state_dim[0],
                num_outputs=action_dim,
                hidden_size=self.net_config['h_size'],
                output_activation='softmax',
                device=self.device,
                accelerator=self.accelerator) for (action_dim, state_dim) in zip(self.action_dims, self.state_dims)]
            self.actor_targets = copy.deepcopy(self.actors)

            self.critics = [EvolvableMLP(
                num_inputs=self.total_state_dims + sum(self.action_dims),
                num_outputs=1,
                hidden_size=self.net_config['h_size'],
                device=self.device,
                accelerator=self.accelerator) for _ in range(self.n_agents)]
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
                mlp_activation='gumbel_softmax',
                device=self.device,
                accelerator=self.accelerator) for (action_dim, state_dim) in zip(self.action_dims, self.state_dims)]
            self.actor_targets = copy.deepcopy(self.actors)

            self.critics = [EvolvableCNN(
                input_shape=state_dim, #### This needs changing once mlp is working for the base case, it needs to be a summation of all states
                num_actions=sum(self.action_dims),
                channel_size=self.net_config['c_size'],
                kernal_size=self.net_config['k_size'],
                stride_size=self.net_config['s_size'],
                hidden_size=self.net_config['h_size'],
                normalize=self.net_config['normalize'],
                mlp_activation='tanh',
                critic=True,
                device=self.device,
                accelerator=self.accelerator) for state_dim in self.state_dims]
            self.critic_targets = copy.deepcopy(self.critics)

        self.actor_optimizers_type = [optim.Adam(actor.parameters(), lr=self.lr) for actor in self.actors]
        self.critic_optimizers_type = [optim.Adam(critic.parameters(), lr=self.lr) for critic in self.critics]

        if self.accelerator is not None:
            self.actor_optimizers = self.actor_optimizers_type
            self.critic_optimizers = self.critic_optimizers_type
            if wrap:
                self.wrap_models()          
        else:
            self.actors = [actor.to(self.device) for actor in self.actors]
            self.actor_targets = [actor_target.to(self.device) for actor_target in self.actor_targets]
            self.critics = [critic.to(self.device) for critic in self.critics]
            self.critic_targets = [critic_target.to(self.device) for critic_target in self.critic_targets]
            self.actor_optimizers = self.actor_optimizers_type
            self.critic_optimizers = self.critic_optimizers_type

        self.criterion = nn.MSELoss()

    def getAction(self, states, epsilon=0):
        """Returns the next action to take in the environment. 
        Epsilon is the probability of taking a random action, used for exploration.
        For epsilon-greedy behaviour, set epsilon to 0.

        :param state: Environment observations: {'agent_0': state_dim_0, ..., 'agent_n': state_dim_n}
        :type state: Dict[str, numpy.Array]
        :param epsilon: Probablilty of taking a random action for exploration, defaults to 0
        :type epsilon: float, optional
        """
        # Convert states to a list of torch tensors
        states = [torch.from_numpy(state).float() for state in states.values()]

        # Configure accelerator
        if self.accelerator is None:
            states = [state.to(self.device) for state in states]

        #### Need to adjust
        if self.one_hot:
            states = [nn.functional.one_hot(
                state.long(), num_classes=state_dim[0]).float().squeeze() for state, state_dim 
                in zip(states, self.state_dims)]

        states = [state.unsqueeze(0) if len(state.size()) < 2 else state for state in states ]   

        actions = {} 
        for idx, (agent_id, state, actor) in enumerate(zip(self.agent_ids, states, self.actors)):
            if random.random() < epsilon:
                if self.discrete_actions:
                    action = np.random.randint(0, self.action_dims[idx])
                else:
                    action = np.random.rand(state.size()[0], self.action_dims[idx]).astype('float32').squeeze()
            else:
                actor.eval()
                with torch.no_grad():
                    action_values = actor(state)
                actor.train()
                #### Accounts for the discrete action space of the atari environments 
                if self.discrete_actions:
                    action = action_values.squeeze(0).argmax().item()
                else:
                    action = action_values.cpu().data.numpy().squeeze() #\
                    ## Commented out as not present in single agent ddpg
                    #     + np.random.normal(0, self.max_action[idx][0] * self.expl_noise, size=self.action_dims[idx]).astype(np.float32)
                    # action = np.clip(action, self.min_action[idx][0], self.max_action[idx][0])     
            actions[agent_id] = action
        
        return actions
    
    def _squeeze_exp(self, experiences):
        """Remove first dim created by dataloader.
        
        :param experiences: List of batched states, actions, rewards, next_states, dones in that order.
        :type state: List[torch.Tensor[float]]
        """
        st, ac, re, ne, do = experiences
        return st.squeeze(0), ac.squeeze(0), re.squeeze(0), ne.squeeze(0), do.squeeze(0)

    def learn(self, experiences, policy_noise=0.2):
        """Updates agent network parameters to learn from experiences.

        :param experience: Tuple of dictionaries containing batched states, actions, rewards, next_states, 
        dones in that order for each individual agent.
        :type experience: Tuple[Dict[str, torch.Tensor]]
        """

        for agent_id, actor, actor_target, critic, critic_target, actor_optimizer, critic_optimizer in zip(self.agent_ids,
                                                                                                 self.actors,    
                                                                                                 self.actor_targets,
                                                                                                 self.critics, 
                                                                                                 self.critic_targets,
                                                                                                 self.actor_optimizers, 
                                                                                                 self.critic_optimizers):
         
            #### Re-configure once the base case is working
            # if self.one_hot:
            #     states = nn.functional.one_hot(
            #         states.long(), num_classes=self.state_dim[0]).float().squeeze()
            #     next_states = nn.functional.one_hot(
            #         next_states.long(), num_classes=self.state_dim[0]).float().squeeze()

            states, actions, rewards, next_states, dones = experiences

            if self.net_config['arch'] == 'mlp':
                input_combined = torch.cat(list(states.values()) + list(actions.values()), 1)
                q_value = critic(input_combined)
                
            #### Work on cnn once mlp is working
            elif self.net_config['arch'] == 'cnn':
                q_value = critic(states.values(), actions.values())
            next_actions = [self.actor_targets[idx](next_states[agent_id]).detach_() + actions[agent_id].data.normal_(0, policy_noise).clamp(0,1)
                            for idx, agent_id in enumerate(self.agent_ids)]
            #next_actions = torch.stack(next_actions_)
            #### Add in the noise once we have the simplest mlp case working
            # noise = actions.data.normal_(0, policy_noise)
            # noise = noise.clamp(-noise_clip, noise_clip)
            # next_actions = (next_actions + noise)
            # for na, idx in enumerate(next_actions):
            #     noise = actions[id].data.normal_(0, policy_noise)
            #     noise = noise.clamp(0, 1)
            #     na += noise

            if self.net_config['arch'] == 'mlp':
                next_input_combined = torch.cat(list(next_states.values()) + next_actions, 1)
                q_value_next_state = critic_target(next_input_combined)

            #### Work on cnn once mlp is working
            elif self.net_config['arch'] == 'cnn':
                q_value_next_state = critic_target(next_states.values(), next_actions)

            y_j = rewards[agent_id] + (1 - dones[agent_id]) * self.gamma * q_value_next_state

            critic_loss = self.criterion(q_value, y_j.detach_())

            # critic loss backprop
            critic_optimizer.zero_grad()
            if self.accelerator is not None:
                self.accelerator.backward(critic_loss)
            else:
                critic_loss.backward()
            critic_optimizer.step()

            ### Add in the time delay once basic case is working
            # update actor and targets every policy_freq episodes
            if len(self.scores) % self.policy_freq == 0:
                if self.net_config['arch'] == 'mlp':
                    action = actor(states[agent_id])
                    detached_actions = copy.deepcopy(actions)
                    detached_actions[agent_id] = action
                    input_combined = torch.cat(list(states.values()) + list(detached_actions.values()), 1)
                    actor_loss = -critic(input_combined).mean()
                    
                    #### Complete cnns once mlp is working 
                elif self.net_config['arch'] == 'cnn':
                    action = actor(states[agent_id])
                    detached_actions = copy.deepcopy(actions)
                    detached_actions[agent_id] = action
                    actor_loss = - \
                        critic(states.values(), detached_actions.values()).mean()

                # actor loss backprop
                actor_optimizer.zero_grad()
                if self.accelerator is not None:
                    self.accelerator.backward(actor_loss)
                else:
                    actor_loss.backward()
                actor_optimizer.step()

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
                state, _ = env.reset()
                agent_reward = {agent_id: 0 for agent_id in self.agent_ids}
                while env.agents:
                    if swap_channels:
                        state = np.moveaxis(state, [3], [1])
                    action = self.getAction(state, epsilon=0)
                    state, reward, done, trunc, info = env.step(action)
                    for agent_id, r in reward.items():
                        agent_reward[agent_id] += r 
                    score = sum(agent_reward.values())
                rewards.append(score)
        mean_fit = np.mean(rewards)
        self.fitness.append(mean_fit)
        #print(self.fitness)
        return mean_fit

    def clone(self, index=None, wrap=True):
        """Returns cloned agent identical to self.

        :param index: Index to keep track of agent for tournament selection and mutation, defaults to None
        :type index: int, optional
        """
        if index is None:
            index = self.index

        clone = type(self)(state_dims=self.state_dims,
                           action_dims=self.action_dims,
                           one_hot=self.one_hot,
                           n_agents=self.n_agents,
                           agent_ids=self.agent_ids,
                           max_action=self.max_action,
                           min_action=self.min_action,
                           expl_noise=self.expl_noise,
                           index=index,
                           net_config=self.net_config,
                           batch_size=self.batch_size,
                           lr=self.lr,
                           learn_step=self.learn_step,
                           gamma=self.gamma,
                           tau=self.tau,
                           mutation=self.mut,
                           device=self.device,
                           accelerator=self.accelerator,
                           wrap=wrap)
        
        if self.accelerator is not None:
            self.unwrap_models()
        actors = [actor.clone() for actor in self.actors]
        actor_targets = [actor_target.clone() for actor_target in self.actor_targets]
        critics = [critic.clone() for critic in self.critic_targets]
        critic_targets = [critic_target.clone() for critic_target in self.critic_targets]
        actor_optimizers = [optim.Adam(actor.parameters(), lr=clone.lr) for actor in actors]
        critic_optimizers = [optim.Adam(critic.parameters(), lr=clone.lr) for critic in critics]
        clone.actor_optimizers_type = actor_optimizers
        clone.critic_optimizers_type = critic_optimizers

        if self.accelerator is not None:
            if wrap:
                clone.actors, clone.actor_targets, clone.critics, clone.critic_targets, \
                clone.actor_optimizer, clone.critic_optimizer = self.accelerator.prepare(actors,
                                                                                actor_targets,
                                                                                critics,
                                                                                critic_targets,
                                                                                actor_optimizers,
                                                                                critic_optimizers)
            else:
                clone.actors, clone.actor_targets, clone.critics, clone.critic_targets, \
                clone.actor_optimizer, clone.critic_optimizer = actors, actor_targets, critics, \
                critic_targets, actor_optimizers, critic_optimizers
        else:
            clone.actors = [actor.to(self.device) for actor in actors]
            clone.actor_targets = [actor_target.to(self.device) for actor_target in actor_targets]
            clone.critics = [critic.to(self.device) for critic in critics]
            clone.critic_targets = [critic_target.to(self.device) for critic_target in critic_targets]
            clone.actor_optimizers = actor_optimizers
            clone.critic_optimizers = critic_optimizers

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
        
        for agent_id, actor, actor_target, critic, critic_target, actor_optimizer, critic_optimizer in zip(self.agent_ids,
                                                                                                        self.actors,    
                                                                                                        self.actor_targets,
                                                                                                        self.critics, 
                                                                                                        self.critic_targets,
                                                                                                        self.actor_optimizers, 
                                                                                                        self.critic_optimizers):
            torch.save({
                f'actor_init_dict_{agent_id}': actor.init_dict,
                f'actor_state_dict_{agent_id}': actor.state_dict(),
                f'actor_target_init_dict_{agent_id}': actor_target.init_dict,
                f'actor_target_state_dict_{agent_id}': actor_target.state_dict(),
                f'critic_init_dict_{agent_id}': critic.init_dict,
                f'critic_state_dict_{agent_id}': critic.state_dict(),
                f'critic_target_init_dict_{agent_id}': critic_target.init_dict,
                f'critic_target_state_dict_{agent_id}': critic_target.state_dict(),
                f'actor_optimizer_state_dict_{agent_id}': actor_optimizer.state_dict(),
                f'critic_optimizer_state_dict_{agent_id}': critic_optimizer.state_dict()}
                , path, pickle_module=dill)

        torch.save({
            'net_config': self.net_config,
            'batch_size': self.batch_size,
            'lr' : self.lr,
            'learn_step': self.learn_step,
            'gamma': self.gamma,
            'tau': self.tau,
            'mutation': self.mut,
            'index': self.index,
            'scores': self.scores,
            'fitness': self.fitness,
            'steps': self.steps
            }, path, pickle_module=dill)
        

    def loadCheckpoint(self, path):
        """Loads saved agent properties and network weights from checkpoint.

        :param path: Location to load checkpoint from
        :type path: string
        """
        checkpoint = torch.load(path, pickle_module=dill)
        self.net_config = checkpoint['net_config']
        if self.net_config['arch'] == 'mlp':
            self.actors = [EvolvableMLP(**checkpoint[f'actor_init_dict_{agent_id}']) 
                           for agent_id in self.agent_ids]
            self.actor_targets = [EvolvableMLP(**checkpoint[f'actor_target_init_dict_{agent_id}'])
                                  for agent_id in self.agent_ids]
            self.critics = [EvolvableMLP(**checkpoint[f'critic_init_dict_{agent_id}'])
                            for agent_id in self.agent_ids]
            self.critic_targets = [EvolvableMLP(**checkpoint[f'critic_target_init_dict_{agent_id}'])
                                   for agent_id in self.agent_ids]
        elif self.net_config['arch'] == 'cnn':
            self.actors = [EvolvableCNN(**checkpoint[f'actor_init_dict_{agent_id}']) 
                           for agent_id in self.agent_ids]
            self.actor_targets = [EvolvableCNN(**checkpoint[f'actor_target_init_dict_{agent_id}'])
                                  for agent_id in self.agent_ids]
            self.critics = [EvolvableCNN(**checkpoint[f'critic_init_dict_{agent_id}'])
                            for agent_id in self.agent_ids]
            self.critic_targets = [EvolvableCNN(**checkpoint[f'critic_target_init_dict_{agent_id}'])
                                   for agent_id in self.agent_ids]
        self.lr = checkpoint['lr']
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=self.lr) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=self.lr) for critic in self.critic]
        self.actors = [actor.load_state_dict(checkpoint[f'actor_state_dict_{agent_id}']) for actor, agent_id in zip(self.actors, self.agent_ids)]
        self.actor_targets = [actor_target.load_state_dict(checkpoint[f'actor_target_state_dict_{agent_id}']) for actor_target, agent_id in zip(self.actor_targets, self.agent_ids)]
        self.critics = [critic.load_state_dict(checkpoint[f'critic_state_dict_{agent_id}']) for critic, agent_id in zip(self.critics, self.agent_ids)]
        self.critic_targets = [critic_target.load_state_dict(checkpoint[f'critic_target_state_dict_{agent_id}']) for critic_target, agent_id in zip(self.critic_targets, self.agent_ids)]
        self.actor_optimizers = [actor_optimizer.load_state_dict(checkpoint[f'actor_optimizer_state_dict_{agent_id}']) for actor_optimizer, agent_id in zip(self.actor_optimizers, self.agent_ids)]
        self.critic_optimizers = [critic_optimizer.load_state_dict(checkpoint[f'critic_optimizer_state_dict_{agent_id}']) for critic_optimizer, agent_id in zip(self.critic_optimizers, self.agent_ids)]
        self.batch_size = checkpoint['batch_size']
        self.learn_step = checkpoint['learn_step']
        self.gamma = checkpoint['gamma']
        self.tau = checkpoint['tau']
        self.mut = checkpoint['mutation']
        self.index = checkpoint['index']
        self.scores = checkpoint['scores']
        self.fitness = checkpoint['fitness']
        self.steps = checkpoint['steps']

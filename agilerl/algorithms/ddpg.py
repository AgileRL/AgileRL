import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agilerl.networks.evolvable_mlp import EvolvableMLP

class DDPG():
    def __init__(self, n_states, n_actions, index, h_size = [64,64], batch_size=64, lr=1e-4, gamma=0.99, learn_step=5, tau=1e-3, mutation=None, policy_freq=2, device='cpu'):
        self.algo = 'DDPG'
        self.n_states = n_states
        self.n_actions = n_actions
        self.h_size = h_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.learn_step = learn_step
        self.tau = tau
        self.mut = mutation
        self.policy_freq = policy_freq
        self.device = device

        self.index = index
        self.scores = []
        self.fitness = []
        self.steps = [0]

        # model
        self.actor = EvolvableMLP(num_inputs=n_states, num_outputs=n_actions, hidden_size=h_size, output_activation='tanh', device=self.device).to(device)
        self.actor_target = EvolvableMLP(num_inputs=n_states, num_outputs=n_actions, hidden_size=h_size, output_activation='tanh', device=self.device).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = EvolvableMLP(num_inputs=n_states+n_actions, num_outputs=1, hidden_size=h_size, device=self.device).to(device)
        self.critic_target = EvolvableMLP(num_inputs=n_states+n_actions, num_outputs=1, hidden_size=h_size, device=self.device).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def getAction(self, state, epsilon):
        state = torch.from_numpy(state).float().to(self.device)
        if len(state.size())<2:
            state = state.unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            action_values = self.actor(state)
        self.actor.train()

        # epsilon-greedy
        if random.random() < epsilon:
            action = (np.random.rand(state.size()[0], self.n_actions).astype('float32')-0.5)*2
        else:
            action = action_values.cpu().data.numpy()

        return action

    def learn(self, experiences, noise_clip=0.5, policy_noise=0.2):
        states, actions, rewards, next_states, dones = experiences

        input_combined = torch.cat([states, actions], 1)
        q_value = self.critic(input_combined)

        next_actions = self.actor_target(next_states)
        noise = actions.data.normal_(0, policy_noise).to(self.device)
        noise = noise.clamp(-noise_clip, noise_clip)
        next_actions = (next_actions + noise)

        next_input_combined = torch.cat([next_states, next_actions], 1)
        q_value_next_state = self.critic_target(next_input_combined)
        y_j = rewards + (self.gamma * q_value_next_state).detach()

        critic_loss = self.criterion(q_value, y_j)

        # critic loss backprop
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor and targets every policy_freq episodes
        if len(self.scores) % self.policy_freq == 0:
            input_combined = torch.cat([states, self.actor.forward(states)], 1)
            actor_loss = -self.critic(input_combined).mean()
            # actor loss backprop
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.softUpdate(self.actor, self.actor_target)
            self.softUpdate(self.critic, self.critic_target)

    def softUpdate(self, net, target):
        for eval_param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.copy_(self.tau*eval_param.data + (1.0-self.tau)*target_param.data)

    def test(self, env, max_steps=500, loop=3):
        with torch.no_grad():
            rewards = []
            for i in range(loop):
                state = env.reset()[0]
                score = 0
                for idx_step in range(max_steps):
                    action = self.getAction(state, epsilon=0)
                    state, reward, done, _, _ = env.step(action)
                    score += reward
                    if done:
                        break
                rewards.append(score)
        mean_fit = np.mean(rewards)
        self.fitness.append(mean_fit)
        return mean_fit

    def clone(self, index=None):
        if index is None:
            index = self.index

        clone = type(self)(n_states=self.n_states,
                            n_actions=self.n_actions,
                            index=index, 
                            h_size = self.h_size, 
                            batch_size=self.batch_size,
                            lr=self.lr,
                            gamma=self.gamma,
                            learn_step=self.learn_step,
                            tau=self.tau,
                            device=self.device,
                           )
                           
        clone.actor = self.actor.clone().to(self.device)
        clone.actor_target = self.actor_target.clone().to(self.device)
        clone.critic = self.critic.clone().to(self.device)
        clone.critic_target = self.critic_target.clone().to(self.device)

        clone.actor_optimizer = optim.Adam(clone.actor.parameters(), lr=clone.lr)
        clone.critic_optimizer = optim.Adam(clone.critic.parameters(), lr=clone.lr)

        clone.fitness = copy.deepcopy(self.fitness)
        clone.steps = copy.deepcopy(self.steps)
        clone.scores = copy.deepcopy(self.scores)

        return clone
    
    def saveCheckpoint(self, path):
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
                    'batch_size': self.batch_size,
                    'lr': self.lr,
                    'gamma': self.gamma,
                    'learn_step': self.learn_step,
                    'tau': self.tau,
                    'mutation': self.mut,
                    'index': self.index, 
                    'scores': self.scores,
                    'fitness': self.fitness,
                    'steps': self.steps,
                    }, path)
        
    def loadCheckpoint(self, path):
        checkpoint = torch.load(path)
        self.actor = EvolvableMLP(**checkpoint['actor_init_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target = EvolvableMLP(**checkpoint['actor_target_init_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic = EvolvableMLP(**checkpoint['critic_init_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target = EvolvableMLP(**checkpoint['critic_target_init_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.batch_size = checkpoint['batch_size']
        self.lr = checkpoint['lr']
        self.gamma = checkpoint['gamma']
        self.learn_step = checkpoint['learn_step']
        self.tau = checkpoint['tau']
        self.mut = checkpoint['mutation']
        self.index = checkpoint['index']
        self.scores = checkpoint['scores']
        self.fitness = checkpoint['fitness']
        self.steps = checkpoint['steps']
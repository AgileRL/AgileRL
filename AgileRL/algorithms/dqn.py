import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agilerl.networks.evolvable_mlp import EvolvableMLP

class DQN():
    def __init__(self, n_states, n_actions, index, h_size = [64,64], batch_size=64, lr=1e-4, gamma=0.99, learn_step=5, tau=1e-3, mutation=None, device='cpu'):
        self.algo = 'DQN'
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.h_size = h_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.learn_step = learn_step
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

    def getAction(self, state, epsilon):
        state = torch.from_numpy(state).float().to(self.device)
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
        states, actions, rewards, next_states, dones = experiences

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
        for eval_param, target_param in zip(self.net_eval.parameters(), self.net_target.parameters()):
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
                           
        clone.net_eval = self.net_eval.clone().to(self.device)
        clone.net_target = self.net_target.clone().to(self.device)
        clone.optimizer = optim.Adam(clone.net_eval.parameters(), lr=clone.lr)
        clone.fitness = copy.deepcopy(self.fitness)
        clone.steps = copy.deepcopy(self.steps)
        clone.scores = copy.deepcopy(self.scores)

        return clone
    
    def saveCheckpoint(self, path):
        torch.save({
                    'net_eval_init_dict': self.net_eval.init_dict,
                    'net_eval_state_dict': self.net_eval.state_dict(),
                    'net_target_init_dict': self.net_target.init_dict,
                    'net_target_state_dict': self.net_target.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
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
        self.net_eval = EvolvableMLP(**checkpoint['net_eval_init_dict'])
        self.net_eval.load_state_dict(checkpoint['net_eval_state_dict'])
        self.net_target = EvolvableMLP(**checkpoint['net_target_init_dict'])
        self.net_target.load_state_dict(checkpoint['net_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
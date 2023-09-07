import numpy as np
import torch

from agilerl.algorithms.ppo import PPO
from agilerl.utils.utils import makeVectEnvs

    
# Tests that the PPO class is initialized correctly with default parameters.
def test_initialize_with_default_parameters(self):
    ppo = PPO(state_dim=(4,), action_dim=2, one_hot=False, discrete_actions=True)
    assert ppo.algo == 'PPO'
    assert ppo.state_dim == (4,)
    assert ppo.action_dim == 2
    assert ppo.one_hot is False
    assert ppo.discrete_actions is True
    assert ppo.net_config == {'arch': 'mlp', 'h_size': [64, 64]}
    assert ppo.batch_size == 64
    assert ppo.lr == 0.0001
    assert ppo.gamma == 0.99
    assert ppo.gae_lambda == 0.95
    assert ppo.mut is None
    assert ppo.action_std_init == 0.6
    assert ppo.clip_coef == 0.2
    assert ppo.ent_coef == 0.01
    assert ppo.vf_coef == 0.5
    assert ppo.max_grad_norm == 0.5
    assert ppo.target_kl is None
    assert ppo.update_epochs == 4
    assert ppo.device == 'cpu'
    assert ppo.accelerator is None
    assert ppo.index == 0
    assert ppo.scores == []
    assert ppo.fitness == []
    assert ppo.steps == [0]
    
# Tests that the getAction() method returns the expected action, action log probability, distribution entropy, and state value for a valid state.
def test_get_action_with_valid_state(self):
    ppo = PPO(state_dim=(4,), action_dim=2, one_hot=False, discrete_actions=True)
    state = np.array([1, 2, 3, 4])
    action, action_logprob, dist_entropy, state_value = ppo.getAction(state)
    assert isinstance(action, torch.Tensor)
    assert isinstance(action_logprob, torch.Tensor)
    assert isinstance(dist_entropy, torch.Tensor)
    assert isinstance(state_value, torch.Tensor)
    
# Tests that the learn() method updates the actor and critic networks correctly with a set of experiences.
def test_learn_with_set_of_experiences(self):
    ppo = PPO(state_dim=(4,), action_dim=2, one_hot=False, discrete_actions=True)
    experiences = [
        np.array([1, 2, 3, 4]),  # states
        np.array([0, 1]),  # actions
        np.array([0.5, 0.5]),  # log_probs
        np.array([1.0]),  # rewards
        np.array([0]),  # dones
        np.array([0.5]),  # values
        np.array([5, 6, 7, 8])  # next_state
    ]
    ppo.learn(experiences)
    # assert the actor and critic networks have been updated
    
# Tests that the test() method returns the expected fitness score for a valid environment.
def test_test_with_valid_environment(self):
    ppo = PPO(state_dim=(4,), action_dim=2, one_hot=False, discrete_actions=True)
    env = env = makeVectEnvs(env_name="LunarLander-v2", num_envs=1)
    fitness = ppo.test(env)
    assert isinstance(fitness, float)
from pettingzoo.mpe import simple_adversary_v3
import torch


if __name__ == "__main__":
    # Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configure the environment
    env = simple_adversary_v3.env(N=2, max_cycles=25, continuous_actions=True)
    print('Number of agents', env.n)

    # Define the hyperparameters
    state_dim = 
    action_dim =  env.action
    one_hot = False 
    n_agents = env.n 
    index=0
    net_config={'arch': 'mlp', 'h_size': [64,64]}
    batch_size=64
    lr=1e-4
    learn_step=5
    gamma=0.99
    tau=1e-3
    mutation=None
    policy_freq=2
    device='cpu'
    accelerator=None 
    wrap=True

    



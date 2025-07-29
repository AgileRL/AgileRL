On-Policy Rollout Buffer
========================

On-policy RL algorithms like PPO and A2C require collecting experiences from the current policy and computing advantages before updating the policy.
Unlike off-policy algorithms that can reuse old experiences, on-policy methods need fresh data from each policy iteration.

The rollout buffer is designed to efficiently collect experiences from vectorized environments, compute Generalized Advantage Estimation (GAE),
and provide properly formatted batches for policy updates. After each training iteration, the buffer is typically reset since the experiences
are no longer valid for the updated policy.

During environment interaction, use ``RolloutBuffer.add()`` to store transitions. Once an episode or rollout is complete, call
``RolloutBuffer.compute_returns_and_advantages()`` to calculate GAE advantages and returns. Finally, use ``RolloutBuffer.get_tensor_batch()``
to sample minibatches for policy optimization.

.. code-block:: python

    from agilerl.components.rollout_buffer import RolloutBuffer

    buffer = RolloutBuffer(
        capacity=2048,  # Number of steps to collect per environment
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_envs=8,  # Number of parallel environments
        device=device,
        gae_lambda=0.95,  # GAE lambda parameter
        gamma=0.99,  # Discount factor
    )

Parameters
------------

.. autoclass:: agilerl.components.rollout_buffer.RolloutBuffer
  :members:
  :inherited-members:

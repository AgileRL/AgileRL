Algorithm Utils
==============

Space and Observation Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: agilerl.utils.algo_utils.get_input_size_from_space

.. autofunction:: agilerl.utils.algo_utils.get_output_size_from_space

.. autofunction:: agilerl.utils.algo_utils.get_obs_shape

.. autofunction:: agilerl.utils.algo_utils.get_num_actions

.. autofunction:: agilerl.utils.algo_utils.is_image_space

.. autofunction:: agilerl.utils.algo_utils.concatenate_spaces

Network and Model Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: agilerl.utils.algo_utils.share_encoder_parameters

.. autofunction:: agilerl.utils.algo_utils.get_hidden_states_shape_from_model

.. autofunction:: agilerl.utils.algo_utils.format_shared_critic_encoder

.. autofunction:: agilerl.utils.algo_utils.get_deepest_head_config

.. autofunction:: agilerl.utils.algo_utils.is_peft_model

.. autofunction:: agilerl.utils.algo_utils.clone_llm

Observation Processing
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: agilerl.utils.algo_utils.obs_channels_to_first

.. autofunction:: agilerl.utils.algo_utils.obs_to_tensor

.. autofunction:: agilerl.utils.algo_utils.get_vect_dim

.. autofunction:: agilerl.utils.algo_utils.add_placeholder_value

.. autofunction:: agilerl.utils.algo_utils.maybe_add_batch_dim

.. autofunction:: agilerl.utils.algo_utils.preprocess_observation

.. autofunction:: agilerl.utils.algo_utils.apply_image_normalization

Experience Handling
~~~~~~~~~~~~~~~~~~~

.. autofunction:: agilerl.utils.algo_utils.get_experiences_samples

.. autofunction:: agilerl.utils.algo_utils.stack_experiences

.. autofunction:: agilerl.utils.algo_utils.stack_and_pad_experiences

.. autofunction:: agilerl.utils.algo_utils.flatten_experiences

.. autofunction:: agilerl.utils.algo_utils.is_vectorized_experiences

.. autofunction:: agilerl.utils.algo_utils.vectorize_experiences_by_agent

.. autofunction:: agilerl.utils.algo_utils.experience_to_tensors

.. autofunction:: agilerl.utils.algo_utils.concatenate_tensors

.. autofunction:: agilerl.utils.algo_utils.reshape_from_space

.. autofunction:: agilerl.utils.algo_utils.concatenate_experiences_into_batches

Checkpoint and Serialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: agilerl.utils.algo_utils.make_safe_deepcopies

.. autofunction:: agilerl.utils.algo_utils.isroutine

.. autofunction:: agilerl.utils.algo_utils.recursive_check_module_attrs

.. autofunction:: agilerl.utils.algo_utils.chkpt_attribute_to_device

.. autofunction:: agilerl.utils.algo_utils.key_in_nested_dict

.. autofunction:: agilerl.utils.algo_utils.remove_compile_prefix

.. autofunction:: agilerl.utils.algo_utils.module_checkpoint_dict

.. autofunction:: agilerl.utils.algo_utils.module_checkpoint_single

.. autofunction:: agilerl.utils.algo_utils.module_checkpoint_multiagent

Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: agilerl.utils.algo_utils.CosineLRScheduleConfig

.. autofunction:: agilerl.utils.algo_utils.create_warmup_cosine_scheduler

File and Directory Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: agilerl.utils.algo_utils.remove_nested_files

.. _grpo_tutorial:

LLM Reasoning Tutorial
=======================

Reinforcement learning enhances an LLM's reasoning by rewarding it for solving problems with verifiable answers in domains like
math, coding, and science. Unlike standard training where feedback comes token-by-token, this approach evaluates complete reasoning sequences.
The process typically involves:

#. Starting with a base LLM that has decent language capabilities
#. Presenting it with reasoning-intensive problems that have objectively correct answers
#. Having the model generate entire solution paths, including all intermediate steps
#. Computing rewards based on correctness of the final answer, regardless of the specific reasoning path taken
#. Using these rewards to update the model's parameters through algorithms like PPO or GRPO

In this tutorial, we will be using the Group Relative Policy Optimization (GRPO) algorithm to finetune a
large language model and achieve emergent reasoning behaviour using the AgileRL open-source framework.
To read more about LLM reasoning in general, check out the :ref:`page<llm_finetuning>` in our docs.
Visit the AgileRL GitHub `repository <https://github.com/AgileRL/AgileRL/>`_ for more information about the library.

GRPO Overview
-------------
:ref:`GRPO<grpo>` (Group Relative Policy Optimization) is an elegant simplification of :ref:`PPO<ppo>` (Proximal Policy Optimization)
that makes reinforcement learning more computationally efficient, especially for large language models.

The two key innovations are:

* **Eliminating the critic network:** Instead of training a separate value function to estimate expected rewards (which requires additional compute and memory), GRPO normalizes rewards across a batch of samples. It calculates advantage by subtracting the mean reward from each sample's reward and dividing by the standard deviation.
* **Group-based evaluation:** GRPO generates multiple outputs using the same policy, evaluates them as a group, and then updates the model. This approach reduces variance in the training signal by smoothing out the randomness inherent in probabilistic environments.

These changes are particularly valuable for LLM training because they reduce computational overhead by removing the
need for a separate critic model, provide more stable gradient updates in environments with sparse or noisy rewards,
and they simplify implementation while maintaining or improving performance.

Dependencies
------------

.. code-block:: python

    import re
    from typing import Tuple
    import torch
    from accelerate import Accelerator
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from agilerl.algorithms import GRPO
    from agilerl.training.train_llm import finetune_llm
    from agilerl.utils.llm_utils import HuggingFaceGym


Defining our base model and dataset
-----------------------------------

In this tutorial, we use the open-source transformers and datasets libraries from
`Hugging Face <https://huggingface.co/models>`_ to download our pretrained model weights and training data.
There are a huge number of models and datasets hosted on Hugging Face, and different ones can easily be
substituted in. In this tutorial, to keep things simple and inexpensive, we will use a 3 billion parameter Qwen
model, and the Countdown dataset, and initialise them as follows:

.. collapse:: Create Model and Dataset

    .. code-block:: python

        MODEL_PATH = "Qwen/Qwen2.5-3B"
        DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"

        def create_model(pretrained_model_name_or_path):
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            peft_config = LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "up_proj",
                    "down_proj",
                    "gate_proj",
                ],
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            )
            model = get_peft_model(model, peft_config)
            return model

        def make_dataset(dataset_name: str) -> Tuple[Dataset, Dataset]:
            raw_dataset = (
                load_dataset(dataset_name, split="train").shuffle(seed=42).select(range(50000))
            )
            raw_dataset = raw_dataset.rename_column("target", "answer")
            raw_dataset = raw_dataset.rename_column("nums", "question")
            train_test_split = raw_dataset.train_test_split(test_size=0.1)
            train_dataset = train_test_split["train"]
            test_dataset = train_test_split["test"]
            return train_dataset, test_dataset

        # Instantiate the model and the associated tokenizer
        model = create_model(pretrained_model_name_or_path=MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        train_dataset, test_dataset = make_dataset(DATASET)

Create the Reasoning Environment
--------------------------------
**From model to agent:** In reinforcement learning, models are called agents. This is because they are
trained by taking actions, receiving rewards, and learning from this feedback. This enables them to
become very good at taking actions to solve tasks - to develop *agency*. Since we are training our model
with reinforcement learning, it becomes an agent through this process.

We must create a reinforcement learning environment in which our agent can explore possible
solutions and learn to optimise rewards. AgileRL provides a :class:`HuggingFaceGym <agilerl.utils.llm_utils.HuggingFaceGym>`
class that wraps a Hugging Face dataset and converts it into a reinforcement learning, gymnasium-style environment.

So, how does the environment know how to reward an agent for its outputs? Well, we must define a *reward_function*
that the agent learns to optimise. Following the techniques used in the DeepSeek reasoning `paper <https://arxiv.org/pdf/2501.12948>`_,
we will define our reward function as the sum of two rewards:

* Accuracy rewards: Verifying answers against ground truth. In this tutorial, we will reward the model +1 if the final answer it produces is correct, otherwise 0.
* Format rewards: Encouraging structured reasoning with explicit steps. In this tutorial, we will reward the model +1 if it puts its thinking process between `'<think>'` and `'</think>'` tags, otherwise 0.

Therefore, the maximum score an agent can receive is 2, if it produces the correct answer in the correct format. The
key here is that we never tell the agent which answer it should produce or which format it should use. By giving it rewards
for displaying these behaviours, the agent itself discovers the best way to achieve high rewards and learns the behaviour we desire.

.. collapse:: Reward Functions

    .. code-block:: python

        def format_reward_func(completions, target, **kwargs):
            rewards = []

            for completion, gt in zip(completions, target):
                try:
                    # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
                    completion = "<think>" + completion
                    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
                    match = re.search(regex, completion, re.DOTALL)
                    if match is None or len(match.groups()) != 2:
                        rewards.append(0.0)
                    else:
                        rewards.append(1.0)
                except Exception:
                    rewards.append(0.0)
            return rewards


        def equation_reward_func(completions, target, nums, **kwargs):
            rewards = []

            for completion, gt, numbers in zip(completions, target, nums):
                try:
                    # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
                    completion = "<think>" + completion
                    answer_tags = re.findall(r"<answer>([\s\S]*?)<\/answer>", completion)

                    if len(answer_tags) != 1:
                        rewards.append(0.0)
                        continue

                    equation = answer_tags[0].strip()
                    used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

                    if sorted(used_numbers) != sorted(numbers):
                        rewards.append(0.0)
                        continue

                    allowed_pattern = r"^[\d+\-*/().\s]+$"
                    if not re.match(allowed_pattern, equation):
                        rewards.append(0.0)
                        continue

                    result = eval(equation, {"__builtins__": None}, {})

                    if abs(float(result) - float(gt)) < 1e-5:
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                except Exception as e:
                    rewards.append(0.0)
            return rewards


        def combined_rewards(completion, solution, prompt):
            reward = (
                equation_reward_func([completion], [solution], [prompt])[0]
                + format_reward_func([completion], [solution])[0]
            )

            if reward == 2.0:
                with open("countdown_completions.txt", "a") as text_file:
                    text_file.write(
                        f"Prompt {prompt}" + "\n" + completion + "\n" + "=" * 50 + "\n"
                    )

            return reward

Now we have defined our reward functions, we must also design our prompt. This forms the input given
to the agent and provides the context necessary to complete the task. This is a task-specific feature,
and different reasoning problems will require different chat templates, although they can follow a similar
format. We must also define a function to collate our questions and answers, and standardise their length.
Combining all these components, we can now initialise the HuggingFaceGym object.

.. collapse:: Convert HuggingFace Dataset to Gymnasium Environment

    .. code-block:: python

        def countdown_chat_template(q, a, tokenizer):
            conversation = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Using each number in this tensor only once {tuple(i.item() for i in q)}, create an equation that equals {a.item()}. You "
                        "can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. "
                        "And return the final equation and answer in <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>."
                    )
                },
                {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
            ]
            updated_prompt = tokenizer.apply_chat_template(
                conversation, tokenize=False, continue_final_message=True
            )
            tokenized_prompt = tokenizer(
                [updated_prompt],
                return_tensors="pt",
                padding=True,
                padding_side="left",
                return_attention_mask=True,
            )
            return tokenized_prompt

        def custom_collate_fn(batch):
            answers = torch.tensor([item["answer"] for item in batch])

            # For questions of variable length, we need to pad them
            max_len = max(len(item["question"]) for item in batch)
            questions = torch.zeros(len(batch), max_len, dtype=torch.long)
            for i, item in enumerate(batch):
                q_len = len(item["question"])
                questions[i, :q_len] = torch.tensor(item["question"])

            return {"answer": answers, "question": questions}

        # Convert the HuggingFace dataset into a Gymnasium environment
        env = HuggingFaceGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=combined_rewards,
            apply_chat_template_fn=countdown_chat_template,
            data_batch_size_per_gpu=2,
            custom_collate_fn=custom_collate_fn,
            accelerator=accelerator,
        )

Create a GRPO Agent
-------------------
To allow our model to become an agent and learn through reinforcement learning, we can use the
:class:`GRPO <agilerl.algorithms.GRPO>` class. This class follows the same structure as the other
reinforcement learning algorithms in the AgileRL library.

An important part of training aN LLM to display reasoning bahevaiour is distributed training. They are
called *Large* Language Models for a reason, and unless you are a very lucky individual, you may not
have enough capacity on your individual computer to train even a 'small' LLM. If you want to train a
larger, more powerful model, then this becomes even more infeasible. Instead, we can leverage distributed
training, to share the workload across multiple devices and speed up training. To enable distributed
training in this tutorial, we use deepspeed and accelerate.

.. code-block:: python

    agent = GRPO(
        env.observation_space,
        env.action_space,
        actor_network=model,
        pad_token_id=tokenizer.eos_token_id,
        max_output_tokens=1024,
        batch_size=4,
        group_size=12,
        reduce_memory_peak=True,
        accelerator=Accelerator()
    )

Training and Saving an Agent
----------------------------
The simplest way to train an AgileRL agent is to use the :meth:`finetune_llm() <agilerl.training.train_llm.finetune_llm>` function.
This training function will orchestrate the training process, removing the the need to implement a training loop, and will save
checkpoints of the trained agent that can be used later for inference. It also uses Weights and Biases for tracking.

.. code-block:: python

    finetune_llm(
        pop=[agent],
        env=env,
        evaluation_interval=10,
        wb=True,
        save_elite=True,
        elite_path="path/to/model/directory",
        max_reward=2.0,
        evo_steps=10,
        accelerator=Accelerator()
    )

Configuring Accelerate and DeepSpeed
------------------------------------
To generate an accelerate file, run the command ``accelerate config`` in your terminal, following the instructions
on screen to outline the details of the compute you intend to use for your finetuning, saying yes to the question
"Do you want to use DeepSpeed?" and no to the question "Do you want to specify a json file to a DeepSpeed config?"
if you want an auto-generated deepspeed config file. More information on the deepspeed configuration can be found
in their `docs <https://www.deepspeed.ai/docs/config-json/>`_. The accelerate config will handle the details of
the distribution and the GRPO class handles how the accelerator is used during training. You can then launch a training
run using ``accelerate`` with the following command:

.. code-block:: bash

    accelerate launch path/to/training_script

Alternatively, you can avoid ``accelerate config`` by defining your own accelerate-deepspeed config file and pass
it as an argument to ``accelerate launch``:

.. code-block:: bash

    accelerate launch --config_file path/to/accelerate-deepspeed-config.yaml path/to/training_script

Example config file:

.. code-block:: yaml

    compute_environment: LOCAL_MACHINE
    debug: false
    deepspeed_config:
        gradient_accumulation_steps: 2
        gradient_clipping: 1.0
        offload_optimizer_device: cpu
        offload_param_device: cpu
        zero3_init_flag: false
        zero_stage: 2
    distributed_type: DEEPSPEED
    downcast_bf16: no
    enable_cpu_affinity: false
    machine_rank: 0
    main_training_function: main
    mixed_precision: bf16
    num_machines: 4
    num_processes: 1
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false

Using a custom training loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If we wanted to have more control over the training process, it is also possible to write our own custom
training loop to train our agent. The training loop below can be used alternatively to the above ``finetune_llm``
function and is an example of how we might choose to train our agent to exhibit reasoning.

.. collapse:: Custom Training Loop

    .. code-block:: python

        from tqdm import trange
        import torch.distributed as dist

        def gather_tensor(tensor: Union[torch.Tensor, float], accelerator: Accelerator) -> torch.Tensor:
            """Gather tensors from gpus

            :param tensor: Tensor to gather
            :type tensor: torch.Tensor
            :param accelerator: Accelerator object
            :type accelerator: accelerate.Accelerator
            :return: Stacked tensors
            :rtype: torch.Tensor
            """
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor, device=accelerator.device)
            tensor = tensor.to(accelerator.device)
            gathered_tensors = accelerator.gather(tensor)
            return gathered_tensors


        def aggregate_metrics_across_gpus(
            accelerator: Accelerator, metric_tensor: Union[torch.Tensor, float]
        ) -> float:
            """Aggregate gathered tensors

            :param accelerator: Accelerator object
            :type accelerator: accelerate.Accelerator
            :param metric_tensor: Metrics
            :type metric_tensor: torch.Tensor
            :return: Mean metric
            :rtype: float
            """
            all_metrics = gather_tensor(metric_tensor, accelerator)
            avg_metrics = all_metrics.mean().item()
            return avg_metrics

        evaluation_interval = 5
        max_reward = 2.0
        checkpoint_path="path/to/model/directory"

        if agent.accelerator.is_main_process:
            print("\nTraining...")

        bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
        max_steps = len(env) // env.data_batch_size
        if agent.accelerator.is_main_process:
            pbar = trange(
                max_steps,
                unit="step",
                bar_format=bar_format,
                ascii=True,
                dynamic_ncols=True,
            )

        # calling env.reset() supplies the first batch of training data
        prompts = env.reset(reset_dataloaders=True)
        for i in range(max_steps):
            completion_ids, action_masks = agent.get_action(prompts)
            # Use the reward function stored in env.step to calculate reward of the each answer from the group
            next_prompts, rewards = env.step(completion_ids)
            experiences = (
                completion_ids,
                action_masks,
                rewards,
            )
            loss, kl = agent.learn(experiences)
            metrics = [loss, kl, rewards]
            if max_reward is not None:
                accuracy = (rewards == max_reward).sum() / len(rewards.squeeze())
                metrics.append(accuracy)
            agg_metrics = [aggregate_metrics_across_gpus(agent.accelerator, metric) for metric in metrics]
            prompts = next_prompts
            if agent.accelerator.is_main_process:
                metrics = {
                            "Loss": (agg_metrics[0]),
                            "KL-divergence": (agg_metrics[1]),
                            "Mean training reward": (agg_metrics[2]),
                        }
                if max_reward is not None:
                    metrics |= {"Accuracy": (agg_metrics[3])}
                print(
                    metrics
                )
                pbar.update(1)
                if wb:
                    wandb.log(
                        metrics
                    )
                if (i + 1) % evaluation_interval == 0:
                    test_reward = agent.test(env)
                    print(f"Test reward: {test_reward}")
                    if wb:
                        wandb.log({"Test reward": test_reward})
                if (
                    checkpoint_path is not None
                    and checkpoint_interval is not None
                    and (i + 1) % checkpoint_interval == 0
                ):
                    if agent.accelerator is not None:
                        unwrapped_model = agent.accelerator.unwrap_model(agent.actor)
                        unwrapped_model.save_pretrained(checkpoint_path)
                        print(f"Saved checkpoint {save_path}")
                    else:
                        agent.actor.save_pretrained(checkpoint_path)


Loading a Trained Agent for Inference
-------------------------------------
Once we have finetuned our LLM, we may want to use it for inference. Below outlines how to load the model
in this tutorial, this `forum <https://discuss.huggingface.co/t/save-load-and-do-inference-with-fine-tuned-model/76291/2>`_
provides more info for loading finetuned models.

Load fine-tuned LLM
~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    model = PeftModel.from_pretrained(base_model, "path/to/model/directory")

Inference
~~~~~~~~~

.. code-block:: python

    # Put model in evaluation mode
    model.eval()

    # Tokenize input
    inputs = countdown_chat_template(torch.tensor([33, 19, 27, 5]), # Numbers
                                    torch.tensor([39]),            # Answer
                                    tokenizer)

    # Move inputs to the same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate text (inference)
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,  # Control the length of generated text
            temperature=0.7,     # Control randomness (lower = more deterministic)
            top_p=0.9,           # Nucleus sampling parameter
            do_sample=True,      # Use sampling instead of greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)

The Aha Moment
~~~~~~~~~~~~~~

We finetuned a Qwen2.5-3B using the full training code below and witnessed the model having 'aha' moments. In a notable example from our training run
the model literally exclaims "Aha!" as it reasons its way to the correct solution:

.. code-block:: text

    58 - 1 = 57, then 57 - 25 = 32, and finally 32 + 1 = 33.
    However, 33 is not 82. Let's try another combination: 58 + 25 = 83, but we need 82.
    Aha! If we use 58 + 1 = 59, then 59 - 25 = 34, and finally 34 - 1 = 33. This doesn't work either.
    Hmm... what if we use 58 + 1 = 59, then 59 - 25 = 34, and finally 34 + 1 = 35.
    Nope... closer, but not quite. What if we try 58 + 1 = 59, then 59 - 25 = 34, and finally 34 + 25 = 59.
    Nope... still not 82.
    Ah-ha! One more try: 58 - 1 = 57, then 57 + 25 = 82.</think>
    <answer>(58 - 1) + 25</answer>

Full Training Code
------------------
.. collapse:: Full code

   .. literalinclude:: ../../../tutorials/LLM_Finetuning/grpo_reasoning.py
      :language: python

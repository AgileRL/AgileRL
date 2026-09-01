.. _llm_finetuning_hpo:

LLM Fine-Tuning with HPO
========================

To build on the :ref:`LLM reasoning tutorial<grpo_tutorial>`, we will now introduce how you can perform hyperparameter optimisation (HPO)
on GRPO whilst finetuning an LLM, leading to superior reasoning performance with smaller model sizes. Using our evolutionary approach,
as referenced in the :ref:`evo_hyperparam_opt` section, we can select GRPO hyperparameters to maximise the performance of the LLM finetuning process.

.. note::
    Population-based LLM training with GRPO is computationally intensive by nature and at larger
    population sizes, wall-clock time becomes the bottleneck fast. This tutorial is
    intentionally self-contained and runs agents sequentially, so expect it to be slow if
    you're scaling up.

    Parallelising this efficiently at scale is a hard infrastructure problem, and it's one
    we've spent a lot of time on. `AgileRL Arena <https://arena.agilerl.com>`_ handles the
    scheduling, parallelism, and resource management so you don't have to. If you're planning
    to run serious experiments, it's worth taking a look.

Dependencies
------------

.. code-block:: python

    import re
    import torch
    import yaml
    from accelerate import Accelerator
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from agilerl.algorithms import GRPO
    from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
    from agilerl.utils.algo_utils import VLLMConfig
    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.training.llm import train_llm_rollout
    from agilerl.llm_envs import RolloutHarness
    from agilerl.utils.utils import create_population

Defining Hyperparameters
------------------------
Before we commence training, it's easiest to define all of our hyperparameters in one dictionary. Below is an example of
such for the GRPO algorithm. Additionally, we also define a mutations parameters dictionary, in which we determine what
mutations we want to happen, to what extent we want these mutations to occur, and what RL hyperparameters we want to tune.
Additionally, we also define our upper and lower limits for these hyperparameters to define search spaces. It is worth noting,
unlike the rest of the AgileRL framework, we can only tune the RL hyperparameters and not architecture hyperparameters.

.. collapse:: Hyperparameter Config

    .. code-block:: python

        mut_p = {
            "NO_MUT": 0.1,
            "RL_HP_MUT": 0.6,
            "MUT_SD": 0.1,
            "RAND_SEED": 42,
            "MIN_LR": 0.0000001,
            "MAX_LR": 0.00001,
            "MIN_BETA": 0.0001,
            "MAX_BETA": 0.01,
            "MIN_GROUP_SIZE": 4,
            "MAX_GROUP_SIZE": 12,
        }

        hp_config = HyperparameterConfig(
            beta=RLParameter(min=mut_p["MIN_BETA"], max=mut_p["MAX_BETA"]),
            lr=RLParameter(min=mut_p["MIN_LR"], max=mut_p["MAX_LR"]),
            group_size=RLParameter(min=mut_p["MIN_GROUP_SIZE"], max=mut_p["MAX_GROUP_SIZE"], dtype=int),
        )

        # Algorithm hyperparameters
        init_hp = {
            "hp_config": hp_config,
            "batch_size": 16,
            "beta": 0.001,
            "lr": 5e-6,
            "clip_coef": 0.2,
            "max_grad_norm": 0.1,
            "update_epochs": 1,
            "group_size": 8,
            "temperature": 0.9,
            "max_model_len": 1024,
            "use_vllm": True,
            "vllm_config": VLLMConfig(
                sleep_mode=False,
                max_num_seqs=4,
            ),
            "lora_config": LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
            ),
        }

Defining our Base Model and Dataset
-----------------------------------

In this tutorial, we use the open-source transformers and datasets libraries from
`Hugging Face <https://huggingface.co/models>`_ to download our pretrained model weights and training data.
There are a huge number of models and datasets hosted on Hugging Face, and different ones can easily be
substituted in. In this tutorial, to keep things simple, we will use a 1.5 billion parameter Qwen
model, and the Countdown dataset:

.. collapse:: Model and Dataset Initialisation

    .. code-block:: python

        MODEL_PATH = "Qwen/Qwen2.5-1.5B"
        DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"

        def make_dataset(dataset_name: str) -> tuple[Dataset, Dataset]:
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
        tokenizer.pad_token_id = tokenizer.eos_token_id
        train_dataset, test_dataset = make_dataset(DATASET)

Create the Reasoning Environment
--------------------------------
**From model to agent:** In reinforcement learning, models are called agents. This is because they are
trained by taking actions, receiving rewards, and learning from this feedback. This enables them to
become very good at taking actions to solve tasks - to develop *agency*. Since we are training our model
with reinforcement learning, it becomes an agent through this process.

We must create a reinforcement learning environment in which our agent can explore possible
solutions and learn to optimise rewards. AgileRL provides a :class:`RolloutHarness <agilerl.llm_envs.RolloutHarness>`
class that turns lists of questions and answers into a reinforcement learning, gymnasium-style environment.
Reasoning is the single-turn case, so we build the environment with ``max_turns=1``.

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

                    if sorted(used_numbers) != sorted(numbers.flatten().tolist()):
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
                except Exception:
                    rewards.append(0.0)
            return rewards


        def combined_rewards(completion, solution, prompt):
            reward = (
                equation_reward_func([completion], [solution], [prompt])[0]
                + format_reward_func([completion], [solution])[0]
            )

            return reward

Now we have defined our reward functions, we must also design our prompt. This forms the input given
to the agent and provides the context necessary to complete the task. This is a task-specific feature,
and different reasoning problems will require different conversation templates, although they can follow a similar
format. We define the conversation template as follows (using ``question`` and ``answer`` as placeholders for the question and answer data)
and then drive a single-turn rollout env over the question and answer
columns of our dataset with an in-process ``env_factory`` (a prompt dataset is just an
environment: each rollout runs its own env instance in-process via
:meth:`RolloutHarness.local <agilerl.llm_envs.RolloutHarness.local>`).

.. collapse:: Build the Single-Turn Rollout Environment

    .. code-block:: python

        conversation_template = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You first think about the reasoning process in your mind and then provide the user with the answer.",
            },
            {
                "role": "user",
                "content": "Using each number in this list only once {question}, create an equation that equals {answer}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>.",
            },
            {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
        ]

        # Define accelerators for distributed training
        accelerator = Accelerator()

        def prompt_builder(question: str) -> str:
            parts = [
                m["content"].format(question=question, answer="")
                for m in conversation_template
            ]
            return "\n".join(p for p in parts if p)

        # A single-turn rollout environment from the dataset — a prompt dataset is
        # just an env, driven in-process by RolloutHarness.local.
        class QADataset:
            """Single-turn dataset env: a question on reset, a score on step."""

            def __init__(self, questions, answers, reward_fn, prompt_builder,
                         test_questions=None, test_answers=None):
                self.questions, self.answers = questions, answers
                self.test_questions, self.test_answers = test_questions, test_answers
                self.reward_fn, self.prompt_builder = reward_fn, prompt_builder
                self._cursor, self._split = 0, ""

            @property
            def dataset_size(self) -> int:
                return len(self.questions)

            def reset(self, seed=None, *, row_index=None, evaluation=None):
                if evaluation and self.test_questions is not None:
                    qs, ans, split = self.test_questions, self.test_answers, "eval"
                else:
                    qs, ans, split = self.questions, self.answers, "train"
                if row_index is None:
                    if split != self._split:
                        self._cursor, self._split = 0, split
                    row_index, self._cursor = self._cursor, self._cursor + 1
                self._q, self._a = qs[row_index % len(qs)], ans[row_index % len(ans)]
                return self.prompt_builder(self._q), {}

            def step(self, action):
                return "", float(self.reward_fn(action, self._a, self._q)), True, False, {}

        env_factory = lambda: RolloutHarness.local(
            QADataset(
                questions=list(train_dataset["question"]),
                answers=list(train_dataset["answer"]),
                reward_fn=combined_rewards,
                prompt_builder=prompt_builder,
                test_questions=list(test_dataset["question"]),
                test_answers=list(test_dataset["answer"]),
            ),
            tokenizer,
            max_turns=1,
            pad_id=tokenizer.pad_token_id,
            apply_chat_template=True,
            max_model_len=1024,
        )


Create a population of GRPO Agents
----------------------------------
To allow our model to become an agent and learn through reinforcement learning, we can use the
:class:`GRPO <agilerl.algorithms.GRPO>` class. This class follows the same structure as the other
reinforcement learning algorithms in the AgileRL library. We also define a initialisation dictionaries
for the GRPO hyperparameters and the mutation parameters.

An important part of training an LLM to display reasoning behavaiour is distributed training. They are
called *Large* Language Models for a reason, and are often too large to train on a single GPU. If you want
to train a larger, more powerful model, then this becomes even more infeasible. Instead, we can leverage
distributed training, to share the workload across multiple devices and speed up training. To enable distributed
training in this tutorial, we use deepspeed and accelerate.

.. code-block:: python

    # Initialise the population
    pop = GRPO.population(
        size=4,
        model_name=MODEL_PATH,
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
        accelerator=accelerator,
        **init_hp,
    )

Creating Mutations and Tournament Objects
-----------------------------------------
Tournament selection is used to select the agents from a population which will make up the next generation of agents. If
elitism is used, the best agent from a population is automatically preserved and becomes a member of the next generation.
Then, for each tournament, k individuals are randomly chosen, and the agent with the best evaluation fitness is preserved.
This is repeated until the population for the next generation is full.

The class ``TournamentSelection()`` defines the functions required for tournament selection. ``TournamentSelection.select()``
returns the best agent, and the new generation of agents.

.. code-block:: python

    tournament = TournamentSelection(
        tournament_size=2,
        elitism=True,
        population_size=4,
    )

Mutation is periodically used to explore the hyperparameter space, allowing different hyperparameter combinations to be
trialled during training. If certain hyperparameters prove relatively beneficial to training, then that agent is more
likely to be preserved in the next generation, and so those characteristics are more likely to remain in the population.

The ``Mutations()`` class is used to mutate agents with pre-set probabilities. The available mutations for GRPO currently implemented are:

* No mutation
* RL algorithm mutation - mutation of learning hyperparameter, such as learning rate or batch size.

``Mutations.mutation()`` returns a mutated population. Tournament selection and mutation should be applied sequentially to fully evolve a population between evaluation and learning cycles.

.. code-block:: python

    mutations = Mutations(
        no_mutation=mut_p["NO_MUT"],
        architecture=0,
        new_layer_prob=0,
        parameters=0,
        activation=0,
        rl_hp=mut_p["RL_HP_MUT"],
        mutation_sd=mut_p["MUT_SD"],
        rand_seed=mut_p["RAND_SEED"],
        device=device,
    )

Training and Saving an Agent
----------------------------
The simplest way to train an AgileRL agent is to use the :meth:`train_llm_rollout() <agilerl.training.llm.train_llm_rollout>` function
with ``max_turns=1`` for single-turn reasoning.

.. code-block:: python

    train_llm_rollout(
        pop=pop,
        max_turns=1,
        env_factory=env_factory,
        init_hp=init_hp,
        evaluation_interval=10,
        wb=True,
        save_elite=True,
        elite_path="path/to/model/directory",
        max_reward=2.0,
        evo_steps=10,
        mutation=mutations,
        selection_strategy=tournament,
        accelerator=accelerator,
        verbose=True,
        num_epochs=1
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


Using a Custom Training Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you need lower-level control than :meth:`train_llm_rollout() <agilerl.training.llm.train_llm_rollout>`,
build a :class:`~agilerl.llm_envs.RolloutCollector` and collect trajectories with
:func:`~agilerl.rollouts.on_policy.collect_rollouts_llm`. This is the same rollout API the trainer uses internally.
Do **not** use dataset-env calls like ``env.reset(reset_dataloaders=True)`` / ``env.step(token_ids)``
for rollout training.

.. collapse:: Custom Training Loop

    .. code-block:: python

        import numpy as np
        from agilerl.llm_envs import RolloutCollector
        from agilerl.rollouts.on_policy import collect_rollouts_llm
        from agilerl.utils.llm_utils import aggregate_metrics_across_gpus
        from agilerl.utils.utils import run_selection_and_mutation

        batch_size = init_hp["BATCH_SIZE"]
        group_size = getattr(pop[0], "group_size", 1)
        accelerator = pop[0].accelerator
        rollout_env = RolloutCollector(env_factory, batch_size, group_size)
        group_seed = int(np.random.randint(0, 1_000_000))

        try:
            for i in range(max_steps):
                for agent_idx, agent in enumerate(pop):
                    (
                        token_ids_list,
                        action_masks_list,
                        all_turn_ids,
                        all_rewards,
                        batch_steps,
                        group_seed,
                        all_sampling_logps,
                    ) = collect_rollouts_llm(
                        agent=agent,
                        env=rollout_env,
                        n_steps=1,  # single-turn reasoning
                        batch_size=batch_size,
                        group_size=group_size,
                        group_seed=group_seed,
                    )

                    experiences = (token_ids_list, action_masks_list, all_rewards)
                    learn_kwargs = {"turn_ids": all_turn_ids}
                    if all_sampling_logps is not None:
                        learn_kwargs["sampling_logps"] = all_sampling_logps
                    metrics = agent.learn(experiences, **learn_kwargs)

                    # Example distributed-safe metric aggregation.
                    mean_loss = aggregate_metrics_across_gpus(accelerator, metrics["mean_loss"])

                if tournament and mutation is not None and (i + 1) % evo_steps == 0:
                    pop = run_selection_and_mutation(
                        tournament,
                        population=pop,
                        mutation=mutations,
                        env_name="reasoning_env",
                        accelerator=None,
                        language_model=True,
                        elite_path=elite_path,
                        save_elite=save_elite,
                    )
        finally:
            rollout_env.close()


Loading a Trained Agent for Inference
-------------------------------------
Once we have finetuned our LLM, we may want to use it for inference. Below outlines how to load the model
in this tutorial, this `forum <https://discuss.huggingface.co/t/save-load-and-do-inference-with-fine-tuned-model/76291/2>`_
provides more info for loading finetuned models.


Load fine-tuned LLM into vLLM Engine for inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: python

    from vllm import LLM

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_num_seqs=1024,
        max_model_len=1536,
        distributed_executor_backend="external_launcher",
        seed=0,
        model_impl="vllm",
        enable_lora=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1024,
        seed=42,
    )

    prompts = "Using each number in this list only once 33, 19, 27, 5, create an equation that equals 82. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once."
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest(
            lora_name="trained_model",
            lora_int_id=1,
            lora_path=checkpoint_path + "/actor",
        ),
    )

Full Training Code
------------------
.. collapse:: Full code

   .. literalinclude:: ../../../tutorials/llm_finetuning/grpo_reasoning_hpo.py
      :language: python

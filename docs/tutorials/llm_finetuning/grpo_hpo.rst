.. _llm_finetuning_hpo:

LLM Finetuning with HPO
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
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.training.train_llm import train_llm_rollout
    from agilerl.llm_envs import RolloutEnv
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

        MUTATION_PARAMS = {
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

        INIT_HP = {
            "ALGO": "GRPO",
            "BATCH_SIZE": 16,
            "BETA": 0.001,
            "LR": 0.000005,
            "CLIP_COEF": 0.2,
            "MAX_GRAD_NORM": 0.1,
            "UPDATE_EPOCHS": 1,
            "GROUP_SIZE": 8,
            "TEMPERATURE": 0.9,
            "MAX_MODEL_LEN": 1024,
            "TOURN_SIZE": 2,
            "ELITISM": True,
            "POP_SIZE": 4,
            "EVAL_LOOP": 1,
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
solutions and learn to optimise rewards. AgileRL provides a :class:`RolloutEnv <agilerl.llm_envs.RolloutEnv>`
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
and then host a single-turn rollout env over the question and answer
columns of our dataset with :meth:`RolloutEnv.serving <agilerl.llm_envs.RolloutEnv.serving>`
inside an ``env_factory`` (a prompt dataset is just an environment we host).

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

        def prompt_builder(question: str) -> str:
            parts = [
                m["content"].format(question=question, answer="")
                for m in conversation_template
            ]
            return "\n".join(p for p in parts if p)

        # A single-turn rollout environment from the dataset — a prompt dataset is
        # just an env, hosted on its own server by RolloutEnv.serving.
        class PromptDataset:
            """Single-turn dataset env: serve a question on reset, score it on step."""

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

        def env_factory(evaluation_mode: bool = False):
            env = RolloutEnv.serving(
                lambda: PromptDataset(
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
            env.evaluation_mode = evaluation_mode
            return env


Create a population of GRPO Agents
----------------------------------
To allow our model to become an agent and learn through reinforcement learning, we can use the
:class:`GRPO <agilerl.algorithms.GRPO>` class. This class follows the same structure as the other
reinforcement learning algorithms in the AgileRL library. We also define a initialisation dictionaries
for the GRPO hyperparameters and the mutation parameters.

An important part of training an LLM to display reasoning behavaiour is distributed training. They are
called *Large* Language Models for a reason, and are often too large to train on a single GPU. If you want
to train a larger, more powerful model, then this becomes even more infeasible. Instead, we can leverage
distributed training, to share the workload across multiple devices and speed up training. AgileRL's LLM
algorithms use native ``torch.distributed``: when the script is launched with ``torchrun`` the constructor
initialises the process group automatically, so the same code runs on one GPU or many.

.. code-block:: python

    hp_config = HyperparameterConfig(
        beta=RLParameter(min=mut_p["MIN_BETA"], max=mut_p["MAX_BETA"]),
        lr=RLParameter(min=mut_p["MIN_LR"], max=mut_p["MAX_LR"]),
        group_size=RLParameter(min=mut_p["MIN_GROUP_SIZE"], max=mut_p["MAX_GROUP_SIZE"], dtype=int),
    )

    # Define the algorithm kwargs
    algo_kwargs = {
        "model_name": MODEL_PATH,
        "lora_config": LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        ),
        "use_vllm": True,
        "vllm_config": VLLMConfig(
            sleep_mode=False,
            max_num_seqs=4
        ),
        "pad_token_id": tokenizer.pad_token_id,
        "pad_token": tokenizer.pad_token,
    }

    pop = create_population(
        algo=init_hp["ALGO"],
        net_config=None,
        INIT_HP=init_hp,
        hp_config=hp_config,
        population_size=init_hp["POP_SIZE"],
        algo_kwargs=algo_kwargs,
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
        INIT_HP["TOURN_SIZE"],
        INIT_HP["ELITISM"],
        INIT_HP["POP_SIZE"],
        INIT_HP["EVAL_LOOP"],
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
        no_mutation=MUT_P["NO_MUT"],
        architecture=0,
        new_layer_prob=0,
        parameters=0,
        activation=0,
        rl_hp=MUT_P["RL_HP_MUT"],
        mutation_sd=MUT_P["MUT_SD"],
        rand_seed=MUT_P["RAND_SEED"],
        device=device,
    )

Training and Saving an Agent
----------------------------
The simplest way to train an AgileRL agent is to use the :meth:`train_llm_rollout() <agilerl.training.train_llm.train_llm_rollout>` function
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
        tournament=tournament,
        verbose=True,
    )

Launching distributed training
------------------------------
To train across multiple GPUs, launch the training script with ``torchrun``:

.. code-block:: bash

    torchrun --nproc_per_node 4 path/to/training_script

``torchrun`` sets the standard rendezvous environment variables and the LLM algorithm
constructors call ``init_distributed()`` automatically — no launcher config files and no
changes to the script are needed. For LoRA fine-tuning, plain data parallelism is the
recommended choice: only the small adapter gradients are synchronized between devices,
and the base model weights stay whole on each rank (which colocated vLLM generation requires).

Gradient clipping is configured via the ``max_grad_norm`` argument to the algorithm,
and gradient accumulation via the ``gradient_accumulation_steps`` argument (or
``micro_batch_size_per_gpu``), passed through ``algo_kwargs``. For models too large to
train unsharded, shard the actor with PyTorch FSDP2 by passing an
:class:`~agilerl.utils.distributed.FSDPConfig`:

.. code-block:: python

    from agilerl.utils.distributed import FSDPConfig

    algo_kwargs = {
        ...,
        "gradient_accumulation_steps": 2,
        "fsdp_config": FSDPConfig(
            reshard_after_forward=True,  # ZeRO-3-like memory profile
            cpu_offload=False,
        ),
    }


Using a Custom Training Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If we wanted to have more control over the training process, it is also possible to write our own custom
training loops to train our agents. The training loop below can be used alternatively to the above ``train_llm_rollout``
function and is an example of how we might choose to make use of a population of AgileRL agents in our own training loop.

.. collapse:: Custom Training Loop

    .. code-block:: python

        from agilerl.llm_envs import BatchRolloutEnv
        from agilerl.rollouts.on_policy import collect_rollouts_llm
        from agilerl.utils.algo_utils import stack_and_pad_experiences
        from agilerl.utils.utils import (
            aggregate_metrics_across_gpus,
            tournament_selection_and_mutation,
        )
        from agilerl.utils.distributed import barrier, is_main_process
        from tqdm import trange
        import numpy as np
        import torch

        batch_size = pop[0].batch_size
        group_size = pop[0].group_size
        effective_data_batch_size = batch_size * group_size

        # One RolloutEnv per grouped rollout, driven in lock-step; a separate
        # test env keeps evaluation isolated from mid-rollout training state
        env = BatchRolloutEnv(env_factory, batch_size, group_size)
        test_env = env_factory(evaluation_mode=True)

        evaluation_interval = 10
        max_reward = 2.0
        max_steps = 500
        evo_steps = 10
        group_seed = 42
        elite_path = "path/to/model/directory"
        save_elite = True
        verbose = True

        if is_main_process():
            print("\nTraining...")

        bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
        pbar = trange(
            max_steps * effective_data_batch_size,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
        )

        total_steps = 0
        for i in range(max_steps):
            agent_metrics_dict = {}
            for agent_idx, agent in enumerate(pop):
                # Collect one batch of grouped single-turn episodes
                # (reset -> get_action -> step -> get_episode_data under the hood)
                (
                    completion_ids,
                    action_masks,
                    turn_ids,
                    rewards,
                    batch_steps,
                    group_seed,
                    sampling_logps,
                ) = collect_rollouts_llm(
                    agent=agent,
                    env=env,
                    n_steps=1,  # single-turn reasoning
                    batch_size=batch_size,
                    group_seed=group_seed,
                )
                completion_lengths = np.mean([x.shape[1] for x in completion_ids])

                # Stack per-episode rewards into a (batch, max_turns) tensor
                (rewards_2d,) = stack_and_pad_experiences(
                    [r.unsqueeze(0) if r.dim() == 1 else r for r in rewards],
                    padding_values=[0.0],
                )
                rewards_2d = rewards_2d.float()
                episode_scores = rewards_2d.sum(dim=1)

                experiences = (
                    completion_ids,
                    action_masks,
                    rewards_2d,
                )
                learn_metrics = agent.learn(experiences)
                metrics = [
                    learn_metrics["mean_loss"],
                    learn_metrics["mean_kl"],
                    episode_scores,
                    completion_lengths,
                ]
                if max_reward is not None:
                    accuracy = (episode_scores >= max_reward).float().mean()
                    metrics.append(accuracy)
                agg_metrics = [
                    aggregate_metrics_across_gpus(metric) for metric in metrics
                ]
                agg_test_metrics = None
                if (i + 1) % evaluation_interval == 0:
                    test_reward = agent.test(test_env)
                    agg_test_metrics = [aggregate_metrics_across_gpus(test_reward)]
                    if verbose and is_main_process():
                        fitness = [str(round(agent.fitness[-1], 2)) for agent in pop]
                        avg_fitness = [
                            "%.2f" % np.mean(agent.fitness[-5:]) for agent in pop
                        ]
                        avg_score = ["%.2f" % np.mean(agent.scores[-10:]) for agent in pop]
                        agents = [agent.index for agent in pop]
                        num_steps = [agent.steps[-1] for agent in pop]
                        muts = [agent.mut for agent in pop]
                        print(
                            f"""
                            --- Global Steps {total_steps} ---
                            Fitness:\t\t{fitness}
                            Score:\t\t{mean_scores}
                            5 fitness avgs:\t{avg_fitness}
                            10 score avgs:\t{avg_score}
                            Agents:\t\t{agents}
                            Steps:\t\t{num_steps}
                            Mutations:\t\t{muts}
                            """,
                            end="\r",
                        )
                if is_main_process():
                    metrics_dict = {
                        "Train/Loss": agg_metrics[0],
                        "Train/KL-divergence": agg_metrics[1],
                        "Train/Mean reward": (mean_scores := agg_metrics[2]),
                        "Train/Average completion length": int(agg_metrics[3]),
                    }
                    if max_reward is not None:
                        metrics_dict |= {"Train/Accuracy": agg_metrics[4]}
                    agent_metrics_dict[f"agent_{agent_idx}/train_metrics"] = metrics_dict
                    if agg_test_metrics is not None:
                        test_metrics_dict = {"Eval/Mean reward": agg_test_metrics[0]}
                        agent_metrics_dict[f"agent_{agent_idx}/test_metrics"] = (
                            test_metrics_dict
                        )
                    pbar.update(effective_data_batch_size)
                    agent.steps.append(effective_data_batch_size)
                    agent.scores.append(mean_scores)
                    total_steps += effective_data_batch_size

            barrier()
            if tournament is not None and mutations is not None:
                if (i + 1) % evo_steps == 0:
                    pop = tournament_selection_and_mutation(
                        population=pop,
                        tournament=tournament,
                        mutation=mutations,
                        env_name="countdown",
                        language_model=True,
                        elite_path=elite_path,
                        save_elite=save_elite
                    )
        pbar.close()
        env.close()
        test_env.close()


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

    prompts = "Using each number in this list only once 33, 19, 27, 5, create an equation that equals 82. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.""
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

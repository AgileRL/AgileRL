.. _llm_finetuning_hpo:

LLM Finetuning with HPO
========================

To build on the :ref:`LLM reasoning tutorial<grpo_tutorial>`, we will now introduce how you can perform hyperparameter optimisation (HPO)
on GRPO whilst finetuning an LLM, leading to superior reasoning performance with smaller model sizes. Using our evolutionary approach,
as referenced in the :ref:`evo_hyperparam_opt` section, we can select GRPO hyperparameters to maximise the performance of the LLM finetuning process.


Dependencies
------------

.. code-block:: python

    import re
    from typing import Tuple
    import torch
    import yaml
    from accelerate import Accelerator
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.training.train_llm import finetune_llm
    from agilerl.utils.llm_utils import HuggingFaceGym
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
                "MAX_GROUP_SIZE": 12
            }

        INIT_HP = {
            "ALGO": "GRPO",
            "BATCH_SIZE_PER_GPU": 1,
            "REDUCE_MEMORY_PEAK": True,
            "BETA": 0.001,
            "LR": 0.000005,
            "CLIP_COEF": 0.2,
            "MAX_GRAD_NORM": 0.1,
            "UPDATE_EPOCHS": 1,
            "GROUP_SIZE": 8,
            "TEMPERATURE": 0.9,
            "CALC_POSITION_EMBEDDINGS": True,
            "MIN_OUTPUT_TOKENS": None,
            "MAX_OUTPUT_TOKENS": 1024,
            "COSINE_lR_SCHEDULER": None,
            "TOURN_SIZE": 2,
            "ELITISM": True,
            "POP_SIZE": 4,
            "EVAL_LOOP": 1
        }

Defining our Base Model and Dataset
-----------------------------------

In this tutorial, we use the open-source transformers and datasets libraries from
`Hugging Face <https://huggingface.co/models>`_ to download our pretrained model weights and training data.
There are a huge number of models and datasets hosted on Hugging Face, and different ones can easily be
substituted in. In this tutorial, to keep things simple, we will use a 1.5 billion parameter Qwen
model, and the Countdown dataset, and initialise them as follows:

.. collapse:: Model and Dataset Initialisation

    .. code-block:: python

        MODEL_PATH = "Qwen/Qwen2.5-1.5B"
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
        tokenizer.pad_token_id = tokenizer.eos_token_id
        train_dataset, test_dataset = make_dataset(DATASET)
        INIT_HP["PAD_TOKEN_ID"] = tokenizer.pad_token_id

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

            if reward == 2.0:
                with open("countdown_completions.txt", "a") as text_file:
                    text_file.write(
                        f"Prompt {prompt}" + "\n" + completion + "\n" + "=" * 50 + "\n"
                    )

            return reward

Now we have defined our reward functions, we must also design our prompt. This forms the input given
to the agent and provides the context necessary to complete the task. This is a task-specific feature,
and different reasoning problems will require different chat templates, although they can follow a similar
format. Combining all these components, we can now initialise the HuggingFaceGym object.

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
                    "content": f"Using each number in this tensor only once {tuple(i.item() for i in q)}, create an equation that equals {a.item()}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>.",
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
            # Extract answers and questions
            answers = torch.tensor([item["answer"] for item in batch])

            # For questions of variable length, we need to pad them
            # First, find the maximum length
            max_len = max(len(item["question"]) for item in batch)

            # Create padded tensor
            questions = torch.zeros(len(batch), max_len, dtype=torch.long)
            for i, item in enumerate(batch):
                q_len = len(item["question"])
                questions[i, :q_len] = torch.tensor(item["question"])

            return {"answer": answers, "question": questions}


        # Define accelerators for distributed training
        accelerator = Accelerator()

        # Convert the HuggingFace dataset into a Gymnasium environment
        env = HuggingFaceGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=combined_rewards,
            apply_chat_template_fn=countdown_chat_template,
            data_batch_size=8,
            custom_collate_fn=custom_collate_fn,
            accelerator=accelerator,
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

    hp_config = HyperparameterConfig(
        beta=RLParameter(min=mut_p["MIN_BETA"], max=mut_p["MAX_BETA"]),
        lr=RLParameter(min=mut_p["MIN_LR"], max=mut_p["MAX_LR"]),
        group_size=RLParameter(
            min=mut_p["MIN_GROUP_SIZE"], max=mut_p["MAX_GROUP_SIZE"], dtype=int
        ),
    )

    pop = create_population(
        algo=init_hp["ALGO"],
        observation_space=env.observation_space,
        action_space=env.action_space,
        actor_network=model,
        net_config=None,
        INIT_HP=INIT_HP,
        hp_config=hp_config,
        population_size=init_hp["POP_SIZE"],
        accelerator=accelerator,
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
The simplest way to train an AgileRL agent is to use the :meth:`finetune_llm() <agilerl.training.train_llm.finetune_llm>` function.

.. code-block:: python

    finetune_llm(
        pop=pop,
        env=env,
        init_hp=init_hp,
        evaluation_interval=10,
        wb=True,
        save_elite=True,
        elite_path="path/to/model/directory",
        max_reward=2.0,
        evo_steps=10,
        mutation=mutations,
        tournament=tournament,
        accelerator=accelerator,
        verbose=True,
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
If we wanted to have more control over the training process, it is also possible to write our own custom
training loops to train our agents. The training loop below can be used alternatively to the above ``finetune_llm``
function and is an example of how we might choose to make use of a population of AgileRL agents in our own training loop.

.. collapse:: Custom Training Loop

    .. code-block:: python

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


        accelerator = Accelerator()
        if accelerator is None or accelerator.is_main_process:
            print("\nTraining...")

        bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
        max_steps = len(env) // effective_data_batch_size
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
        )

        total_steps = 0
        # calling env.reset() supplies the first batch of training data
        prompts = env.reset(reset_dataloaders=True)
        for i in range(max_steps):
            agent_metrics_dict = {}
            for agent_idx, agent in enumerate(pop):
                completion_ids, action_masks = agent.get_action(prompts)
                completion_lengths = np.mean([x.shape[1] for x in completion_ids])

                # Use the reward function stored in env.step to calculate reward of the each answer from the group
                next_prompts, rewards = env.step(completion_ids)
                experiences = (
                    completion_ids,
                    action_masks,
                    rewards,
                )
                loss, kl = agent.learn(experiences)
                metrics = [loss, kl, rewards, completion_lengths]
                if max_reward is not None:
                    accuracy = (rewards == max_reward).sum() / len(rewards.flatten())
                    metrics.append(accuracy)
                agg_metrics = [
                    aggregate_metrics_across_gpus(accelerator, metric) for metric in metrics
                ]
                prompts = next_prompts
                agg_test_metrics = None
                if (i + 1) % evaluation_interval == 0:
                    test_reward = agent.test(env)
                    test_metrics = [test_reward]
                    if max_reward is not None:
                        test_accuracy = (test_reward == max_reward).sum() / len(
                            rewards.flatten()
                        )
                        test_metrics.append(test_accuracy)
                    agg_test_metrics = [
                        aggregate_metrics_across_gpus(accelerator, metric)
                        for metric in test_metrics
                    ]
                    if verbose and (accelerator is None or accelerator.is_main_process):
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
                if accelerator is None or accelerator.is_main_process:
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
                        if max_reward is not None:
                            test_metrics_dict |= {"Eval/Accuracy": agg_test_metrics[1]}
                        agent_metrics_dict[f"agent_{agent_idx}/test_metrics"] = (
                            test_metrics_dict
                        )
                    pbar.update(effective_data_batch_size)
                    agent.steps.append(effective_data_batch_size)
                    agent.scores.append(mean_scores)
                    total_steps += effective_data_batch_size

            if accelerator is not None:
                accelerator.wait_for_everyone()
            if tournament and mutation is not None:
                if (i + 1) % evo_steps == 0:
                    pop = tournament_selection_and_mutation(
                        population=pop,
                        tournament=tournament,
                        mutation=mutations,
                        env_name=env.name,
                        accelerator=None,  # Set as None for LLM finetuning as it does not require the same accelerator handling as standard RL models
                        language_model=True,
                        elite_path=elite_path,
                        save_elite=save_elite
                    )
        pbar.close()


Loading a Trained Agent for Inference
-------------------------------------
Once we have finetuned our LLM, we may want to use it for inference. Below outlines how to load the model
in this tutorial, this `forum <https://discuss.huggingface.co/t/save-load-and-do-inference-with-fine-tuned-model/76291/2>`_
provides more info for loading finetuned models.

Load Fine-tuned LLM
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

Full Training Code
------------------
.. collapse:: Full code

   .. literalinclude:: ../../../tutorials/LLM_Finetuning/grpo_reasoning_hpo.py
      :language: python

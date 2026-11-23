.. _tutorial_arena_grpo_gsm8k:

Training GRPO on GSM8K in Arena
================================

This tutorial walks through the full Arena workflow for LLM reasoning: registering a dataset,
writing a reward function, submitting a training job, monitoring progress, and deploying the
fine-tuned model. We fine-tune ``Qwen/Qwen3.5-0.8B`` with **GRPO** on **GSM8K**, a
dataset of grade-school math word problems, as our running example.

Prerequisites
-------------

Installation
~~~~~~~~~~~~

Arena is provided by the separate ``agilerl-arena`` distribution (``agilerl.arena.*``
in the shared namespace). It pulls in lightweight client dependencies (``httpx``,
``rich``, etc.) rather than core training stacks. Install it directly, or install
AgileRL, which depends on it:

.. code-block:: bash

   pip install agilerl-arena
   # or
   pip install agilerl

Authentication
~~~~~~~~~~~~~~

All Arena operations require authentication. You can authenticate in one of two ways:

1. **API Key**: set the ``ARENA_API_KEY`` environment variable:

   .. code-block:: bash

      export ARENA_API_KEY="arena_pat..."

   .. note::

      Personal access tokens can be found in the Arena dashboard under
      *Profile Management* → *CLI API Key*.

2. **Device login** (interactive): run the login command, which opens a browser for OAuth authentication:

   .. tab-set::
      :sync-group: interface

      .. tab-item:: Python
         :sync: python

         .. code-block:: python

            from agilerl.arena import ArenaClient

            client = ArenaClient()
            client.login()

      .. tab-item:: CLI
         :sync: cli

         .. code-block:: bash

            arena login

   Credentials are persisted locally so you only need to authenticate once per machine.

The Task
~~~~~~~~

GRPO (Group Relative Policy Optimization) rewards a model for solving problems that have a
verifiable answer. Instead of training a separate critic to estimate value, GRPO samples a group
of completions for each prompt and scores each one relative to the group mean. This removes the
critic network and gives a stable training signal, which makes it a good fit for reasoning tasks.

Our task is GSM8K. Each row is a ``question`` (a math word problem) and an ``answer`` whose final
line holds the solution after a ``####`` marker:

.. code-block:: text

   question: Natalia sold clips to 48 of her friends in April, and then she sold half as many
             clips in May. How many clips did she sell altogether in April and May?
   answer:   Natalia sold 48/2 = 24 clips in May.
             Altogether she sold 48+24 = 72 clips.
             #### 72

We ask the model to reason step by step and put its final number in ``<answer></answer>`` tags,
then reward it for getting that number right and for using the tags.

Register the Dataset
~~~~~~~~~~~~~~~~~~~~~

Before training, we register the dataset on Arena. We import GSM8K straight from Hugging Face and
map its columns onto the keys Arena's reasoning datasets expect (``question`` and ``answer``).
GSM8K already uses those names, so the mapping is one-to-one.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         client.create_dataset(
             name="gsm8k",
             category="reasoning",
             hf_dataset_name="openai/gsm8k",
             hf_config="main",
             hf_split="train",
             column_mapping={"question": "question", "answer": "answer"},
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena datasets create gsm8k \
             --category reasoning \
             --hf-dataset openai/gsm8k \
             --hf-config main \
             --hf-split train \
             --column-mapping '{"question": "question", "answer": "answer"}'

Datasets fall into one of three categories, and each expects its own column mapping keys:

* ``reasoning``: question and answer pairs for algorithms like GRPO, as here.
  Expects ``{"question": "<column>", "answer": "<column>"}``.
* ``preference``: a prompt with a chosen and a rejected completion, for algorithms like DPO.
  Expects ``{"prompt": "<column>", "chosen": "<column>", "rejected": "<column>"}``.
* ``sft``: supervised fine-tuning pairs mapping a prompt to a target completion.
  Expects ``{"prompt": "<column>", "target": "<column>"}``.

GRPO trains on ``reasoning`` datasets.

You can browse datasets already registered for your organization, or search Hugging Face, with
``arena datasets list`` (add ``--search`` to query Hugging Face).

The Reward Function
~~~~~~~~~~~~~~~~~~~

A reasoning job needs a reward function: a Python file that scores each completion the model
produces. Arena calls a function named ``reward_fn`` with three arguments, ``completion``,
``answer`` and ``question``, and expects a single ``float`` back. ``completion`` is the model's
whole turn, so it includes any reasoning the model did before answering.

Our reward reads only the text after the last ``</think>``, or the whole completion when the model
did not think, so a number the model tries while reasoning does not count. It has two parts:

* **Correctness** returns ``1.0`` when the number in the last ``<answer>`` block matches the
  number after ``####`` in the dataset's answer.
* **Format** returns ``1.0`` when that text ends with exactly one ``<answer>`` block that holds
  only a number.

The most a completion can score is ``2.0``. The reward does not require ``<think>`` tags, so it
scores the same whether the model's thinking mode is on or off.

.. collapse:: reward.py

   .. literalinclude:: /_static/examples/gsm8k-grpo/reward.py
      :language: python

The reward function runs on Arena's infrastructure during training. Arena also runs it once when
you submit the job, as a quick check that it imports and returns a number.

Creating a Project
~~~~~~~~~~~~~~~~~~

Before submitting a training job, we need a project to submit it to. Language-model runs live in
an LLM-based project, so we pass that flag when creating it.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         client.create_project(name="GSM8K Tutorial", description=None, llm_based=True)

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena projects create "GSM8K Tutorial" --llm-based

.. tip::

   You can set a default project to work on by using the ``arena projects set-default <project-name>`` CLI command.

Submit a Training Job
~~~~~~~~~~~~~~~~~~~~~

We define the training configuration in a YAML manifest (``gsm8k_grpo.yaml``). The ``environment``
section is a ``rollout`` over the dataset we registered:

* ``name`` and ``dataset`` both give the dataset's registered name.
* ``rubric_file_path`` names the reward file. It matches the file we pass with ``--reward-file``.
* ``prompt_template`` turns each row into the chat the model sees: a system message that asks for
  step-by-step reasoning and an ``<answer>`` tag, then the row's ``question``.
* ``chat_template_kwargs: {enable_thinking: false}`` keeps Qwen3.5's thinking block closed, so the
  model reasons in plain text before its answer. ``Qwen/Qwen3.5-0.8B`` already defaults to this;
  set it to ``true`` to train the model's thinking mode instead. The reward works either way.
* ``max_reward: 2.0`` counts a completion as solved when it scores full marks.

``replay_buffer: {kind: llm}`` is the buffer GRPO needs, because Arena runs LLM training
asynchronously: rollout engines generate while the trainer learns. The ``network`` section names
the base model, ``Qwen/Qwen3.5-0.8B``, and a LoRA adapter, so training only updates the adapter
weights.

.. collapse:: gsm8k_grpo.yaml

   .. literalinclude:: /_static/examples/gsm8k_grpo.yaml
      :language: yaml

Check the manifest before you submit. ``arena manifest validate`` runs the same manifest checks as
``arena experiments submit``, without sending anything:

.. code-block:: bash

   arena manifest validate gsm8k_grpo.yaml

We submit the manifest together with the reward function. Passing ``--reward-file`` tells Arena
this is a reasoning job, and it validates the reward function against a sample completion before
the run starts.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         result = client.submit_experiment(
             manifest="gsm8k_grpo.yaml",
             reward_file="reward.py",
             project="GSM8K Tutorial",
             experiment_name="gsm8k-grpo-v1",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena experiments submit gsm8k_grpo.yaml \
             --reward-file reward.py \
             --project 'GSM8K Tutorial' \
             --experiment-name gsm8k-grpo-v1

Choosing GPUs
~~~~~~~~~~~~~

Leave out ``--gpu-type`` and ``--num-gpus``, as above. Arena then sizes the run with its memory
estimate and picks the cheapest GPUs that fit. Asynchronous GRPO needs at least two GPUs: one
trains the adapter and one runs the vLLM rollout engine. A 0.8B model fits on two single-GPU
nodes, or on one 8-GPU node.

LLM training runs on the GPU types ``arena resources list`` shows. Arena cloud L4 and A100 GPUs are
not available for LLM training. To choose the GPUs yourself, pass both flags, for example
``--gpu-type h100-80 --num-gpus 8``, and check the fit first:

.. code-block:: bash

   arena resources list
   arena memory estimate gsm8k_grpo.yaml --gpu h100-80 --num-gpus 8

Monitor Training
~~~~~~~~~~~~~~~~

You can monitor training progress directly from the Arena dashboard, or download
metrics programmatically. ``--overwrite`` (``overwrite=True`` in Python) replaces
``metrics.csv`` when you download again:

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         # Download training metrics
         client.download_experiment_metrics(
             experiment_name="gsm8k-grpo-v1",
             output_path="metrics.csv",
             overwrite=True,
         )

         # List available checkpoints
         checkpoints = client.list_checkpoints(
             experiment_name="gsm8k-grpo-v1"
         )
         print(checkpoints)

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Download metrics
         arena experiments metrics gsm8k-grpo-v1 --output-file metrics.csv --overwrite

         # List checkpoints
         arena experiments checkpoints gsm8k-grpo-v1

Deploy the Trained Model
~~~~~~~~~~~~~~~~~~~~~~~~~

Once training is complete, deploy the best checkpoint to an inference endpoint:

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         # Deploy the best checkpoint
         client.deploy_agent(experiment_name="gsm8k-grpo-v1")

         # Or deploy a specific checkpoint
         client.deploy_agent(
             experiment_name="gsm8k-grpo-v1",
             checkpoint="step_500",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Deploy the best checkpoint
         arena agent deploy gsm8k-grpo-v1

         # Deploy a specific checkpoint
         arena agent deploy gsm8k-grpo-v1 --checkpoint step_500

Interact with the Deployed Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After deployment, open the model by name and send it a prompt to receive a completion. The
deployment name matches the experiment you deployed; you can also list your deployments with
``arena agent list``.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from agilerl.arena import ArenaClient

         prompt = "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"

         with ArenaClient() as client:
             with client.open_inference_agent("gsm8k-grpo-v1") as agent:
                 # Generate a full completion
                 result = agent.generate(prompt)
                 print(result.results[0].completion)

                 # Or stream tokens as they are produced
                 for chunk in agent.generate_stream(prompt):
                     print(chunk, end="")

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Always streams tokens as they are generated
         arena agent generate gsm8k-grpo-v1 \
             --prompt "Weng earns \$12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"

.. tip::

   Set a default agent with ``arena agent run gsm8k-grpo-v1`` and you can drop the deployment
   name from later commands. ``arena agent generate --prompt "..."`` then uses the active agent.

.. seealso::

   :ref:`arena_client` for the full reference on all Arena client methods.

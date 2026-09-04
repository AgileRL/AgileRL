.. _tutorial_arena_grpo_gsm8k:

Training GRPO on GSM8K in Arena
================================

This tutorial walks through the full Arena workflow for LLM reasoning: registering a dataset,
writing a reward function, submitting a training job, monitoring progress, and deploying the
fine-tuned model. We fine-tune ``Qwen/Qwen2.5-0.5B-Instruct`` with **GRPO** on **GSM8K**, a
dataset of grade-school math word problems, as our running example.

It is the language-model counterpart to :ref:`tutorial_arena_end_to_end`. Where that tutorial
validates a custom Gymnasium environment, here we register a Hugging Face dataset and supply a
reward function instead.

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

We reward the model for producing the correct final number, and for wrapping its reasoning and
answer in the format we ask for.

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
produces. Arena calls a function named ``reward`` with three arguments, ``question``, ``answer``,
and ``completion``, and expects a single ``float`` back. The higher the reward, the better the
completion.

Our reward has two parts. A **correctness** reward parses the gold answer after ``####``, extracts
the final number from the model's ``<answer>`` block, and returns ``1.0`` when they match. A
**format** reward returns ``1.0`` when the completion wraps its reasoning and answer in
``<think>...</think>`` and ``<answer>...</answer>`` tags. The most a completion can score is
``2.0``. We never tell the model which answer or format to produce; it discovers both from the
reward.

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
section references the dataset by the name we registered it under. The ``network`` section names
the base model and its LoRA adapter, so training only updates the adapter weights.

.. collapse:: gsm8k_grpo.yaml

   .. literalinclude:: /_static/examples/gsm8k_grpo.yaml
      :language: yaml

Unlike the :ref:`PPO tutorial <tutorial_arena_end_to_end>`, there is no observation space for Arena
to infer an encoder from. The model architecture comes entirely from
``pretrained_model_name_or_path`` and the ``lora_config``.

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
             resource_id="arena-medium",
             num_nodes=2,
             project="GSM8K Tutorial",
             experiment_name="gsm8k-grpo-v1",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena experiments submit gsm8k_grpo.yaml \
             --reward-file reward.py \
             --resource-id arena-medium \
             --num-nodes 2 \
             --project 'GSM8K Tutorial' \
             --experiment-name gsm8k-grpo-v1

.. tip::

   You can view all of the available resources to train on by running the CLI command ``arena resources list``.

Monitor Training
~~~~~~~~~~~~~~~~

You can monitor training progress directly from the Arena dashboard, or download
metrics programmatically:

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         # Download training metrics
         client.download_experiment_metrics(
             experiment_name="gsm8k-grpo-v1",
             output_path="metrics.csv",
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
         arena experiments metrics gsm8k-grpo-v1 --output-file metrics.csv

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

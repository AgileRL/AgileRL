.. _tutorial_arena_grpo_llm_judge:

Training GRPO with an LLM Judge in Arena
========================================

This tutorial trains an LLM on a reward that mixes a programmatic check with an **LLM
judge**. A judge is another LLM, called through its provider's API, that reads each
completion and scores it against criteria you specify. It suits rewards
that are hard to code, like whether the working is clear, while a code check keeps the
parts you can verify exactly.

We build on :ref:`tutorial_arena_grpo_gsm8k`: the same GSM8K dataset, project, model and
GRPO settings. Read that one first if you have not registered the ``gsm8k`` dataset yet.
Here the reward is:

* **Correctness** (weight 0.6): ``1.0`` when the final number matches the GSM8K answer.
* **Judge** (weight 0.4): a score from 0 to 1 for how clear and sound the working is,
  from ``gpt-4.1-mini``.

At the end we swap the per-completion judge for **RULER**, where one judge call compares
every completion in a GRPO group.

Prerequisites
-------------

* ``agilerl-arena`` installed and logged in (see :ref:`tutorial_arena_grpo_gsm8k`).
* The ``gsm8k`` reasoning dataset registered and the ``GSM8K Tutorial`` project created,
  as in that tutorial.
* An API key for the judge's provider. This tutorial uses OpenAI; any
  OpenAI-compatible endpoint, or Anthropic, also works.

The example files are in ``docs/_static/examples/gsm8k-judge/``: ``reward.py``,
``manifest.yaml`` and ``manifest_ruler.yaml``.

Store the Judge's API Key
-------------------------

The training job calls the judge, so it needs your provider key. Arena keeps it as an
**organization secret**: the manifest names the secret, and Arena passes it to the job as
an environment variable of that name. The key never goes in the manifest.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import os

         client.set_secret(name="OPENAI_API_KEY", value=os.environ["OPENAI_API_KEY"])

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Prompts for the value without echoing it
         arena secrets set OPENAI_API_KEY

         # Or read it from a local environment variable
         arena secrets set OPENAI_API_KEY --from-env OPENAI_API_KEY

Run ``arena secrets list`` to check it is there. Arena shows each secret's name and when it
was last updated, never the value. You can also add and replace secrets in the web app: open
your account settings, then the **Organization** tab, and use the **Organization secrets**
card. The secret belongs to your organization, so one key serves every run in it. See
:ref:`arena_secrets` for more.

The Reward File
---------------

A rubric file can export a plain ``reward_fn``, as in the GSM8K tutorial, or an OpenEnv
``Rubric``. A judge needs the second form. The harness calls the rubric on every turn with a
:class:`~agilerl.llm_envs.rubrics.TurnAction` (the model's reply as ``action.message``) and a
:class:`~agilerl.llm_envs.rubrics.TurnObservation` (the dataset row's ``question`` and
``answer``, among other fields).

.. collapse:: reward.py

   .. literalinclude:: /_static/examples/gsm8k-judge/reward.py
      :language: python

What each piece does:

* ``Correctness`` is a plain ``Rubric``. It parses the gold number after ``####`` in
  ``observation.answer`` and the number inside the reply's ``<answer>`` tags, and returns
  ``1.0`` when they match. ``component_name = "correct"`` names its metric.
* :class:`~agilerl.llm_envs.rubrics.LLMJudgeRubric` takes the judge client and your
  criteria. It shows the judge the question and the reply, plus the row's answer as
  ``REFERENCE ANSWER``, and asks for ``{"score": <0 to 1>, "reason": "..."}``. A reply it
  cannot read scores 0.
* ``RUBRIC`` combines them with OpenEnv's ``WeightedSum``. The weights must add up to 1.
* ``RUBRIC.__init__`` takes a ``judge`` argument. That is how the judge reaches the
  rubric: when the manifest has a ``judge`` block, Arena builds the rubric class with the
  run's shared judge client. A rubric that is built at import time, such as a module-level
  instance, cannot receive it, so ``RUBRIC`` must be a class.
* ``RULER`` is only used in the variant at the end. It is ignored until a manifest names
  it.

Write criteria the way you would brief a human grader: what a good reply does, and what to
mark down. The judge also sees the reference answer, so the criteria can refer to it.

.. note::

   The judge sees the reply without its reasoning block. Pass
   ``LLMJudgeRubric(judge, CRITERIA, include_reasoning=True)`` to show it the reasoning
   too. That trains the model to write reasoning the judge likes, so leave it off unless
   that is what you want.

The Manifest
------------

The manifest is the GSM8K one with three additions to ``environment``:

.. collapse:: manifest.yaml

   .. literalinclude:: /_static/examples/gsm8k-judge/manifest.yaml
      :language: yaml

* ``rubric_file_path: reward.py`` is the reward file. It must have the same file name as
  the file you pass to ``--reward-file``. You can leave it out and let ``--reward-file``
  fill it in.
* ``rubric_name: RUBRIC`` is the symbol to load from that file. Without it, Arena looks for
  ``reward_fn``.
* ``judge`` describes the judge:

  * ``provider``: ``openai`` (the default) or ``anthropic``. ``openai`` covers any
    OpenAI-compatible endpoint.
  * ``model``: the judge's model id.
  * ``api_key_env``: the **name** of the organization secret holding the key, never the key.
  * ``max_tokens``: the judge's reply budget (default 4096). A judge that thinks before it
    answers spends this budget on its thinking too, so keep it high for thinking models or
    the verdict gets cut off. 512 is plenty for ``gpt-4.1-mini``.

The judge block also takes ``base_url``, ``max_concurrency`` (requests in flight per
process, default 16), ``timeout_s`` (seconds per attempt, default 120) and ``temperature``
(default 0; Anthropic judges reject it). To use a judge hosted somewhere other than OpenAI,
set ``base_url``. For example, through OpenRouter:

.. code-block:: yaml

   judge:
       provider: openai
       model: openai/gpt-4.1-mini
       base_url: https://openrouter.ai/api/v1
       api_key_env: OPENROUTER_API_KEY
       max_tokens: 512

A URL with a path is used as given. A bare host, such as a self-hosted vLLM server at
``http://judge:8000``, gets ``/v1`` added. Leave ``api_key_env`` out only when the endpoint
needs no key; without ``base_url`` it is required.

The prompt asks the model to show its working and put the final number in
``<answer></answer>`` tags, which is what ``Correctness`` parses and the criteria check.

.. note::

   On a dataset registered in Arena, Arena currently trains with the dataset's default
   prompt in place of ``prompt_template``. That prompt also asks for ``<answer></answer>``
   tags, so this reward works either way. Check the prompt in the experiment's Completions
   view before relying on a custom one.

.. tip::

   Validate the manifest before you submit. ``arena experiments submit`` validates it too,
   but this is quicker while editing:

   .. code-block:: python

      from agilerl.arena.models.manifest import TrainingManifest

      TrainingManifest.get_validated("manifest.yaml")

Submit the Job
--------------

Submit the manifest with the reward file. The judge's key is not passed here; the job reads
it from the organization secret.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         result = client.submit_experiment(
             manifest="manifest.yaml",
             reward_file="reward.py",
             project="GSM8K Tutorial",
             experiment_name="gsm8k-judge-v1",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena experiments submit manifest.yaml \
             --reward-file reward.py \
             --project 'GSM8K Tutorial' \
             --experiment-name gsm8k-judge-v1

Every completion now costs a judge call. With ``group_size: 8`` that is eight calls per
prompt, so check your provider's rate limits and spend before a long run.
``max_concurrency`` caps how many calls each process has in flight. The provider SDK
retries rate limits, server errors and timeouts. A call that still fails is an episode
failure. Arena drops that group and keeps training. The run stops after 8 episode
failures in a row. The RULER variant below uses the same rule.

Watch the Judge
---------------

On top of the usual training metrics, a judge run logs:

.. list-table::
   :widths: 32 68
   :header-rows: 1

   * - Metric
     - What it tells you
   * - ``reward_correct``
     - The ``Correctness`` score, before weighting.
   * - ``reward_judge``
     - The judge's score, before weighting.
   * - ``judge_calls``
     - Judge calls since the last log.
   * - ``judge_parse_failure_rate``
     - Share of judge replies with no readable score. Each one scores 0, so a high rate
       is noise in your reward. Check ``max_tokens`` first, then the judge model.
   * - ``judge_mean_latency_s``
     - Mean seconds per judge call. If it climbs, the judge is slowing training down: raise
       ``max_concurrency`` or use a faster model.

The success and accuracy metrics count a completion as solved when its reward reaches
``max_reward``, which Arena sets to 1 for a reasoning dataset. With the weights here, only a
correct answer that the judge scores 1 reaches it, so read ``reward_correct`` for accuracy.

Download them as in the GSM8K tutorial:

.. code-block:: bash

   arena experiments metrics gsm8k-judge-v1 --output-file metrics.csv

To see *why* the judge scored a completion, open the experiment in the Arena web app and go
to its **Completions** view. For a run with a judge, each completion has a **Judge notes**
column: the judge's one-sentence ``reason``, labelled with the rubric component that wrote
it (``judge`` here). When the judge's reply could not be read, the note is the raw reply,
which usually shows what went wrong.

Read a sample of notes early in the run. If the judge rewards something you did not mean,
such as long answers, tighten the criteria and resubmit. A judge that is easy to please
trains a model to please it.

Variant: RULER
--------------

GRPO already compares the completions in a group: each one's advantage is its reward minus
the group mean. **RULER** gives the judge the same view. Instead of scoring each completion
alone, one call shows the judge every completion in the group and asks for a score for each.
Comparing replies side by side is often easier for a judge than giving each one an absolute
score, and a group of eight needs one call instead of eight.

:class:`~agilerl.llm_envs.rubrics.LLMGroupJudgeRubric` implements it. ``reward.py`` already
defines ``RULER``; it uses the same ``__init__(self, judge)`` pattern:

.. code-block:: python

   class RULER(LLMGroupJudgeRubric):
       def __init__(self, judge) -> None:
           super().__init__(judge, CRITERIA)

To use it, the manifest names it as ``group_rubric_name`` and points ``rubric_name`` at
``Correctness`` alone, so the judge only runs once per group:

.. code-block:: yaml

   environment:
       rubric_file_path: reward.py
       rubric_name: Correctness
       group_rubric_name: RULER
       judge:
           provider: openai
           model: gpt-4.1-mini
           api_key_env: OPENAI_API_KEY
           max_tokens: 512

The full file is ``manifest_ruler.yaml``. Submit it the same way:

.. code-block:: bash

   arena experiments submit manifest_ruler.yaml \
       --reward-file reward.py \
       --project 'GSM8K Tutorial' \
       --experiment-name gsm8k-ruler-v1

How it differs from the per-completion judge:

* Each group score is added to that completion's ``Correctness`` reward before the
  success check, so a completion can score up to 2. It is logged as
  ``reward_group_judge``. Accuracy still uses Arena's ``max_reward`` of 1, so a
  completion counts when ``Correctness`` plus the group score is at least 1: every
  correct answer, and a wrong answer the judge scores 1.
* Leave ``rubric_name`` out and the turn reward on dataset rows is 0, so the group score is
  the whole reward.
* The judge sees the completions in a new random order on every call, so a judge that
  favours one position adds noise rather than bias.
* A reply it cannot read, or with the wrong number of scores, scores the whole group 0
  and counts in ``judge_parse_failure_rate``.
* A judge call that still fails after the SDK's retries fails that group's episode
  tasks, which drops the group. The run stops after 8 episode failures in a row.
* Raise ``max_tokens`` with ``group_size``: the judge writes one reason per completion.
* Judge notes still appear per completion in the Completions view, labelled
  ``group_judge`` (the ``LLMGroupJudgeRubric`` component name).
* Held-out evaluation episodes get no group score.

.. seealso::

   :ref:`arena_client` for the full CLI and Python SDK reference, and the "An LLM judge"
   and "Scoring a group together (RULER)" sections of :ref:`llm_environments` for every
   rubric and judge option.

.. _sft:

Supervised Fine-Tuning (SFT)
====================================

"`SFT <https://arxiv.org/pdf/2109.01652>`__ is a post-training technique used to align LLM responses to a set of desired responses using a dataset of (prompt, response) pairs. This technique is the simplest way to shift a model's behaviour toward a target style or task and does not utilise reinforcement learning."

It's similar to a continuation of the pre-training stage of an LLM, but using a curated dataset that is specific to the LLM's application. Cross-entropy loss is computed exclusively on the **response tokens**, so the model is
never penalised for how it encodes the prompt.

SFT is typically the *first* stage of a two-step alignment pipeline:

* **SFT** (this class) — warm-up the model to follow instructions by minimising cross-entropy on ``(prompt, good_response)`` pairs.
* **DPO** — further align the SFT-initialised model using ``(prompt, chosen_response, rejected_response)`` triples.

This technique is surprisingly effective, as pre-trained LLMs have been shown to easily adapt to a relatively small
amount of new data.

Example
-------

.. code-block:: python

  from agilerl.algorithms.sft import SFT
  from agilerl.utils.llm_utils import SFTGym
  from accelerate import Accelerator
  from datasets import load_dataset
  from peft import LoraConfig
  from transformers import AutoModelForCausalLM, AutoTokenizer
  import torch

  # Instantiate the model and the associated tokenizer
  model = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen2.5-3B",
      torch_dtype=torch.bfloat16,
  )
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

  # Instantiate an accelerator object for distributed training
  accelerator = Accelerator()

  # Load the dataset into an SFTGym environment
  raw_dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset", split="train").shuffle(seed=42)
  train_test_split = raw_dataset.train_test_split(test_size=0.1)
  train_dataset = train_test_split["train"]
  test_dataset = train_test_split["test"]
  env = SFTGym(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    tokenizer=tokenizer,
    data_batch_size_per_gpu=16,
    response_column="chosen",
    accelerator=accelerator,
  )

  # Configure LoRA adapters
  lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
  )

  # Instantiate the agent
  agent = SFT(
    actor_network=model,
    pad_token_id=tokenizer.eos_token_id,
    pad_token=tokenizer.eos_token,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=32,
    lr=5e-5,
    update_epochs=1,
    lora_config=lora_config,
    seed=42,
    reduce_memory_peak=True,
    accelerator=accelerator,
  )

Training an SFT agent
---------------------

To train an SFT agent on a single SFT gym environment, use the :ref:`finetune_llm_sft<finetune_llm_sft>` function:

.. code-block:: python

  from agilerl.training.train_llm import finetune_llm_sft

  finetune_llm_sft(
    pop=[agent],
    env=env,
    init_hp={"BATCH_SIZE": 32, "UPDATE_EPOCHS": 1},
    checkpoint_steps=250,
    accelerator=accelerator,
  )

Saving and Loading Agents
--------------------------

To save an agent, use the :ref:`save_llm_checkpoint<save_llm_checkpoint>` function:

.. code-block:: python

  from agilerl.utils.utils import save_llm_checkpoint

  save_llm_checkpoint(agent, "path/to/checkpoint")


To load a trained model, you must use the HuggingFace `.from_pretrained` method, AgileRL is
compatible with HuggingFace and Peft models:

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

Parameters
------------

.. autoclass:: agilerl.algorithms.sft.SFT
  :members:
  :inherited-members:

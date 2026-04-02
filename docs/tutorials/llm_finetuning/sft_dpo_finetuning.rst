.. _llm_finetuning_hpo:

LLM Fine-Tuning with SFT and DPO
========================

To build on the :ref:`LLM reasoning tutorial<grpo_tutorial>`, we will now introduce how you can perform hyperparameter optimisation (HPO)
on GRPO whilst finetuning an LLM, leading to superior reasoning performance with smaller model sizes. 

Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) are two widely used LLM finetuning algorithms. 
SFT is a simple algorithm that fine-tunes an LLM on a dataset of human-generated examples, 
while DPO is a more advanced algorithm that fine-tunes an LLM on a dataset of human preferences.

SFT, also known as instruction tuning, uses a supervised learning approach to fine-tune the LLM. It calculates a simple cross-entropy loss between the model's output logits for each token and the target token.
DPO, on the other hand, constructs an implicit reward function by comparing the model's output logits for each token with the "chosen" and "rejected" tokens. 
The objective is to maximize the output logits similarity to the chosen tokens and minimize similarity to the rejected tokens.
To prevent "reward hacking" leading to nonsensical outputs, an additional KL-divergence term (controlled by a \beta parameter) is added to the loss function to limit divergence from the base model.

Both methods make use of Low Rank Adaptation (LoRA) to fine-tune the LLM, a technique that allows for fine-tuning the LLM with a small number of parameters. 
Recent work has shown this to be just as effective as full fine-tuning (in which every parameter of the base model is updated), but much more compute efficient (https://thinkingmachines.ai/blog/lora/).

In this tutorial, we show how to run each of the algorithms in the AgileRL framework using an open source model and dataset. 

We will use the Qwen2.5-0.5B model (https://huggingface.co/Qwen/Qwen2.5-0.5B) 
and the Human-Like-DPO-Dataset dataset (https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset), 
which can run on a cheap L4 GPU instance or a sufficently souped-up laptop.

First, we look at SFT, then DPO, then combine them in a pipeline SFT->DPO and compare the outputs.


Getting Started
---------------

Take a look at the `benchmarking/benchmarking_sft.py` script for a full example of how to run SFT.

.. code-block:: python

    python benchmarking/benchmarking_sft.py

This will run SFT on the Human-Like-DPO-Dataset dataset using the Qwen2.5-0.5B model. Don't worry if you haven't downloaded the model or dataset - Huggingface will take care of this the first time you run the script, then cache the files for future use.

The first block of code applies to model's tokenizer to the dataset, and creates an SFTGym environment. This is a wrapper around the dataset that allows for easy training of the LLM.

.. code-block:: python

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset, test_dataset = make_dataset(DATASET)
    env = SFTGym(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        data_batch_size_per_gpu=16,
        response_column="chosen",
        accelerator=accelerator,
    )

The next block of code configures the LoRA adapter and instantiates the SFT agent.

.. code-block:: python

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    agent = SFT(
        actor_network=model,
        pad_token_id=tokenizer.eos_token_id,
        pad_token=tokenizer.eos_token,
        batch_size=16,
        lr=5e-5,
        update_epochs=1,
        lora_config=lora_config,
        accelerator=accelerator,
    )

% Include an illustration of LoRA



global HAS_LLM_DEPENDENCIES
try:
    import peft

    del peft
    HAS_LLM_DEPENDENCIES = True
except ImportError:
    HAS_LLM_DEPENDENCIES = False

global HAS_LLM_DEPENDENCIES
try:
    import peft

    HAS_LLM_DEPENDENCIES = True
except ImportError:
    HAS_LLM_DEPENDENCIES = False

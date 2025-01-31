import warnings

# Suppress specific syntax warnings
warnings.filterwarnings(
    "ignore", category=SyntaxWarning, message=".*assertion is always true.*"
)

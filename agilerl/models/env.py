from pydantic import BaseModel, Field


class EnvironmentSpec(BaseModel):
    """Pydantic model for Environment object.

    :param name: The name of the environment as seen in Arena.
    :type name: str
    :param version: The version of the environment. Defaults to "latest".
    :type version: str
    :param num_envs: The number of environments to run in parallel. Defaults to 16.
    :type num_envs: int
    """

    name: str
    version: str = Field(default="latest")
    num_envs: int = Field(default=16, ge=1)

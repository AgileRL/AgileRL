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

    @classmethod
    def from_dict(cls, data: dict[str, str | int]) -> "EnvironmentSpec":
        """Create an EnvironmentSpec from a dictionary.

        :param data: Dictionary containing environment configuration.
        :type data: dict[str, Any]
        :returns: EnvironmentSpec instance.
        :rtype: EnvironmentSpec
        """
        return cls(
            name=data["name"],
            version=data.get("version", "latest"),
            num_envs=data.get("num_envs", 16),
        )

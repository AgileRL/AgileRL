# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Image-token rows of a context-parallel forward with images.

Every CP rank runs the vision tower on all images of the forward and keeps
the feature rows of the image tokens in its own sequence shard. The layout
publishes the full-row ids and positions the shards are cut from in
:data:`IMAGE_TOKEN_SHARD` for the duration of the forward.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

import torch

from agilerl.distributed.context_parallel.sequence import shard_for_cp

__all__ = ["IMAGE_TOKEN_SHARD", "ImageTokenShard"]


@dataclass(frozen=True)
class ImageTokenShard:
    """Full rows of a CP forward with images, and this rank's shard of them.

    :param input_ids: ``(rows, L)`` token ids every rank's shard is cut from.
    :param position_ids: ``(rows, L)`` text positions of ``input_ids``.
    :param rank: This rank in the CP group.
    :param cp: Context-parallel degree.
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    rank: int
    cp: int

    def shard(self, tensor: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """This rank's slice of a full-row tensor along ``seq_dim``."""
        return shard_for_cp(tensor, self.rank, self.cp, seq_dim)

    def feature_rows(self, image_token_id: int) -> torch.Tensor:
        """Rows of the concatenated image features this shard's image tokens read, in token order.

        Image features concatenate in token order, so image token ``k`` of the
        flattened full rows reads feature row ``k``.

        :param image_token_id: Token the image features replace.
        :return: ``(n,)`` feature row of each image token in this shard.
        """
        is_image = self.input_ids == image_token_id
        ordinal = is_image.flatten().cumsum(0).view_as(is_image) - 1
        return self.shard(ordinal)[self.shard(is_image)]


#: The running CP forward's full rows, or ``None`` outside a CP forward with images.
IMAGE_TOKEN_SHARD: ContextVar[ImageTokenShard | None] = ContextVar(
    "image_token_shard", default=None
)

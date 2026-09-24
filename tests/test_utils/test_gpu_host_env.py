# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import os

from tests.gpu_host_env import apply


class TestGpuHostEnvApply:
    def test_overwrites_host_gib_nccl_net(self, monkeypatch):
        monkeypatch.setenv("NCCL_NET", "gIB")
        monkeypatch.setenv(
            "NCCL_TUNER_CONFIG_PATH", "/usr/local/gib/configs/tuner.txtpb"
        )
        monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/gib/lib64:/usr/lib")

        apply()

        assert os.environ["NCCL_NET"] == "Socket"
        assert os.environ["NCCL_IB_DISABLE"] == "1"
        assert "NCCL_TUNER_CONFIG_PATH" not in os.environ
        assert "/gib/" not in os.environ["LD_LIBRARY_PATH"]

# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

from agilerl.hpo.mutation import Mutations, reinit_shared_networks
from agilerl.modules.mlp import EvolvableMLP


class TestReinitSharedNetworksDecorator:
    def test_recompiles_and_reinits_shared_networks(self, monkeypatch, device):
        class DummyGroup:
            eval_network = "actor"
            shared_networks: ClassVar[list[str]] = ["target_actor"]

            def eval_network_name(self):
                return self.eval_network

            def shared_network_names(self):
                return self.shared_networks

        class DummyIndividual:
            mut = "arch"
            torch_compiler = "default"
            accelerator = None
            registry = type("R", (), {"groups": [DummyGroup()]})()

            def __init__(self, device_name):
                self.device = device_name
                self.actor = EvolvableMLP(4, 2, [8], device=device_name)
                self.target_actor = EvolvableMLP(4, 2, [8], device=device_name)
                self.recompile_calls = 0

            def recompile(self):
                self.recompile_calls += 1

        muts = Mutations(0, 1, 0.5, 0, 0, 0, 0.1, device=device)
        individual = DummyIndividual(device)

        def fake_mut(self, ind):
            return ind

        fake_mut = reinit_shared_networks(fake_mut)

        compile_calls = []

        def fake_compile(model, mode):
            compile_calls.append(mode)
            return model

        monkeypatch.setattr(
            "agilerl.utils.mutation_utils.compile_model",
            fake_compile,
        )
        monkeypatch.setattr(
            muts,
            "_reinit_from_mutated",
            lambda eval_net, remove_prefix=False: eval_net.clone(),
        )

        out = fake_mut(muts, individual)

        assert out.recompile_calls == 1
        assert len(compile_calls) == 1
        assert compile_calls[0] == "default"

import gymnasium as gym
import pytest
import torch

from agilerl.algorithms import (
    CQN,
    DDPG,
    DQN,
    IPPO,
    MADDPG,
    MATD3,
    PPO,
    TD3,
    NeuralTS,
    NeuralUCB,
    RainbowDQN,
)
from agilerl.algorithms.core import MultiAgentRLAlgorithm
from agilerl.modules import (
    EvolvableCNN,
    EvolvableLSTM,
    EvolvableMLP,
    EvolvableMultiInput,
    EvolvableResNet,
    EvolvableSimBa,
    ModuleDict,
)
from agilerl.networks import (
    ContinuousQNetwork,
    DeterministicActor,
    QNetwork,
    RainbowQNetwork,
    StochasticActor,
    ValueNetwork,
)
from agilerl.protocols import (
    AgentWrapperProtocol,
    EvolvableAlgorithmProtocol,
    EvolvableModuleProtocol,
    EvolvableNetworkProtocol,
    LoraConfigProtocol,
    ModuleDictProtocol,
    MutationMethodProtocol,
    MutationRegistryProtocol,
    MutationType,
    OptimizerConfig,
    OptimizerLikeClass,
    PeftModelProtocol,
    PreTrainedModelProtocol,
    PretrainedConfigProtocol,
)
from tests.helper_functions import (
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_multidiscrete_space,
    generate_random_box_space,
)


@pytest.fixture
def action_space(network_cls: type[EvolvableNetworkProtocol]) -> gym.Space | None:
    if issubclass(
        network_cls,
        (DeterministicActor, StochasticActor, ContinuousQNetwork),
    ):
        return generate_random_box_space(shape=(2,))
    if issubclass(network_cls, (RainbowQNetwork, QNetwork)):
        return generate_discrete_space(2)
    return None


class TestProtocols:
    """Test that classes implement their respective protocols."""

    @pytest.mark.parametrize(
        "algorithm_cls, algo_action_space",
        [
            (CQN, generate_discrete_space(2)),
            (DQN, generate_discrete_space(2)),
            (RainbowDQN, generate_discrete_space(2)),
            (NeuralTS, generate_discrete_space(2)),
            (NeuralUCB, generate_discrete_space(2)),
            (DDPG, generate_random_box_space(shape=(2,))),
            (TD3, generate_random_box_space(shape=(2,))),
            (PPO, generate_random_box_space(shape=(2,))),
            (PPO, generate_multidiscrete_space(2, 2)),
            (PPO, generate_discrete_space(2)),
            (IPPO, generate_discrete_space(2)),
            (IPPO, generate_random_box_space(shape=(2,))),
            (IPPO, generate_multidiscrete_space(2, 2)),
            (MADDPG, generate_random_box_space(shape=(2,), low=-1.0, high=1.0)),
            (MATD3, generate_random_box_space(shape=(2,), low=-1.0, high=1.0)),
        ],
    )
    @pytest.mark.parametrize(
        "observation_space",
        [
            generate_random_box_space(shape=(4,)),
            generate_random_box_space(shape=(3, 32, 32)),
            generate_dict_or_tuple_space(2, 2),
            generate_discrete_space(6),
            generate_multidiscrete_space(4, 4),
        ],
    )
    def test_algorithm_instances_implement_evolvable_algorithm_protocol(
        self,
        algorithm_cls,
        algo_action_space,
        observation_space,
    ):
        """Test that algorithm instances implement the EvolvableAlgorithm protocol."""
        # Skip actual instantiation, just check the class definitions
        if issubclass(algorithm_cls, MultiAgentRLAlgorithm):
            instance = algorithm_cls(
                observation_spaces=[observation_space],
                action_spaces=[algo_action_space],
                agent_ids=["agent_0"],
            )
        else:
            instance = algorithm_cls(observation_space, algo_action_space)

        assert isinstance(instance, EvolvableAlgorithmProtocol)

    @pytest.mark.parametrize(
        "network_cls",
        [
            QNetwork,
            RainbowQNetwork,
            ContinuousQNetwork,
            ValueNetwork,
            DeterministicActor,
            StochasticActor,
        ],
    )
    @pytest.mark.parametrize(
        "observation_space",
        [
            generate_random_box_space(shape=(4,)),
            generate_random_box_space(shape=(3, 32, 32)),
            generate_dict_or_tuple_space(2, 2),
            generate_discrete_space(6),
            generate_multidiscrete_space(4, 4),
        ],
    )
    def test_network_instances_implement_evolvable_network_protocol(
        self,
        network_cls,
        observation_space,
        action_space,
    ):
        """Test that network instances implement the EvolvableNetwork protocol."""
        kwargs = {}
        if issubclass(network_cls, RainbowQNetwork):
            kwargs["support"] = torch.linspace(-100, 100, 51)

        # Create network instance
        network = network_cls(observation_space, action_space, **kwargs)
        assert isinstance(network, EvolvableNetworkProtocol)

    @pytest.mark.parametrize(
        "module_cls",
        [
            EvolvableMLP,
            EvolvableLSTM,
            EvolvableSimBa,
            EvolvableResNet,
            EvolvableCNN,
            EvolvableMultiInput,
        ],
    )
    def test_module_instances_implement_evolvable_module_protocol(self, module_cls):
        """Test that module instances implement the EvolvableModule protocol."""
        kwargs = {"num_outputs": 10}
        if issubclass(module_cls, (EvolvableCNN, EvolvableResNet)):
            kwargs["input_shape"] = (3, 32, 32)
            kwargs["channel_size"] = [8, 8]
            kwargs["kernel_size"] = [3, 3]
            kwargs["stride_size"] = [1, 1]
        elif issubclass(module_cls, (EvolvableMLP, EvolvableSimBa)):
            kwargs["num_inputs"] = 4
            kwargs["hidden_size"] = [16, 32]
        elif issubclass(module_cls, EvolvableLSTM):
            kwargs["input_size"] = 4
            kwargs["hidden_state_size"] = 16
        elif issubclass(module_cls, EvolvableMultiInput):
            kwargs["observation_space"] = generate_dict_or_tuple_space(2, 2)

        if issubclass(module_cls, (EvolvableSimBa, EvolvableResNet)):
            kwargs["num_blocks"] = 2
            if issubclass(module_cls, EvolvableResNet):
                kwargs["channel_size"] = 8
                kwargs["kernel_size"] = 3
                kwargs["stride_size"] = 1
            else:
                kwargs["hidden_size"] = 16

        instance = module_cls(**kwargs)
        assert isinstance(instance, EvolvableModuleProtocol)

        mutation_methods = instance.get_mutation_methods()
        for method_name, method in mutation_methods.items():
            assert isinstance(
                method,
                MutationMethodProtocol,
            ), (
                f"Mutation method {method_name} does not implement MutationMethod protocol"
            )

    def test_pretrained_model_instances_implement_pretrained_model_protocol(self):
        """Test that HuggingFace PreTrainedModel instances satisfy PreTrainedModelProtocol."""
        pytest.importorskip("transformers")
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=100,
            n_positions=64,
            n_embd=32,
            n_layer=1,
            n_head=2,
        )
        model = GPT2LMHeadModel(config)
        assert isinstance(model, PreTrainedModelProtocol)

    def test_peft_model_instances_implement_peft_model_protocol(self):
        """Test that PEFT PeftModel instances satisfy PeftModelProtocol."""
        pytest.importorskip("transformers")
        pytest.importorskip("peft")
        from peft import LoraConfig, get_peft_model
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=100,
            n_positions=64,
            n_embd=32,
            n_layer=1,
            n_head=2,
        )
        base_model = GPT2LMHeadModel(config)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(base_model, lora_config)
        assert isinstance(peft_model, PeftModelProtocol)

    @pytest.mark.parametrize("module_cls", [EvolvableMLP, EvolvableCNN])
    def test_evolvable_module_protocol_methods_executed(self, module_cls):
        if module_cls is EvolvableMLP:
            kwargs = {"num_outputs": 4, "num_inputs": 4, "hidden_size": [8]}
            x = torch.randn(2, 4)
        else:
            kwargs = {
                "num_outputs": 4,
                "input_shape": [3, 8, 8],
                "channel_size": [4],
                "kernel_size": [3],
                "stride_size": [1],
            }
            x = torch.randn(2, 3, 8, 8)
        mod = module_cls(**kwargs)
        _ = EvolvableModuleProtocol.activation.fget(mod)
        EvolvableModuleProtocol.change_activation(mod, "relu", False)
        EvolvableModuleProtocol.forward(mod, x)
        EvolvableModuleProtocol.parameters(mod)
        EvolvableModuleProtocol.to(mod, "cpu")
        _ = EvolvableModuleProtocol.state_dict(mod)
        EvolvableModuleProtocol.disable_mutations(mod)
        _ = EvolvableModuleProtocol.get_mutation_methods(mod)
        _ = EvolvableModuleProtocol.get_mutation_probs(mod, 0.5)
        _ = EvolvableModuleProtocol.sample_mutation_method(mod, 0.5, None)
        _ = EvolvableModuleProtocol.clone(mod)
        EvolvableModuleProtocol.load_state_dict(mod, mod.state_dict())
        for m in mod.get_mutation_methods().values():
            MutationMethodProtocol.__call__(m, mod)

    @pytest.mark.parametrize("network_cls", [QNetwork, ContinuousQNetwork])
    def test_evolvable_network_protocol_methods_executed(self, network_cls):
        obs = generate_random_box_space(shape=(4,))
        act = (
            generate_discrete_space(2)
            if network_cls is QNetwork
            else generate_random_box_space(shape=(2,))
        )
        net = network_cls(obs, act)
        lat = torch.randn(2, 8)
        x = torch.randn(2, 4)
        _ = EvolvableNetworkProtocol.forward_head(net, lat)
        _ = EvolvableNetworkProtocol.extract_features(net, x)
        _ = EvolvableNetworkProtocol.add_latent_node(net, 1)
        _ = EvolvableNetworkProtocol.remove_latent_node(net, 1)
        EvolvableNetworkProtocol.recreate_encoder(net)
        _ = EvolvableNetworkProtocol.initialize_hidden_state(net, 2)
        EvolvableNetworkProtocol.init_weights_gaussian(net)
        head_cfg = net.head_net.net_config
        EvolvableNetworkProtocol.build_network_head(net, head_cfg)
        enc_cfg = net.encoder_config
        EvolvableNetworkProtocol._build_encoder(net, enc_cfg)

    @pytest.mark.parametrize(
        "algo_cls,obs,act",
        [
            (
                DQN,
                generate_random_box_space(shape=(4,)),
                generate_discrete_space(2),
            ),
            (
                PPO,
                generate_random_box_space(shape=(4,)),
                generate_random_box_space(shape=(2,)),
            ),
        ],
    )
    def test_evolvable_algorithm_protocol_methods_executed(self, algo_cls, obs, act):
        inst = algo_cls(obs, act)
        EvolvableAlgorithmProtocol.unwrap_models(inst)
        EvolvableAlgorithmProtocol.wrap_models(inst)
        _ = EvolvableAlgorithmProtocol.get_action(inst, obs.sample())
        _ = EvolvableAlgorithmProtocol.evolvable_attributes(inst)
        _ = EvolvableAlgorithmProtocol.evolvable_attributes(inst, networks_only=True)
        _ = EvolvableAlgorithmProtocol.inspect_attributes(inst)
        _ = EvolvableAlgorithmProtocol.inspect_attributes(inst, input_args_only=True)
        EvolvableAlgorithmProtocol.recompile(inst)
        EvolvableAlgorithmProtocol.mutation_hook(inst)

    def test_module_dict_protocol_methods_executed(self):
        mdl = ModuleDict(
            {"a": EvolvableMLP(num_inputs=4, num_outputs=2, hidden_size=[8])},
        )
        assert isinstance(mdl, ModuleDictProtocol)
        _ = ModuleDictProtocol.__getitem__(mdl, "a")
        _ = ModuleDictProtocol.keys(mdl)
        _ = ModuleDictProtocol.values(mdl)
        _ = ModuleDictProtocol.items(mdl)
        _ = ModuleDictProtocol.modules(mdl)
        _ = ModuleDictProtocol.get_mutation_methods(mdl)
        ModuleDictProtocol.filter_mutation_methods(mdl, "add")
        _ = ModuleDictProtocol.mutation_methods.fget(mdl)
        _ = ModuleDictProtocol.layer_mutation_methods.fget(mdl)
        _ = ModuleDictProtocol.node_mutation_methods.fget(mdl)

    def test_optimizer_config_and_registry_protocols_executed(self):
        from agilerl.algorithms.core.registry import (
            MutationRegistry,
            OptimizerConfig as RegistryOptimizerConfig,
        )

        opt_cfg = RegistryOptimizerConfig(
            name="opt",
            networks="net",
            lr="1e-3",
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={},
        )
        _ = OptimizerConfig.get_optimizer_cls(opt_cfg)
        reg = MutationRegistry()
        _ = MutationRegistryProtocol.networks(reg)

    @pytest.mark.parametrize(
        "algo_cls,obs,act",
        [
            (
                DQN,
                generate_random_box_space(shape=(4,)),
                generate_discrete_space(2),
            ),
            (
                PPO,
                generate_random_box_space(shape=(4,)),
                generate_random_box_space(shape=(2,)),
            ),
        ],
    )
    def test_evolvable_algorithm_load_checkpoint_learn_test_clone(
        self, algo_cls, obs, act, tmp_path
    ):
        inst = algo_cls(obs, act)
        EvolvableAlgorithmProtocol.save_checkpoint(inst, str(tmp_path / "ckpt.pt"))
        EvolvableAlgorithmProtocol.load_checkpoint(
            inst, str(tmp_path / "ckpt.pt"), "cpu", None
        )
        _ = EvolvableAlgorithmProtocol.load(algo_cls, str(tmp_path / "ckpt.pt"))
        exp = (
            (
                torch.randn(4, 4),
                torch.randn(4, 2),
                torch.randn(4),
                torch.randn(4, 4),
                torch.zeros(4, dtype=torch.bool),
            )
            if algo_cls is DQN
            else (
                torch.randn(4, 4),
                torch.randn(4, 2),
                torch.randn(4),
                torch.randn(4),
                torch.randn(4),
                torch.randn(4),
                torch.zeros(4, dtype=torch.bool),
            )
        )
        EvolvableAlgorithmProtocol.learn(inst, exp)
        _ = EvolvableAlgorithmProtocol.test(inst)
        _ = EvolvableAlgorithmProtocol.clone(inst, None, False)

    def test_agent_wrapper_protocol_methods_executed(self):
        from agilerl.wrappers.agent import RSNorm

        obs = generate_random_box_space(shape=(4,))
        act = generate_discrete_space(2)
        algo = DQN(obs, act)
        wrapper = RSNorm(algo)
        _ = AgentWrapperProtocol.get_action(wrapper, obs.sample())
        AgentWrapperProtocol.learn(
            wrapper,
            (
                torch.randn(4, 4),
                torch.randn(4, 1).long(),
                torch.randn(4),
                torch.randn(4, 4),
                torch.zeros(4, dtype=torch.bool),
            ),
        )

    def test_lora_config_pretrained_config_generation_config_protocols_executed(
        self, tmp_path
    ):
        pytest.importorskip("transformers")
        pytest.importorskip("peft")
        from peft import LoraConfig
        from transformers import GenerationConfig, GPT2Config

        lora = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        )
        assert isinstance(lora, LoraConfigProtocol)
        _ = lora.r
        _ = lora.lora_alpha
        _ = lora.target_modules
        _ = lora.task_type
        _ = lora.lora_dropout

        config = GPT2Config(
            vocab_size=100, n_positions=64, n_embd=32, n_layer=1, n_head=2
        )
        _ = PretrainedConfigProtocol.to_dict(config)
        _ = PretrainedConfigProtocol.to_json_string(config)
        PretrainedConfigProtocol.save_pretrained(config, str(tmp_path))
        _ = PretrainedConfigProtocol.from_dict(config.to_dict())
        _ = PretrainedConfigProtocol.from_pretrained(str(tmp_path))
        _ = PretrainedConfigProtocol.from_json_file(str(tmp_path / "config.json"))

        gen_cfg = GenerationConfig()
        _ = gen_cfg.do_sample
        _ = gen_cfg.temperature
        _ = gen_cfg.pad_token_id

    def test_optimizer_like_class_protocol_executed(self):
        OptimizerLikeClass.__call__(
            torch.optim.SGD,
            [torch.nn.Parameter(torch.zeros(1))],
            0.01,
        )

    def test_mutation_type_enum_values(self):
        assert MutationType.LAYER.value == "layer"
        assert MutationType.NODE.value == "node"
        assert MutationType.ACTIVATION.value == "activation"

    def test_pretrained_and_peft_protocol_methods_executed(self, tmp_path):
        pytest.importorskip("transformers")
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=100, n_positions=64, n_embd=32, n_layer=1, n_head=2
        )
        model = GPT2LMHeadModel(config)
        PreTrainedModelProtocol.eval(model)
        PreTrainedModelProtocol.train(model, True)
        PreTrainedModelProtocol.forward(model, torch.randint(0, 100, (1, 4)))
        PreTrainedModelProtocol.parameters(model)
        _ = PreTrainedModelProtocol.state_dict(model)
        PreTrainedModelProtocol.load_state_dict(model, model.state_dict())
        PreTrainedModelProtocol.to(model, "cpu")
        _ = PreTrainedModelProtocol.generate(
            model, torch.randint(0, 100, (1, 4)), generation_config=None
        )

        pytest.importorskip("peft")
        from peft import LoraConfig, get_peft_model

        base = GPT2LMHeadModel(config)
        peft_model = get_peft_model(
            base,
            LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["c_attn"],
                lora_dropout=0.0,
                task_type="CAUSAL_LM",
            ),
        )
        PeftModelProtocol.eval(peft_model)
        PeftModelProtocol.train(peft_model, True)
        PeftModelProtocol.parameters(peft_model)
        _ = PeftModelProtocol.state_dict(peft_model)
        PeftModelProtocol.load_state_dict(
            peft_model, peft_model.state_dict(), strict=True
        )
        PeftModelProtocol.to(peft_model, "cpu")
        _ = PeftModelProtocol.generate(
            peft_model, torch.randint(0, 100, (1, 4)), generation_config=None
        )
        peft_model.save_pretrained(str(tmp_path))
        base2 = GPT2LMHeadModel(config)
        _ = PeftModelProtocol.from_pretrained(base2, str(tmp_path))

    def test_protocol_type_aliases_importable(self):
        from agilerl.protocols import (
            DeviceType,
            EvolvableAttributeDict,
            EvolvableNetworkDict,
            NumpyObsType,
            ObservationType,
            TorchObsType,
        )

        assert NumpyObsType is not None
        assert TorchObsType is not None

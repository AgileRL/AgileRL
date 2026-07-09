"""Tests for agilerl.train — parse_args and main."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import agilerl.train as train_mod


class TestParseArgs:
    def test_manifest_positional(self):
        with patch("sys.argv", ["train", "config.yaml"]):
            args = train_mod.parse_args()
        assert args.manifest == Path("config.yaml")

    def test_device_flag(self):
        with patch("sys.argv", ["train", "m.yaml", "-d", "cuda:1"]):
            args = train_mod.parse_args()
        assert args.device == "cuda:1"

    def test_use_accelerator_flag(self):
        with patch("sys.argv", ["train", "m.yaml", "--use-accelerator"]):
            args = train_mod.parse_args()
        assert args.use_accelerator is True

    def test_wb_flag(self):
        with patch("sys.argv", ["train", "m.yaml", "--wb"]):
            args = train_mod.parse_args()
        assert args.wb is True

    def test_wandb_api_key(self):
        with patch("sys.argv", ["train", "m.yaml", "--wandb-api-key", "key123"]):
            args = train_mod.parse_args()
        assert args.wandb_api_key == "key123"

    def test_checkpoint_flags(self):
        with patch(
            "sys.argv",
            [
                "train",
                "m.yaml",
                "--checkpoint-steps",
                "50",
                "--checkpoint-path",
                "/tmp/ckpt",
            ],
        ):
            args = train_mod.parse_args()
        assert args.checkpoint_steps == 50
        assert args.checkpoint_path == "/tmp/ckpt"

    def test_overwrite_checkpoints(self):
        with patch("sys.argv", ["train", "m.yaml", "--overwrite-checkpoints"]):
            args = train_mod.parse_args()
        assert args.overwrite_checkpoints is True

    def test_resume_from_checkpoint(self):
        with patch(
            "sys.argv", ["train", "m.yaml", "--resume-from-checkpoint", "/ckpt"]
        ):
            args = train_mod.parse_args()
        assert args.resume_from_checkpoint == "/ckpt"

    def test_save_elite_and_path(self):
        with patch(
            "sys.argv", ["train", "m.yaml", "--save-elite", "--elite-path", "/elite"]
        ):
            args = train_mod.parse_args()
        assert args.save_elite is True
        assert args.elite_path == "/elite"

    def test_tensorboard_flag(self):
        with patch("sys.argv", ["train", "m.yaml", "--tensorboard"]):
            args = train_mod.parse_args()
        assert args.tensorboard is True

    def test_no_tensorboard_flag(self):
        with patch("sys.argv", ["train", "m.yaml", "--no-tensorboard"]):
            args = train_mod.parse_args()
        assert args.tensorboard is False

    def test_tensorboard_log_dir(self):
        with patch("sys.argv", ["train", "m.yaml", "--tensorboard-log-dir", "/logs"]):
            args = train_mod.parse_args()
        assert args.tensorboard_log_dir == "/logs"

    def test_verbose_default(self):
        with patch("sys.argv", ["train", "m.yaml"]):
            args = train_mod.parse_args()
        assert args.verbose is True

    def test_no_verbose_flag(self):
        with patch("sys.argv", ["train", "m.yaml", "--no-verbose"]):
            args = train_mod.parse_args()
        assert args.verbose is False

    def test_defaults(self):
        with patch("sys.argv", ["train", "m.yaml"]):
            args = train_mod.parse_args()
        assert args.use_accelerator is False
        assert args.wb is False
        assert args.wandb_api_key is None
        assert args.checkpoint_steps is None
        assert args.checkpoint_path is None
        assert args.overwrite_checkpoints is False
        assert args.resume_from_checkpoint is None
        assert args.save_elite is False
        assert args.elite_path is None
        assert args.tensorboard is False
        assert args.tensorboard_log_dir == "tensorboard_logs"


class TestMain:
    def test_main_calls_from_manifest_and_train(self):
        mock_trainer = MagicMock()
        mock_trainer.algorithm_spec.name = "PPO"
        mock_trainer.env_spec.name = "CartPole-v1"
        mock_trainer.training_spec.pop_size = 6
        mock_trainer.training_spec.max_steps = 100
        mock_trainer.train.return_value = ([], [0.5])

        with patch.object(
            train_mod.LocalTrainer, "from_manifest", return_value=mock_trainer
        ) as mock_from:
            with patch("sys.argv", ["train", "config.yaml", "-d", "cpu"]):
                train_mod.main()

            mock_from.assert_called_once_with(
                manifest=Path("config.yaml"),
                resume_from_checkpoint=None,
                device="cpu",
                accelerator=None,
            )
        mock_trainer.train.assert_called_once_with(
            wb=False,
            wandb_api_key=None,
            tensorboard=False,
            tensorboard_log_dir="tensorboard_logs",
            checkpoint_steps=None,
            checkpoint_path=None,
            overwrite_checkpoints=False,
            save_elite=False,
            elite_path=None,
            verbose=True,
        )

    def test_main_with_accelerator(self):
        mock_trainer = MagicMock()
        mock_trainer.algorithm_spec.name = "DQN"
        mock_trainer.env_spec.name = "LunarLander-v2"
        mock_trainer.training_spec.pop_size = 4
        mock_trainer.training_spec.max_steps = 50
        mock_trainer.train.return_value = ([], [1.0])

        mock_accel_instance = MagicMock()

        with patch.object(
            train_mod.LocalTrainer, "from_manifest", return_value=mock_trainer
        ) as mock_from:
            with patch.object(
                train_mod, "Accelerator", return_value=mock_accel_instance
            ) as mock_accel:
                with patch(
                    "sys.argv", ["train", "m.yaml", "--use-accelerator", "-d", "cpu"]
                ):
                    train_mod.main()

            mock_accel.assert_called_once()
            call_kwargs = mock_from.call_args[1]
            assert call_kwargs["accelerator"] is mock_accel_instance

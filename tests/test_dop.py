from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from backends import flux2, krea2
from musubi_tuner.training.dop import (
    add_dop_loss,
    dop_signature,
    is_valid_dop_cache,
    make_class_caption,
    validate_cached_signature,
    validate_dop_config,
)
from musubi_tuner_gui import MusubiTunerGUI
from musubi_tuner.flux_2_train_network import DiTOutput, Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.training.parser_common import setup_parser_common


def test_make_class_caption_replaces_literal_trigger_case_insensitively():
    assert make_class_caption("Portrait of ALICE beside Alice", "Alice", "woman") == "Portrait of woman beside woman"


def test_make_class_caption_rejects_missing_trigger():
    with pytest.raises(ValueError, match="Every training caption"):
        make_class_caption("portrait of a woman", "Alice", "woman")


def test_dop_config_rejects_same_trigger_and_class():
    with pytest.raises(ValueError, match="must be different"):
        validate_dop_config("Person", "person", 1.0)


def test_cached_signature_detects_stale_class_configuration():
    batch = {"dop_signature": dop_signature("Alice", "woman").unsqueeze(0)}
    validate_cached_signature(batch, "Alice", "woman")
    with pytest.raises(ValueError, match="different trigger word"):
        validate_cached_signature(batch, "Alice", "person")


def test_dop_cache_validator_reuses_only_matching_caption_and_signature(tmp_path):
    path = tmp_path / "text.safetensors"
    save_file(
        {
            "dop_signature": dop_signature("Alice", "a woman"),
            "dop_ctx_vec_0": torch.ones(1),
        },
        str(path),
        metadata={"caption1": "portrait of Alice"},
    )
    item = SimpleNamespace(text_encoder_output_cache_path=str(path), caption="portrait of Alice")
    assert is_valid_dop_cache(item, "Alice", "a woman", "dop_ctx_vec_")
    item.caption = "close-up of Alice"
    assert not is_valid_dop_cache(item, "Alice", "a woman", "dop_ctx_vec_")
    item.caption = "portrait of Alice"
    assert not is_valid_dop_cache(item, "Alice", "a person", "dop_ctx_vec_")


class _FakeNetwork:
    def __init__(self):
        self.multiplier = 1.0
        self.history = []

    def set_multiplier(self, value):
        self.multiplier = float(value)
        self.history.append(float(value))


class _FakeAccelerator:
    @staticmethod
    def unwrap_model(model):
        return model


class _FakeOutput:
    def __init__(self, pred):
        self.pred = pred


class _FakeTrainer:
    def __init__(self, network):
        self.network = network
        self.seen_embeddings = []

    def call_dit(self, args, accelerator, transformer, latents, batch, noise, noisy, timesteps, dtype):
        self.seen_embeddings.append(batch["ctx_vec"].clone())
        # The frozen teacher predicts 2; the enabled LoRA predicts 3.
        value = 2.0 + self.network.multiplier
        return _FakeOutput(torch.full_like(noisy, value))


def test_add_dop_loss_uses_class_embedding_and_restores_adapter_multiplier():
    network = _FakeNetwork()
    trainer = _FakeTrainer(network)
    args = Namespace(dop_loss_weight=0.5, dop_trigger_word="Alice", dop_class_word="woman")
    class_embed = torch.tensor([[9.0]])
    batch = {
        "ctx_vec": torch.tensor([[1.0]]),
        "dop_ctx_vec": class_embed,
        "dop_signature": dop_signature("Alice", "woman").unsqueeze(0),
    }
    normal = torch.tensor(4.0, requires_grad=True)
    total, metrics = add_dop_loss(
        trainer,
        args,
        _FakeAccelerator(),
        object(),
        network,
        batch,
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.zeros(1),
        torch.float32,
        normal,
        {},
        embedding_key="ctx_vec",
        dop_embedding_key="dop_ctx_vec",
    )
    assert total.item() == pytest.approx(4.5)
    assert metrics["loss/dop"].item() == pytest.approx(1.0)
    assert metrics["loss/dop_weighted"].item() == pytest.approx(0.5)
    assert all(torch.equal(value, class_embed) for value in trainer.seen_embeddings)
    assert network.multiplier == 1.0
    assert network.history == [0.0, 1.0]


@pytest.mark.parametrize(
    "backend,settings,script",
    [
        (krea2, {"training_mode": "Krea 2"}, "krea2_cache_text_encoder_outputs.py"),
        (flux2, {"flux2_model_version": "Klein Base 4B ★"}, "flux_2_cache_text_encoder_outputs.py"),
    ],
)
def test_supported_backends_forward_dop_to_training_and_cache(backend, settings, script):
    common = {
        **settings,
        "dop_enabled": True,
        "dop_loss_weight": "1.0",
        "dop_trigger_word": "Alice",
        "dop_class_word": "woman",
        "dataset_config": "dataset.toml",
        "vae_model": "vae.safetensors",
        "recache_text": True,
        "recache_latents": False,
        "output_dir": ".",
        "output_name": "test",
        "network_type": "LoRA",
        "network_dim_low": "8",
        "network_alpha_low": "8",
    }
    if backend is krea2:
        common.update(krea2_dit_model="raw.safetensors", krea2_text_encoder="text.safetensors")
    else:
        common.update(flux2_dit_model="klein.safetensors", flux2_text_encoder="text")
    train = backend.build_commands(common)[0]
    cache = backend.build_cache_commands(common, "python")[0]
    assert "--dop_loss_weight" in train
    assert train[train.index("--num_processes") + 1] == "1"
    assert "--dop_trigger_word" in train
    assert "--dop_class_word" in train
    assert any(str(part).endswith(script) for part in cache)
    assert "--dop_trigger_word" in cache
    assert "--dop_class_word" in cache


@pytest.mark.parametrize(
    "backend,settings",
    [
        (krea2, {"training_mode": "Krea 2", "krea2_text_encoder": "text.safetensors"}),
        (flux2, {"flux2_model_version": "Klein Base 4B ★", "flux2_text_encoder": "text"}),
    ],
)
def test_staged_dop_cache_reuse_requests_validated_skip_existing(backend, settings):
    common = {
        **settings,
        "dop_enabled": True,
        "dop_cache_reuse": True,
        "dop_trigger_word": "Alice",
        "dop_class_word": "a woman",
        "dataset_config": "dataset.toml",
        "recache_text": True,
    }
    cache = backend.build_cache_commands(common, "python")[0]
    assert "--skip_existing" in cache


def test_flux2_dev_does_not_forward_dop():
    settings = {
        "flux2_model_version": "Dev",
        "dop_enabled": True,
        "dop_loss_weight": "1.0",
        "dop_trigger_word": "Alice",
        "dop_class_word": "woman",
        "dataset_config": "dataset.toml",
        "vae_model": "vae.safetensors",
        "flux2_dit_model": "dev.safetensors",
        "flux2_text_encoder": "text",
        "recache_text": True,
        "output_dir": ".",
        "output_name": "test",
        "network_type": "LoRA",
        "network_dim_low": "8",
        "network_alpha_low": "8",
    }
    assert "--dop_loss_weight" not in flux2.build_commands(settings)[0]
    assert "--dop_trigger_word" not in flux2.build_cache_commands(settings, "python")[0]


def test_flux2_klein_reference_guided_command_is_explicit_and_disabled_by_default():
    settings = {
        "flux2_model_version": "Klein Base 4B ★",
        "dataset_config": "dataset.toml", "vae_model": "vae.safetensors",
        "flux2_dit_model": "klein.safetensors", "flux2_text_encoder": "text",
        "output_dir": ".", "output_name": "test", "network_type": "LoRA",
        "network_dim_low": "8", "network_alpha_low": "8",
    }
    assert "--flux2_reference_guided" not in flux2.build_commands(settings)[0]

    settings.update(
        flux2_reference_guided=True,
        flux2_reference_conditions="Other pictures of subject (image JSONL)",
        flux2_reference_sigma_max="0.7",
        flux2_reference_direct_loss_weight="0.25",
    )
    command = flux2.build_commands(settings)[0]
    assert command[command.index("--flux2_reference_conditions") + 1] == "subject_ref"
    assert command[command.index("--flux2_reference_sigma_max") + 1] == "0.7"
    assert command[command.index("--flux2_reference_direct_loss_weight") + 1] == "0.25"


def test_flux2_reference_guided_parser_rejects_dev():
    parser = flux2_setup_parser(setup_parser_common())
    args = parser.parse_args([
        "--model_version", "dev", "--flux2_reference_guided",
    ])
    with pytest.raises(ValueError, match="Klein"):
        Flux2NetworkTrainer().handle_model_specific_args(args)


def test_flux2_same_item_teacher_hides_reference_from_student(monkeypatch):
    trainer = Flux2NetworkTrainer()
    latents = torch.zeros(1, 4, 2, 2)
    calls = []

    class Network:
        enabled = True
        def set_enabled(self, enabled):
            self.enabled = enabled

    network = Network()
    monkeypatch.setattr(
        trainer, "get_noisy_model_input_and_timesteps",
        lambda *args, **kwargs: (torch.zeros_like(latents), torch.tensor([500.0])),
    )
    def fake_call(_args, _accelerator, _transformer, _latents, batch, *_rest, **_kwargs):
        calls.append((network.enabled, sorted(key for key in batch if key.startswith("latents_control_"))))
        value = 2.0 if not network.enabled else 0.0
        return DiTOutput(pred=torch.full_like(latents, value, requires_grad=network.enabled), target=torch.ones_like(latents))
    monkeypatch.setattr(trainer, "call_dit", fake_call)
    monkeypatch.setattr(
        trainer, "compute_loss",
        lambda _args, output, *_rest, **_kwargs: (torch.nn.functional.mse_loss(output.pred, output.target), {}),
    )
    args = Namespace(
        flux2_reference_guided=True, flux2_reference_conditions="same_item",
        flux2_reference_sigma_max=0.75, flux2_reference_direct_loss_weight=0.0,
        dop_loss_weight=0.0,
    )
    accelerator = SimpleNamespace(
        device=torch.device("cpu"), unwrap_model=lambda model: model,
    )

    loss, metrics = trainer.process_batch(
        args, accelerator, object(), network,
        {"timesteps": torch.tensor([0.5]), "ctx_vec": torch.zeros(1, 1, 1)},
        latents, torch.zeros_like(latents), object(), torch.float32, torch.float32, None, 0,
    )

    assert calls == [(False, ["latents_control_0"]), (True, [])]
    assert network.enabled is True
    assert loss.item() == pytest.approx(4.0)
    assert metrics["teacher/conditioned"].item() == 1.0


def test_staged_dop_can_inherit_override_and_disable():
    base = {
        "dop_enabled": True,
        "dop_loss_weight": "1.0",
        "dop_trigger_word": "Alice",
        "dop_class_word": "woman",
    }
    inherited = MusubiTunerGUI._apply_stage_dop_settings(dict(base), {"dop_mode": "inherit"})
    assert inherited == base

    overridden = MusubiTunerGUI._apply_stage_dop_settings(
        dict(base),
        {"dop_mode": "enable", "dop_loss_weight": "0.5", "dop_class_word": "a person"},
    )
    assert overridden["dop_enabled"] is True
    assert overridden["dop_loss_weight"] == "0.5"
    assert overridden["dop_class_word"] == "a person"

    disabled = MusubiTunerGUI._apply_stage_dop_settings(dict(base), {"dop_mode": "disable"})
    assert disabled["dop_enabled"] is False


def test_staged_depth_helper_memory_can_inherit_or_override():
    base = {"krea2_keep_depth_helpers_on_gpu": False}
    assert MusubiTunerGUI._apply_stage_depth_memory_settings(dict(base), {"depth_helpers_mode": "inherit"}) == base
    kept = MusubiTunerGUI._apply_stage_depth_memory_settings(dict(base), {"depth_helpers_mode": "keep on GPU"})
    assert kept["krea2_keep_depth_helpers_on_gpu"] is True
    offloaded = MusubiTunerGUI._apply_stage_depth_memory_settings(
        {"krea2_keep_depth_helpers_on_gpu": True}, {"depth_helpers_mode": "offload to CPU"}
    )
    assert offloaded["krea2_keep_depth_helpers_on_gpu"] is False

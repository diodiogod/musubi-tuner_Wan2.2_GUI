import pytest
import torch

from musubi_tuner.convert_lora import convert_from_diffusers, convert_to_diffusers


@pytest.mark.parametrize("module", [
    "blocks.0.attn.qkv_proj", "blocks.35.attn.out_proj",
    "blocks.0.adaln_proj.linear", "blocks.0.mlp.fc1",
    "blocks.0.self_attn.q", "double_blocks.0.img_attn_proj",
])
def test_conversion_preserves_module_names_and_effective_weights(module):
    name = "lora_unet_" + module.replace(".", "_")
    down = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    up = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    source = {
        name + ".lora_down.weight": down,
        name + ".lora_up.weight": up,
        name + ".alpha": torch.tensor(8.0),
    }
    converted = convert_to_diffusers("lora_unet_", None, source)
    prefix = "diffusion_model." + module
    assert set(converted) == {prefix + ".lora_A.weight", prefix + ".lora_B.weight"}
    restored = convert_from_diffusers("lora_unet_", converted)
    assert set(restored) == set(source)
    torch.testing.assert_close(
        restored[name + ".lora_up.weight"] @ restored[name + ".lora_down.weight"],
        (up @ down) * 4,
    )

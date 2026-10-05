# Checkpoints

Trained TokenDial sliders. Each file holds **only** the per-layer `rgb_token`
(~180 KB) and is loaded on top of the base Wan2.1-T2V-1.3B model via `--ckpt`.

Shipped demo sliders:

| File | Type | Direction |
|------|------|-----------|
| `dog_fluffy.safetensors` | appearance | make a dog's coat fluffier (see the README inference example) |
| `person_older.safetensors` | appearance | make a person look older (reverse at negative scale) |
| `cat_more_kitten.safetensors` | appearance | make a cat look more kitten-like |
| `person_east_asian.safetensors` | appearance | shift a person toward East Asian features (used in the README V2V example) |
| `motion.safetensors` | motion | speed up / slow down motion, bidirectional (see the README motion example) |

Train your own with `tokendial/train_appearance.sh` / `tokendial/train_motion.sh`;
the resulting `runs/**/epoch-*.safetensors` can be moved here and used directly.

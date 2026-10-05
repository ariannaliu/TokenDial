import os

VisionEncoders = dict()

# Get the directory containing this file
_config_dir = os.path.dirname(os.path.abspath(__file__))

TextEncoders = dict()
TextEncoders["bert"] = dict(
    name="bert_base",
    pretrained="bert-base-uncased",
    config=os.path.join(_config_dir, "config_bert.json"),
    d_model=768,
    fusion_layer=9,
)
TextEncoders["bert_large"] = dict(
    name="bert_large",
    pretrained="bert-large-uncased",
    config=os.path.join(_config_dir, "config_bert_large.json"),
    d_model=1024,
    fusion_layer=19,
)
TextEncoders["med_bert"] = dict(
    name="med_bert_base",
    pretrained="bert-base-uncased",
    config=os.path.join(_config_dir, "med_config.json"),
    d_model=768,
)

TextEncoders["med_bert_large"] = dict(
    name="med_bert_large",
    pretrained="bert-base-uncased", # not a bug, it just follows BLIP.
    config=os.path.join(_config_dir, "med_large_config.json"),
    d_model=768
)
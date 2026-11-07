"""Exact Qwen3.5 template lengths without repeatedly materializing image pixels.

The pinned native template inserts one vision placeholder per image, then
repeats its token according to the processor grid. This helper uses those same
native text tokens and real processor grids. Final rows still undergo ordinary
full processor encoding and label checks; this is only the trimming search.
"""
from copy import deepcopy
from pathlib import Path

from ..sft.common import ConversionError
from .artifacts import fingerprint
from .encoding import normalize, values


class QwenTokenLength:
    def __init__(self, backend, image_root):
        if getattr(backend.arguments, "model_type", None) != "qwen3_5":
            raise ConversionError("unsupported_length_backend", "This token expansion requires the pinned Qwen3.5 template")
        self.backend = backend
        self.root = Path(image_root)
        self.image_id = backend.template.image_token_id
        self.path_counts = {}
        self.geometry_counts = {}

    def image_tokens(self, relative):
        if relative in self.path_counts:
            return self.path_counts[relative]
        from PIL import Image
        from qwen_vl_utils import fetch_image

        path = self.root / relative
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ConversionError("image_path_escape", "Length measurement image leaves the package")
        with Image.open(path) as image:
            geometry = (*image.size, image.mode)
        if geometry not in self.geometry_counts:
            processor = self.backend.processor.image_processor
            image = fetch_image({"image": str(path)}, image_patch_size=processor.patch_size)
            encoded = processor(images=[image], return_tensors="pt", do_resize=False)
            grid = encoded["image_grid_thw"][0]
            patches = int(grid.prod().item())
            merge = processor.merge_size ** 2
            if patches < 1 or patches % merge:
                raise ConversionError("invalid_image_grid", "Processor returned an invalid vision-token grid")
            self.geometry_counts[geometry] = patches // merge
            del encoded
            image.close()
        self.path_counts[relative] = self.geometry_counts[geometry]
        return self.path_counts[relative]

    def _text_ids(self, record):
        row = deepcopy(record)
        row["images"] = []
        for message in row["messages"]:
            if any(marker in message["content"] for marker in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")):
                raise ConversionError("reserved_vision_text", "Unexpected literal native vision tokens in source text")
            message["content"] = message["content"].replace(
                "<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        encoded = self.backend.template.encode(normalize(row, self.backend.preprocessor), return_length=True)
        ids = values(encoded["input_ids"])
        if ids.count(self.image_id) != len(record["images"]):
            raise ConversionError("vision_placeholder_mismatch", "Native text encoding changed the number of vision placeholders")
        return ids

    def __call__(self, record):
        ids = self._text_ids(record)
        return len(ids) + sum(self.image_tokens(p) - 1 for p in record["images"])

    def input_fingerprint(self, record):
        counts = iter(self.image_tokens(p) for p in record["images"])
        expanded = []
        for token in self._text_ids(record):
            if token == self.image_id:
                expanded.extend([token] * next(counts))
            else:
                expanded.append(token)
        assert next(counts, None) is None
        return fingerprint(expanded)

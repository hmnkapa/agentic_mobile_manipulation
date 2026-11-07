"""Namespaced ms-swift extension making the EOS follow message loss flags.

ms-swift 4.6.0 applies a message's loss flag to RESPONSE but not SUFFIX.
Reclassifying only a fully masked final response's suffix as OTHER makes
native default loss handling respect that flag without changing any tokens.
"""
NATIVE_LOSS_SCALE = "default+manipulation_message_mask"


def explicitly_masked(value):
    return value is False or isinstance(value, list) and bool(value) and all(explicitly_masked(v) for v in value)


def suffix_context_types(context_types, messages, suffix, other):
    if messages and explicitly_masked(messages[-1].get("loss")):
        return [other if kind == suffix else kind for kind in context_types]
    return list(context_types)


def register_loss():
    from swift.loss_scale import LossScale, loss_scale_map
    from swift.template import ContextType

    class MessageMaskLossScale(LossScale):
        def __call__(self, context_list, context_types, messages, **kwargs):
            if self.base_strategy != "default":
                raise ValueError("Manipulation message masking requires default base loss")
            types = suffix_context_types(context_types, messages, ContextType.SUFFIX, ContextType.OTHER)
            return super().__call__(context_list, types, messages, **kwargs)

    loss_scale_map["manipulation_message_mask"] = MessageMaskLossScale

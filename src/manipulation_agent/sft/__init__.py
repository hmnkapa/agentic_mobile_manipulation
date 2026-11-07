"""CPU-only trajectory conversion and shared student protocol helpers."""

from .protocol import build_student_context, from_student_arguments, to_student_arguments
from .history import build_sample_messages, compress_history
from .history_compaction import COMPACT_HISTORY, compact_history, compact_record

VERSION = "manipulation_sft_v1.3.0"

__all__ = ["VERSION", "build_student_context", "compress_history", "build_sample_messages",
           "from_student_arguments", "to_student_arguments", "COMPACT_HISTORY", "compact_history", "compact_record"]

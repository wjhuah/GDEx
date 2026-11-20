import numpy as np
from .inclusion import COLUMNS_INCLUSION

COMPACT = ['bound_group_len', 'mean_log_word_frequency', 'pronoun_ratio', 'n_subordinate_clauses', 'mean_dep_distance', 'context_content_count', 'filled_argument_roles', 'sense_cosine', 'semantic_cache_missing', 'sense_cosine_margin', 'sense_overlap', 'target_pos_match_ratio', 'frame_probability', 'collocation_log_mean', 'target_position_ratio']

def compact_vector(candidate, entry, full, frequencies):
    values = dict(zip(COLUMNS_INCLUSION, full))
    values["bound_group_len"] = len(candidate["sent"].split())
    values["semantic_cache_missing"] = float(not np.isfinite(values["sense_cosine"]) or not np.isfinite(values["sense_cosine_margin"]))
    return [values[name] for name in COMPACT]

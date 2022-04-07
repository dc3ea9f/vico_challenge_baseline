from bisect import bisect_left
import torch


class Memo:
    def __init__(self, iterable=None):
        self.data = []
        self.count_all = 0
        if iterable is not None:
            self.extend(iterable)

    def add(self, value):
        if value not in self.data:
            self.data.append(value)
            self.count_all += 1

    def extend(self, iterable):
        for value in iterable:
            self.add(value)
    
    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        return Memo(self.data).extend(other)

    def __sub__(self, other):
        return Memo([e for e in self.data if e not in other])

    def __and__(self, other):
        return Memo([e for e in self.data if e in other])

    __or__ = __and__

    def __xor__(self, other):
        return (self - other) | (self + other)

    def __str__(self):
        return "Memo({})".format(', '.join(str(e) for e in self.data))

    __repr__ = __str__


def zip_dot_keys(keys, level=2):
    keys = ['.'.join(key.split('.')[:level + 1]) for key in keys]
    keys = Memo(keys)
    return keys


def limit_key_nums(keys, target_size=64):
    if len(keys) < target_size:
        return keys
    key_prefixs_in_level = [zip_dot_keys(keys, level) for level in [0, 1, 2]]
    lengths_in_level = [len(e) for e in key_prefixs_in_level]
    sample_idx = bisect_left(lengths_in_level, target_size) - 1
    if sample_idx < 0:
        sample_idx = 0
    return key_prefixs_in_level[sample_idx]


def align_state_dict(state_dict, ref_state_dict):
    result = {}
    key_match_memo = Memo()
    key_not_found_memo = Memo()
    key_mismatch_memo = Memo()
    for ref_k, ref_v in ref_state_dict.items():
        if ref_k not in state_dict:
            key_not_found_memo.add(ref_k)
            result[ref_k] = ref_v
        else:
            v = state_dict[ref_k]
            if v.shape != ref_v.shape:
                key_mismatch_memo.add(ref_k)
                result[ref_k] = ref_v
            else:
                key_match_memo.add(ref_k)
                result[ref_k] = v
    key_useless_memo = Memo(state_dict.keys()) - Memo(ref_state_dict.keys())

    key_match_memo_limited = limit_key_nums(key_match_memo)
    key_not_found_memo_limited = limit_key_nums(key_match_memo)
    key_mismatch_memo_limited = limit_key_nums(key_mismatch_memo)
    key_useless_memo_limited = limit_key_nums(key_useless_memo)

    if key_match_memo.count_all == 0:
        print(f"[ERROR] load 0 keys")
    else:
        print(f"[SUCCESS] load {key_match_memo.count_all} keys")
    if key_not_found_memo.count_all > 0:
        print(f"[WARNING] {key_not_found_memo.count_all} not found in new state dict, use original")
        for key in key_not_found_memo_limited:
            print(f"\t'{key}' not found")
    if key_mismatch_memo.count_all > 0:
        print(f"[WARNING] {key_mismatch_memo.count_all} mismatch in new state dict, use original")
        for key in key_mismatch_memo_limited:
            print(f"\t'{key}' mismatch")
    if key_useless_memo.count_all > 0:
        print(f"[WARNING] {key_useless_memo.count_all} useless in current state dict, dropped")
        for key in key_useless_memo_limited:
            print(f"\t'{key}' dropped")
    return result

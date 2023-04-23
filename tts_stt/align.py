import argparse
import re
from itertools import product
from collections import deque

# Source: https://johnlekberg.com/blog/2020-10-25-seq-align.html

def read_format(data: dict):
    asr_tokens, slot_tokens, slot_labels = [], [], []
    for key, entry in data.items():
        asr_data = entry['utt'].lower().strip(' .').split()
        slot_data = entry['annot_utt'].lower()
        slot_data = slot_data.strip()
        slot_data = re.split('\[|\]', slot_data)

        tokens, labels = [], []
        for slot in slot_data:
            if slot==' ' or slot=='':
                continue
            if ':' not in slot:
                phrase = slot.strip().split()
                label = 'O'
                for token in phrase:
                    tokens.append(token)
                    labels.append(label)
            else:
                label, phrase = slot.strip().split(':')
                label = label.strip()
                phrase = phrase.strip().split()
                for token in phrase:
                    tokens.append(token)
                    labels.append(label)
        
        asr_tokens.append(asr_data)
        slot_tokens.append(tokens)
        slot_labels.append(labels)

    return (asr_tokens, slot_tokens, slot_labels)

def needleman_wunsch(x, y):
    """Run the Needleman-Wunsch algorithm on two sequences.

    x, y -- sequences.

    Code based on pseudocode in Section 3 of:

    Naveed, Tahir; Siddiqui, Imitaz Saeed; Ahmed, Shaftab.
    "Parallel Needleman-Wunsch Algorithm for Grid." n.d.
    https://upload.wikimedia.org/wikipedia/en/c/c4/ParallelNeedlemanAlgorithm.pdf
    """
    N, M = len(x), len(y)
    def s(a, b): return int(a == b)

    DIAG = -1, -1
    LEFT = -1, 0
    UP = 0, -1

    # Create tables F and Ptr
    F = {}
    Ptr = {}

    F[-1, -1] = 0
    for i in range(N):
        F[i, -1] = -i
    for j in range(M):
        F[-1, j] = -j

    option_Ptr = DIAG, LEFT, UP
    for i, j in product(range(N), range(M)):
        option_F = (
            F[i - 1, j - 1] + s(x[i], y[j]),
            F[i - 1, j] - 1,
            F[i, j - 1] - 1,
        )
        F[i, j], Ptr[i, j] = max(zip(option_F, option_Ptr))

    # Work backwards from (N - 1, M - 1) to (0, 0)
    # to find the best alignment.
    alignment = deque()
    i, j = N - 1, M - 1
    while i >= 0 and j >= 0:
        direction = Ptr[i, j]
        if direction == DIAG:
            element = i, j
        elif direction == LEFT:
            element = i, None
        elif direction == UP:
            element = None, j
        alignment.appendleft(element)
        di, dj = direction
        i, j = i + di, j + dj
    while i >= 0:
        alignment.appendleft((i, None))
        i -= 1
    while j >= 0:
        alignment.appendleft((None, j))
        j -= 1

    return list(alignment)

def aligner(data: dict, multiple = False):
    (asr_slot_tokens, slot_tokens, slot_labels) = read_format(data)
    aligned_slot_tokens, aligned_slot_labels = [], []
    for i in range(len(asr_slot_tokens)):
        dataset_entry_tokens = slot_tokens[i]
        dataset_entry_labels = slot_labels[i]
        asr_entry_tokens = asr_slot_tokens[i]

        alignment = needleman_wunsch(dataset_entry_tokens, asr_entry_tokens)
        aligned_tokens, aligned_labels = [], []
        for value in alignment:
            # Insertion
            if value[0] is None:
                aligned_tokens.append(asr_entry_tokens[value[1]])
                aligned_labels.append('O')
            # Deletion
            elif value[1] is None:
                pass
            # Neither
            else:
                aligned_tokens.append(asr_entry_tokens[value[1]])
                aligned_labels.append(dataset_entry_labels[value[0]])

        assert len(aligned_tokens) == len(aligned_labels)
        aligned_slot_tokens.append(aligned_tokens)
        aligned_slot_labels.append(aligned_labels)

    if multiple==True:
        return aligned_slot_tokens, aligned_slot_labels
    return aligned_slot_tokens[0], aligned_slot_labels[0]

if __name__ == '__main__':
    eg = {
        "0": {
            "id": "0",
            "locale": "en-US",
            "partition": "test",
            "scenario": 16,
            "intent": 48,
            "utt": " Wait me up at 5 am this week.",
            "annot_utt": "wake me up at [time : five am] of maybe [date : this week]",
            "worker_id": "1",
            "slot_method": {
                "slot": [],
                "method": []
            },
            "judgments": {
                "worker_id": [],
                "intent_score": [],
                "slots_score": [],
                "grammar_score": [],
                "spelling_score": [],
                "language_identification": []
            }
        },
        "10": {
            "id": "50",
            "locale": "en-US",
            "partition": "test",
            "scenario": 5,
            "intent": 38,
            "utt": " All tell me the time in a ''M'' mean plus five.",
            "annot_utt": "olly tell me the time in [time_zone : g. m. t. plus five]",
            "worker_id": "1",
            "slot_method": {
                "slot": [],
                "method": []
            },
            "judgments": {
                "worker_id": [],
                "intent_score": [],
                "slots_score": [],
                "grammar_score": [],
                "spelling_score": [],
                "language_identification": []
            }
        }
    }
    aligned_slot_tokens, aligned_slot_labels = aligner(eg, multiple=True)
    print(aligned_slot_tokens, end='\n\n')
    print(aligned_slot_labels, end='\n\n')

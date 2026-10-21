"""Reuse the existing OpenPI tokenizer artifact; verify unchanged token IDs."""
import hashlib
import json
from pathlib import Path
import sentencepiece as spm
from transformers import GemmaTokenizerFast
root=Path(__file__).resolve().parent
from settings import TOKENIZER_MODEL
source=TOKENIZER_MODEL
assert source.exists()
tokenizer=GemmaTokenizerFast(vocab_file=str(source),add_bos_token=True,add_eos_token=False)
reference=spm.SentencePieceProcessor(model_file=str(source))
for text in ['pick up the black bowl and place it on the plate','Task: put the milk in the basket, State: 1 2 3;\nAction:','hello\nworld','pick\tup the bowl']:
    actual=tokenizer.encode(text,add_special_tokens=False)
    expected=reference.encode(text)
    assert actual==expected,(text,actual,expected)
for idx in range(reference.vocab_size()):
    token = reference.id_to_piece(idx)
    # HF's official Gemma converter represents the tab byte as a literal tab.
    if token == '<0x09>':
        assert tokenizer.convert_ids_to_tokens(idx) == '\t'
    else:
        assert tokenizer.convert_tokens_to_ids(token) == idx
target=root/'tokenizer'
tokenizer.save_pretrained(str(target))
(target/'artifact.json').write_text(json.dumps(dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),vocab_size=reference.vocab_size(),verified_token_ids=True),indent=2)+'\n')
print('Existing tokenizer verified and saved locally',reference.vocab_size())

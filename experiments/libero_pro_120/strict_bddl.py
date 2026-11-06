"""Fail closed on invalid tasks; permit only redundant trailing closing parens.

Applied within full-evaluation processes, never to the shared installation.
No instruction, object, placement or success predicate is changed.
"""
from pathlib import Path
import hashlib

def install():
 from liberopro.liberopro.envs import bddl_utils as b
 if getattr(b, '_full_eval_strict', False): return
 original_scan=b.scan_tokens
 def scan(filename=None,string=None):
  text=Path(filename).read_text() if filename is not None else string
  try:return original_scan(string=text)
  except Exception:
   # A complete root followed solely by redundant closing parentheses.
   depth=0; end=None
   for i,c in enumerate(text):
    if c=='(':depth+=1
    elif c==')':
     depth-=1
     if depth==0:end=i+1;break
   if end is None or not text[end:].strip() or set(text[end:].strip()) != {')'}:raise
   repaired=text[:end]
   tokens=original_scan(string=repaired)
   print('BDDL syntax repair: redundant trailing close only',filename,
         hashlib.sha256(text.encode()).hexdigest(),flush=True)
   return tokens
 b.scan_tokens=scan
 # Disable the installed fallback, which can silently select a different task.
 b.robosuite_parse_problem=b._core_robosuite_parse_problem
 b._full_eval_strict=True

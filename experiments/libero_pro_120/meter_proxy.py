"""Meter every local Qwen call, including tool-internal detection calls."""
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import threading
import time
import traceback
import requests

import pilot_common as c
c.configure()

LOCK=threading.Lock()
CONTEXT='setup'
STATS={}
BACKEND='http://127.0.0.1:22909'


def blank():
    return dict(requests=0,completed=0,errors=0,prompt_tokens=0,completion_tokens=0,
        missing_usage=0,seconds=0.,model_json_failures=0,finish_length=0)


def parseable(text):
    # Use the exact upstream extractor; arrays are valid for bounding boxes.
    import sys
    sys.path.insert(0,str(c.ROOT/'runtime'))
    from soma_vlm import _extract_json
    return _extract_json(text or '') is not None


def seed_for(case,ordinal):
    # arm-specific output paths have the same final case label.
    material=(case.rsplit('/',1)[-1]+'|'+str(ordinal)).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4],'big')%(2**31-1)


class Handler(BaseHTTPRequestHandler):
    def respond(self,code,body):
        raw=json.dumps(body).encode() if isinstance(body,dict) else body
        self.send_response(code);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)

    def do_GET(self):
        if self.path=='/metrics':
            with LOCK:r=dict(case=CONTEXT,**STATS.setdefault(CONTEXT,blank()))
            self.respond(200,r)
        elif self.path=='/v1/models':
            try:
                r=requests.get(BACKEND+'/v1/models',timeout=10)
                self.respond(r.status_code,r.content)
            except Exception as e:self.respond(503,dict(error=str(e)))
        else:self.respond(404,dict(error='Unknown local route'))

    def do_POST(self):
        global CONTEXT
        payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path=='/context':
            with LOCK:
                if STATS.get(CONTEXT,{}).get('requests',0)!=STATS.get(CONTEXT,{}).get('completed',0):
                    self.respond(409,dict(error='A Qwen request is still active'));return
                CONTEXT=payload['case']
                if CONTEXT in STATS:
                    self.respond(409,dict(error='Context already used; do not silently rerun'));return
                STATS[CONTEXT]=blank()
            self.respond(200,dict(case=CONTEXT));return
        if self.path!='/v1/chat/completions':self.respond(404,dict(error='Unknown local route'));return
        if payload.get('stream'):self.respond(400,dict(error='Streaming is not part of this protocol'));return
        with LOCK:
            case=CONTEXT;s=STATS.setdefault(case,blank());s['requests']+=1;ordinal=s['requests']
        payload.setdefault('seed',seed_for(case,ordinal))
        start=time.monotonic();record=dict(case=case,ordinal=ordinal,started_unix=time.time(),seed=payload['seed'])
        code=502;body=b'{}';failed=False
        try:
            r=requests.post(BACKEND+self.path,json=payload,timeout=1200)
            code=r.status_code;body=r.content
            data=r.json();failed=not r.ok
            if r.ok and data.get('model','').lower()!='qwen3-vl-32b-instruct':
                failed=True;code=502;body=json.dumps(dict(error='Wrong model identity')).encode()
            usage=data.get('usage');record['usage']=usage
            choice=(data.get('choices') or [{}])[0]
            content=(choice.get('message') or {}).get('content','')
            record.update(finish_reason=choice.get('finish_reason'),json_parseable=parseable(content) if r.ok else None)
            with LOCK:
                s=STATS[case]
                if usage:
                    s['prompt_tokens']+=usage.get('prompt_tokens',0)
                    s['completion_tokens']+=usage.get('completion_tokens',0)
                else:s['missing_usage']+=1
                s['model_json_failures']+=int(r.ok and not record['json_parseable'])
                s['finish_length']+=int(choice.get('finish_reason')=='length')
        except Exception as e:
            failed=True;record['error']=str(e);body=json.dumps(dict(error=str(e))).encode()
        finally:
            record.update(status_code=code,error=record.get('error'),seconds=time.monotonic()-start)
            with LOCK:
                s=STATS[case];s['completed']+=1;s['errors']+=int(failed);s['seconds']+=record['seconds']
                with (c.ROOT/'vlm-usage.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n')
        self.respond(code,body)

    def log_message(self,*args):pass


if __name__=='__main__':
    ThreadingHTTPServer(('127.0.0.1',22910),Handler).serve_forever()

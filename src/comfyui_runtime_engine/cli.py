from __future__ import annotations
import argparse,json,sys
from .config import RuntimeConfig
from .runtime import RuntimeEngine
from .errors import RuntimeEngineError

def parser():
    p=argparse.ArgumentParser(prog="comfy-runtime"); p.add_argument("--config",default="runtime-engine.toml")
    c=p.add_subparsers(dest="command",required=True)
    c.add_parser("doctor"); c.add_parser("gpu-probe")
    m=c.add_parser("models"); ms=m.add_subparsers(dest="models_command",required=True)
    for x in ("scan","plan","verify"): ms.add_parser(x)
    w=c.add_parser("warmup"); ws=w.add_subparsers(dest="warmup_command",required=True)
    ws.add_parser("verify"); run=ws.add_parser("probe"); run.add_argument("--hold-seconds",type=float,default=5)
    g=c.add_parser("gpu-server-probe"); g.add_argument("--hold-seconds",type=float,default=2)
    rep=c.add_parser("gpu-server-probe-repeat"); rep.add_argument("--iterations",type=int,default=3); rep.add_argument("--hold-seconds",type=float,default=2)
    return p

def main(argv=None):
    a=parser().parse_args(argv)
    try:
        e=RuntimeEngine(RuntimeConfig.from_toml(a.config))
        if a.command=="doctor": r=e.doctor()
        elif a.command=="gpu-probe": r=e.gpu_probe(True)
        elif a.command=="models":
            r={"scan":e.residency_scan,"plan":e.residency_plan,"verify":e.residency_verify}[a.models_command]()
        elif a.command=="warmup":
            r=e.warmup_verify() if a.warmup_command=="verify" else e.warmup_probe(a.hold_seconds)
        elif a.command=="gpu-server-probe": r=e.probe_gpu_server(a.hold_seconds)
        else: r=e.repeated_gpu_server_probe(a.iterations,a.hold_seconds)
        print(json.dumps(r,indent=2,ensure_ascii=False))
        return 0 if r.get("success",True) else 1
    except (RuntimeEngineError,OSError,ValueError,KeyError) as exc:
        print(f"comfy-runtime error: {exc}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())

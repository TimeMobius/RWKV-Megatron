"""Start a shared CPU teacher-request pool; vLLM remains an external service."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rwkv_megatron.arguments import add_rwkv_arguments
from rwkv_megatron.online_teacher import build_teacher_client
from rwkv_megatron.teacher_pool import TeacherPool, make_pool_server


def main():
    parser = add_rwkv_arguments(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    client = build_teacher_client(args, use_pool=False)
    pool = TeacherPool(client, workers=args.rwkv_teacher_pool_workers,
                       queue_size=args.rwkv_teacher_pool_queue_size,
                       max_results=args.rwkv_teacher_pool_max_results,
                       result_ttl=args.rwkv_teacher_pool_result_ttl,
                       min_workers=min(args.rwkv_teacher_prefetch_min_workers, args.rwkv_teacher_pool_workers),
                       latency_target=args.rwkv_teacher_prefetch_latency_target)
    try:
        server = make_pool_server(pool, args.rwkv_teacher_pool_host, args.rwkv_teacher_pool_port,
                                  args.rwkv_teacher_pool_authkey)
        print(json.dumps(dict(event="teacher_pool_ready", address=server.address,
                              metadata=pool.get_metadata()), ensure_ascii=True), flush=True)
        server.serve_forever()
    finally:
        pool.close()


if __name__ == "__main__":
    main()

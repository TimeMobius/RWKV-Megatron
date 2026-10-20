"""Diagnostic pool backed by recorded real responses; upstream HTTP is forbidden.

Uses the response files from frozen_teacher_resume.py. Missing input fails,
and never substitutes another sample's distribution.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rwkv_megatron.arguments import add_rwkv_arguments
from rwkv_megatron.online_teacher import OnlineTeacherClient, TeacherTopK, build_teacher_client
from rwkv_megatron.teacher_pool import TeacherPool, make_pool_server


class ReplayTeacher(OnlineTeacherClient):
    def get_logits(self, input_ids, attention_mask=None, prompts=None):
        tokens = input_ids.detach().cpu().long().contiguous()
        mask = (torch.ones_like(tokens, dtype=torch.bool) if attention_mask is None
                else attention_mask.detach().cpu().bool().contiguous())
        signature = dict(shape=list(tokens.shape), prompts=prompts, top_k=self.top_k,
                         seed=self.request_seed, model=self.model_name, temperature=self.request_temperature)
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode())
        digest.update(tokens.numpy().tobytes())
        digest.update(mask.numpy().tobytes())
        path = self.folder / (digest.hexdigest() + '.pt')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert torch.equal(saved['tokens'], tokens) and torch.equal(saved['mask'], mask)
        self._thread_local.last_request_metrics = dict(latency_seconds=0.0, retries=0, rate_limits=0)
        return TeacherTopK(saved['token_ids'].clone(), saved['logprobs'].clone())

    def _get_session(self):
        raise AssertionError('Upstream HTTP is forbidden in replay pool')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-dir')
    args = add_rwkv_arguments(parser).parse_args()
    if not args.teacher_dir:
        parser.error('--teacher-dir is required')
    with patch('rwkv_megatron.online_teacher.OnlineTeacherClient', ReplayTeacher):
        client = build_teacher_client(args, use_pool=False)
    client.folder = Path(args.teacher_dir)
    pool = TeacherPool(client, workers=args.rwkv_teacher_pool_workers,
                       queue_size=args.rwkv_teacher_pool_queue_size,
                       max_results=args.rwkv_teacher_pool_max_results)
    try:
        server = make_pool_server(pool, args.rwkv_teacher_pool_host, args.rwkv_teacher_pool_port,
                                  args.rwkv_teacher_pool_authkey)
        print(json.dumps(dict(event='replay_pool_ready', address=server.address,
                              teacher_dir=str(client.folder))), flush=True)
        server.serve_forever()
    finally:
        pool.close()


if __name__ == '__main__':
    main()

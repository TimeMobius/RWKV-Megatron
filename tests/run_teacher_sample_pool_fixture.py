"""Separate teacher-pool process for the distributed replacement audit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from rwkv_megatron.teacher_pool import TeacherPool, make_pool_server
from teacher_sample_pool_fixture import FixturePoolTeacher

torch.set_num_threads(1)
pool = TeacherPool(FixturePoolTeacher(), workers=4, max_results=64)
server = make_pool_server(pool, port=0)
Path(sys.argv[1]).write_text(str(server.address[1]))
try:
    server.serve_forever()
finally:
    pool.close()

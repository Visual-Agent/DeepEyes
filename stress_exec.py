
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

from verl.workers.agent.envs.collab_code.code import code_exec
from tqdm import tqdm

code = r"""import numpy as np
print(np.ones((int(input()),)).sum())"""
input = "256"

MAX_N_JOBS = 1024
MAX_CONCURRENCY = cpu_count()
with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
    futures = []
    for _ in range(MAX_N_JOBS):
        futures.append(executor.submit(code_exec, code, input))

    for future in tqdm(as_completed(futures), total=len(futures)):
        succ, output = future.result()
        assert succ, output

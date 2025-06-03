from pathlib import Path

PROTOGCN_DIR = Path(__file__).parent.parent

ls_path_cfg = [
    str(PROTOGCN_DIR / "configs/fs26" / "v219--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2.py"),
    str(PROTOGCN_DIR / "configs/fs26" / "v220--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2--b.py"),
    str(PROTOGCN_DIR / "configs/fs26" / "v221--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2--jm.py"),
    str(PROTOGCN_DIR / "configs/fs26" / "v222--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2--bm.py"),
    # str(PROTOGCN_DIR / "configs/fs26" / "test.py"),
]
num_models = 10
ls_devices = [2, 3, 4, 5, 2, 3, 4, 5, 4, 5]

import subprocess
import multiprocessing as mp
import concurrent.futures

import time

for path_cfg in ls_path_cfg:
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                subprocess.run,
                args=f"CUDA_VISIBLE_DEVICES={device} bash tools/dist_train.sh {path_cfg} 1 --validate --id_model {id_model}",
                cwd=str(PROTOGCN_DIR),
                shell=True,
                check=True,
                text=True,
            )
            for id_model, device in zip(range(num_models), ls_devices)
        ]
        ls_trained_model = [f.result() for f in futures]

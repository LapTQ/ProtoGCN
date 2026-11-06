import subprocess
import concurrent.futures

ls_path_cfg = [
    "configs/fs26/v255.10_classes.2s-15frames.j.py",
    "configs/fs26/v255.10_classes.2s-15frames.b.py",
    "configs/fs26/v255.10_classes.2s-15frames.jm.py",
    "configs/fs26/v255.10_classes.2s-15frames.bm.py",
]
num_models = 10
ls_devices = [0, 0, 1, 1]

with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    for id_model in range(num_models):
        futures = []
        for path_cfg, device in zip(ls_path_cfg, ls_devices):
            futures.append(
                executor.submit(
                    subprocess.run,
                    args=f"CUDA_VISIBLE_DEVICES={device} bash tools/dist_train.sh {path_cfg} 1 --validate --id_model {id_model}",
                    shell=True,
                    check=True,
                    text=True,
                )
            )
        ls_trained_model = [f.result() for f in futures]

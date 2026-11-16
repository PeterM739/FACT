# Running the experiment on FRIDA

[FRIDA](https://docs.rdc.si/FRIDA/slurm/) is the Slurm GPU cluster of UL FRI. It only runs containers (Enroot +
Pyxis): it has no modules, conda must not be installed, and the login node is only for light work (git, editing,
`sbatch`, `srun`). All compute runs as Slurm jobs. Here everything runs inside one container image: NVIDIA's PyTorch
image plus `requirements.txt`, built once and saved as `~/containers/fact.sqfs`. The repo, the Hugging Face cache
and the results live in `$HOME` (`/shared/home/$USER`), which is persistent but has **no backups**.

| partition | use here | GPUs |
|---|---|---|
| `frida` | **the experiment** (trial and full run), up to 7 days | B200 180 GB (default), B300 288 GB, H100 80 GB, ... |
| `dev` | setup only: building the image, downloading models; the only partition for interactive jobs, up to 12 h | A100, L4 |

The main experiment ([`configs/llms_bias_bench.yaml`](../configs/llms_bias_bench.yaml)) runs 4 models unquantized
in one job on one B200: 255 cases x 14 conditions = 3,570 generations per model, roughly 4-6 h in total.

## 0. Connect VS Code to FRIDA

Once, with `tsh` already set up:

1. In VS Code, install the **Remote - SSH** extension.
2. In PowerShell: `tsh --proxy=rdc.si --user=<user> login` (add `--mfa-mode=otp --auth=local` if you have no
   passwordless login). Repeat this whenever the certificate expires.
3. Run `tsh config` and copy the paths from its output into this entry at the top of `%USERPROFILE%\.ssh\config`
   (on Windows `ProxyCommand` has no quotes and `tsh.exe` must be on PATH):

   ```
   Host login-frida
       HostName login-frida.rdc.si
       User <user>
       Port 3022
       UserKnownHostsFile "C:\Users\<you>\.tsh\known_hosts"
       IdentityFile "C:\Users\<you>\.tsh\keys\rdc.si\<user>"
       CertificateFile "C:\Users\<you>\.tsh\keys\rdc.si\<user>-ssh\rdc.si-cert.pub"
       ProxyCommand tsh proxy ssh --cluster=rdc.si --proxy=rdc.si:443 %r@%h:%p
       ServerAliveInterval 300
       ServerAliveCountMax 2
       TCPKeepAlive yes
   ```

4. In VS Code settings, turn off **Remote.SSH: Use Local Server**.
5. `F1` → **Remote-SSH: Connect to Host...** → `login-frida` → Linux. The terminal (`` Ctrl+` ``) is now on
   `<user>@login-frida`. Use it for git, editing, `sbatch` and `srun`; never run Python or pytest directly there.

## 1. Clone the repo

```bash
git clone https://github.com/PeterM739/FACT.git ~/FACT
```

Then *File → Open Folder → ~/FACT*. Later updates: `cd ~/FACT && git pull`.

## 2. Build the container image (once, 15-30 min)

```bash
mkdir -p ~/containers && cd ~/FACT
srun -p dev --gres=gpu:1 -c 8 --mem=32G -t 1:00:00 \
  --container-image=nvcr.io#nvidia/pytorch:26.09-py3 \
  --container-mounts="$PWD:$PWD" --container-workdir="$PWD" \
  --container-save="$HOME/containers/fact.sqfs" \
  bash -c 'export PIP_CONSTRAINT= && pip install --ignore-installed pyyaml==6.0.3 && pip install -r requirements.txt && python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"'
```

The last line should print the torch and CUDA versions and a GPU name. If the job fails, `fact.sqfs` is still saved
but incomplete: delete it (`rm ~/containers/fact.sqfs`), fix the cause and re-run.

## 3. Packages and PyTorch

- **PyTorch comes from NVIDIA's image**, built for the cluster's CUDA and GPUs, Blackwell included (26.09: torch 2.14,
  CUDA 13, Python 3.12). Never `pip install torch`: it would replace that build with a generic one. This is why
  `requirements.txt` has no torch.
- **All other packages** are installed once by pip and frozen into `fact.sqfs` by `--container-save`.
  `PIP_CONSTRAINT=` turns off NGC's built-in version pins, which would otherwise block ours. Every job starts from a
  fresh copy of the image, so anything installed inside a job is lost when it ends.
- **PyYAML is installed first with `--ignore-installed`** because the image ships a Debian (apt) PyYAML that pip
  cannot uninstall (`uninstall-no-record-file`). The flag is applied to PyYAML only: on the whole
  `requirements.txt` it would make pip reinstall torch from PyPI as a dependency of transformers and accelerate.
  Keep its version in step 2 the same as in `requirements.txt`.
- **To change a package:** edit `requirements.txt`, push, `git pull`, and re-run step 2. Or add one package on top of
  the existing image, saving to a new file, and point `slurm/frida.sbatch` at it:

  ```bash
  srun -p dev -t 0:30:00 --container-image="$HOME/containers/fact.sqfs" \
    --container-save="$HOME/containers/fact2.sqfs" bash -c 'PIP_CONSTRAINT= pip install X==Y'
  ```

  `--container-save` must be a file path, never a folder.
- **To check versions:** `srun -p dev --container-image="$HOME/containers/fact.sqfs" pip list | grep -Ei "torch|transformers"`.
  Every run's `manifest.yaml` also records the Python, torch, CUDA, GPU and package versions.
- If the image tag does not pull or the job reports a too-old driver, use the previous month's tag (e.g. `26.08-py3`).
- Do not make a venv on the login node: its Python (3.10) is not the container's, and it would run outside Slurm.

## 4. Hugging Face login and model download (once)

First accept the licences on the model pages of
[medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it),
[gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) and
[Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) (Meta's approval can take a few
hours; GaMS is not gated), then create a *read* token under *Settings → Access Tokens*.

Download in an interactive job, inside `tmux` so a VS Code disconnect does not stop it. Interactive jobs run only on
`dev`, and `dev` refuses jobs without a GPU, so the job asks for the smallest one (L4) even though it does not use it:

```bash
tmux new -s setup
srun -p dev --gres=gpu:L4:1 -c 8 --mem=16G -t 4:00:00 --container-image="$HOME/containers/fact.sqfs" --container-mounts="$HOME:$HOME" --pty env HOME="$HOME" bash
ls ~/FACT                                        # must list the repo; if not, stop: downloads would be lost
hf auth login                                    # paste the token; saved in ~/.cache/huggingface/token
for m in google/medgemma-27b-text-it google/gemma-3-27b-it cjvt/GaMS3-12B-Instruct meta-llama/Llama-3.3-70B-Instruct; do
  hf download "$m" --exclude "original/*"; done    # skips Llama's 140 GB duplicate original/ folder
hf download mamuto11/LLMs_Bias_Bench Data_Pub.xlsx --repo-type dataset
exit
```

About 275 GB. If no L4 is free, use `--gres=gpu:1` (any GPU). After `tmux detach` (`Ctrl+b d`),
`tmux attach -t setup` returns to it.

## 5. Small trial run

3 cases x 14 conditions x 4 models = 168 generations, on the same B200 as the full run:

```bash
cd ~/FACT
sed -e 's/^name: .*/name: trial/' -e 's/limit: 255/limit: 3/' configs/llms_bias_bench.yaml > configs/trial.yaml
sbatch --time=1:00:00 slurm/frida.sbatch configs/trial.yaml
squeue --me                                       # PD = waiting, R = running
tail -f results/fact_<jobid>.out                  # Ctrl+C stops watching, not the job
python3 -c "import json,collections,sys; c=collections.Counter((r['model'],r['status']) for r in map(json.loads,open(sys.argv[1]))); [print(*k,v) for k,v in sorted(c.items())]" results/trial_*/generations.jsonl
```

- Every model should be mostly `VALID`, and `results/trial_*/alpha_*/report.pdf` should open.
- If a model is mostly invalid, read its `raw_response` in `generations.jsonl` before the full run.
- The rate in the log (generations per second) x 3,570 is that model's time in the full run.

## 6. Full run

```bash
sbatch slurm/frida.sbatch
squeue --me ; tail -f results/fact_<jobid>.out
```

- One job runs all models, so the run has one Benjamini–Hochberg family and one report.
- The job keeps running when VS Code disconnects.
- `frida` shows free GPUs, `scontrol show job <jobid>` the details, and `scancel <jobid>` cancels the job.
- If no B200 is free, use another GPU: `sbatch --gres=gpu:B300:1 slurm/frida.sbatch`, or `--gres=gpu:H100:2` (the
  70B model is then split over the two GPUs).

## 7. Get the results

- The last line of the log names the folder, e.g. `results/llms_bias_bench_<timestamp>/alpha_0.01_pdf-png_300dpi/report.pdf`.
- In the VS Code Explorer, right-click the folder → **Download...** and save it into your local `FACT\results\`.
  Or, from PowerShell: `scp -r login-frida:FACT/results/llms_bias_bench_<timestamp> results\`.
- The folder is self-contained. Locally, `python -m src.main report --results results/<folder>` rebuilds the report,
  and `analyze` rebuilds the statistics and figures, without a model or GPU.

## 8. Clean up

Only `$SCRATCH` and each job's copy of the container are removed automatically; nothing in `$HOME` is. First check
that the downloaded results open locally (FRIDA has no backups). Then, with no job left in `squeue --me`:

```bash
rm -rf ~/FACT ~/containers ~/.cache/huggingface ~/.cache/enroot ~/.local/share/enroot ~/.cache/pip
du -sh ~ ; frida                                  # storage use should be close to zero
```

Also revoke the Hugging Face token on huggingface.co. An account with no jobs for 6 months is disabled; its data is
archived and deleted 3 months later.

## Troubleshooting

| problem | fix |
|---|---|
| `GENERATION_ERROR` with CUDA out of memory | lower that model's `batch_size` in the config |
| Hugging Face 403 / gated repo | accept the licence on the model page; check `~/.cache/huggingface/token` |
| VS Code or ssh cannot connect | `tsh login` again (the certificate expired) |
| job stays `PD` for a long time | try another GPU type (B300, H100) or a shorter `--time` |
| `$'\r': command not found` | the script got Windows line endings: use the copy from `git clone`, not one copied from Windows |

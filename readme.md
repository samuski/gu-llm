# Readme

## Set up

1. Install `Docker Desktop`, make sure to have it on.

2. `git clone https://github.com/samuski/gu-llm.git`

3. Edit `.env sample`, add the personal access token in the empty fields. Rename the file to `.env`

4. If the machine doesn't have GPU:

- In `docker-compose.yml` remove the line with `gpus: all`
- In `requirements.txt` remove `+cu121` in the line `torch==2.3.1+cu121`

5. Additional folders and projects.

- Multiwoz data needs to be downloaded separately. In terminal, run `git clone https://github.com/budzianowski/multiwoz.git` in root of the project.
- `adapter` folder in root, that has LoRA parameters.

6. Build the container by running `docker compose up -d --build` in terminal.

- It's going to take a few minutes as the model gets automatically downloaded into volume.
- Note that if the volume gets cleared, it will need to be downloaded again.

7. Once built, the dashboard can be accessed on browser via `localhost:8001`

## Data Clean Up

- In container exec, run `python core/llm/convert.py`.
- It converts the json data files into jsonl format and puts it in `data` directory in the root.
- The logic expects that the multiwoz project directory is in root of the project.

## Finetune

- In container exec, run `python core/llm/finetune.py`.
- It trains with given parameters in the file, default key parameters are `num_train_epochs=2`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=16`, `learning_rate=1.5e-4`.
- It's set to not save any intermediate steps. When finished all gets saved to `adapters` directory in the root.

## Clean Up

- In terminal, run `docker compose down -v` to turn off and remove volume.
- To remove residual objects, run `docker system prune`.
  - WARNING: This attempts to remove unused networks, stopped containers, dangling images, and build caches. This can remove objects from other projects!

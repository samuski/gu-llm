# Readme

## Set up

1. Install `Docker Desktop`, make sure to have it on.

2. `git clone https://github.com/samuski/gu-llm.git`

3. Edit `.env sample`, add the personal access token in the empty fields. Rename the file to `.env`

4. If the machine doesn't have GPU:

- In `docker-compose.yml` remove the line with `gpus: all`
- In `requirements.txt` remove `+cu121` in the line `torch==2.3.1+cu121`

5. Additional folders and projects.

- msmarco data needs to be downloaded separately. In terminal, `run hf download infosense/msmarco-reduced-trecrag --repo-type dataset --local-dir ./msmarco-reduced-trecrag`. Or in Docker desktop's exec.

6. Build the container by running `docker compose up -d --build` in terminal.

- It's going to take a few minutes as the model gets automatically downloaded into volume.
- Note that if the volume gets cleared, it will need to be downloaded again.

7. Go into the container with `docker compose exec backend bash` and run `docker compose exec backend bash` to migrate the db.

8. Import the data with `python manage.py import_segments --data-dir ./msmarco-reduced-trecrag`

9. Generate logs

- `python manage.py rag_base_run`
- `python manage.py rag_generate_retrieve`
- `python manage.py rag_retrieve_generate`
- `python manage.py rag_generate_retrieve_generate`

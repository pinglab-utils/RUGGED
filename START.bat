docker run --name k2bio_neo4j -d --rm --net jove_network --publish=7474:7474 --publish=7687:7687 --volume=%USERPROFILE%\workspace\JoVE_LLM\scratch:/data neo4j
docker run --name rugged -d --gpus all -it --rm -v %USERPROFILE%\workspace:/workspace -p 8888:8888 pinglabutils/rugged:v1 /bin/bash -c "jupyter notebook --ip=0.0.0.0 --port=8888 --notebook-dir=/workspace --allow-root --no-browser --NotebookApp.token='ghost'"
docker run -d --rm --gpus=all -v %USERPROFILE%\workspace\ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

docker network connect jove_network jove_llm

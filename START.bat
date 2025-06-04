docker run --name neo4j -d --rm --net rugged_network --publish=7474:7474 --publish=7687:7687 --volume=%USERPROFILE%\workspace\RUGGED\data\knowledge_graph:/data neo4j
docker run --name rugged -d --gpus all -it --rm --net rugged_network -v %USERPROFILE%\workspace:/workspace pinglabutils/rugged:v1 /bin/bash
docker run -d --rm --gpus=all --net rugged_network -v %USERPROFILE%\workspace\ollama:/root/.ollama -p 11434:11434 -e OLLAMA_HOST=0.0.0.0 --name ollama ollama/ollama

docker network connect rugged_network rugged

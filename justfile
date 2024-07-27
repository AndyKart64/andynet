default: run

run:
    docker-compose down; docker-compose up --build -d; docker logs -f andynet-agent-1

up:
    docker-compose up --build -d

down:
    docker-compose down

logs:
    docker logs -f andynet-agent-1

sh:
    docker exec -it andynet-agent-1 /bin/bash

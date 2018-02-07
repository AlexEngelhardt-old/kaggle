# Toxic comment classification challenge

- Start the Docker container with the Jupyter Data Science Notebook:
    sudo systemctl start docker.service
    make start # this runs docker-compose
	docker container ls  # write down the container-id for the next command
	docker container exec <container-id> jupyter notebook list  # get the access token

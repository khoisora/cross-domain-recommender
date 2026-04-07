IMAGE_NAME := crossrec
CONTAINER_NAME := crossrec-app
PORT := 8000

.PHONY: build run stop logs clean dev export

# Build Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run container (detached)
run:
	docker run -d --name $(CONTAINER_NAME) \
		-p $(PORT):8000 \
		-v $(PWD)/data:/app/data \
		$(IMAGE_NAME)
	@echo "CrossRec running at http://localhost:$(PORT)"

# Stop and remove container
stop:
	docker stop $(CONTAINER_NAME) 2>/dev/null || true
	docker rm $(CONTAINER_NAME) 2>/dev/null || true

# View logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Stop, rebuild, run
restart: stop build run

# Remove image and container
clean: stop
	docker rmi $(IMAGE_NAME) 2>/dev/null || true

# Run locally without Docker
dev:
	PYTHONPATH=. uvicorn backend.demo.main:app --host 0.0.0.0 --port $(PORT) --reload

# Export model artifacts (run before first build)
export:
	PYTHONPATH=. python ml/scripts/export_demo_artifacts.py

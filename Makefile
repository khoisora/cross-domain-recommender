IMAGE_NAME := crossrec
CONTAINER_NAME := crossrec-app
PORT := 8000

.PHONY: install export dev all \
        build run stop logs restart clean

# ============================================================
# Local (non-Docker) workflow
# ============================================================

# Create venv and install Python dependencies
install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

# Export model artifacts (run before first build or first dev run)
export:
	PYTHONPATH=. python ml/scripts/export_demo_artifacts.py

# Run locally without Docker
dev:
	PYTHONPATH=. uvicorn backend.demo.main:app --host 0.0.0.0 --port $(PORT) --reload

# Full local setup: install deps, export artifacts, then run dev server
all: install export dev

# ============================================================
# Docker workflow
# ============================================================

# Build Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run container (detached). SQLite DB lives inside the container.
# To persist ratings across restarts, add: -v $(PWD)/data:/app/data
run:
	docker run -d --name $(CONTAINER_NAME) \
		-p $(PORT):8000 \
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

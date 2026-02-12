# Installer les dépendances
install:
	uv sync

# Lancer l'API en local
run:
	uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# # Builder l'image Docker
# docker-build:
# 	docker build -t housing-api .

# # Lancer le container
# docker-run:
# 	docker run -p 8000:8000 housing-api

# Stop et supprime les containers
compose-down:
	docker compose down

# Build + lance en arrière-plan
compose-up:
	docker compose up -d --build

# Redémarrage complet
compose-restart: 
	compose-down compose-up


# Lancer les tests
test:
	uv run pytest

# Nettoyer le projet
clean:
	rm -rf .venv __pycache__ .pytest_cache

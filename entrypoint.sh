#!/bin/bash
set -e

echo "🚀 Étape 1 : Entraînement du modèle..."
python src/train.py

echo "✅ Entraînement terminé !"
echo "🚀 Étape 2 : Démarrage de l'API..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
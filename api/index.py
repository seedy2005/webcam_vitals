"""
index.py — Entrypoint for Vercel serverless deployment.
"""

try:
    from main import app
except ImportError:
    try:
        from .main import app
    except ImportError:
        from api.main import app

# Vercel serverless expects 'app'
__all__ = ["app"]

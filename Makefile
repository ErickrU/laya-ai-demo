.PHONY: setup demo route serve clean

setup:            ## create .venv and install laya (pinned in uv.lock)
	uv sync

demo:             ## triage the built-in sample tickets (first run downloads ~2 checkpoints)
	uv run python demo.py

route:            ## show which checkpoint each sample ticket would hit, no model load
	uv run python demo.py --route-only

serve:            ## optional: HTTP server on http://127.0.0.1:8000 (POST /v1/systemone)
	uv run --with 'laya[serve]==0.3.22' laya-serve

clean:
	rm -rf .venv __pycache__

import os
import subprocess
import sys

import uvicorn


def run_migrations():
    database_url = os.environ.get("DATABASE_URL", "")
    if database_url.startswith("sqlite") or not database_url:
        # SQLite is ephemeral on Render; create_all() in init_db() handles schema.
        print("SQLite detected — skipping alembic, schema created by init_db().", flush=True)
        return
    print("Running alembic upgrade head...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("Migration failed — aborting startup.", flush=True)
        sys.exit(result.returncode)
    print("Migrations complete.", flush=True)


if __name__ == "__main__":
    run_migrations()
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)


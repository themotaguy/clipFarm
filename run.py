#!/usr/bin/env python
"""Development entrypoint: `python run.py`."""

from __future__ import annotations

import config
from app import create_app

app = create_app()

if __name__ == "__main__":
    print(f"\n  clipFarm → http://{config.FLASK_HOST}:{config.FLASK_PORT}\n")
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        threaded=True,      # jobs run on worker threads and SSE holds connections open
        debug=False,        # the reloader would orphan in-flight pipeline threads
        use_reloader=False,
    )

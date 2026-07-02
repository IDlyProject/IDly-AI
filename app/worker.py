from __future__ import annotations

import time

from .db import Base, SessionLocal, engine
from .service import claim_next_job, run_job

POLL_SECONDS = 2


def run_worker() -> None:
    Base.metadata.create_all(bind=engine)

    while True:
        db = SessionLocal()
        try:
            job = claim_next_job(db)
            if not job:
                time.sleep(POLL_SECONDS)
                continue
            run_job(db, job)
        finally:
            db.close()


if __name__ == "__main__":
    run_worker()

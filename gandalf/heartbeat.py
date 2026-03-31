"""Gandalf heartbeat — periodically registers with an Automat server."""

import logging
import threading

import httpx

from gandalf.config import Settings

logger = logging.getLogger("gandalf.heartbeat")


def start_heartbeat(settings: Settings) -> threading.Event:
    """Start a daemon thread that sends heartbeats to the Automat server.

    Returns a :class:`threading.Event` that, when set, stops the heartbeat
    loop and lets the thread exit cleanly.

    Raises:
        ValueError: If *service_address* is not configured.
    """
    if not settings.service_address:
        raise ValueError(
            "GANDALF_SERVICE_ADDRESS must be set when GANDALF_AUTOMAT_HOST is "
            "configured. Set it to the reachable address of this Gandalf instance."
        )

    automat_url = f"{settings.automat_host}/heartbeat"
    payload = {
        "host": settings.service_address,
        "tag": settings.plater_title,
        "port": settings.web_port,
    }

    stop_event = threading.Event()

    def _loop() -> None:
        logger.info(
            "Heartbeat started — reporting to %s every %ds",
            settings.automat_host,
            settings.heart_rate,
        )
        while not stop_event.is_set():
            try:
                resp = httpx.post(automat_url, json=payload, timeout=10)
                logger.debug(
                    "Heartbeat to %s returned %d",
                    settings.automat_host,
                    resp.status_code,
                )
            except Exception as err:
                logger.error("Error contacting Automat server: %s", err)
            stop_event.wait(settings.heart_rate)

    thread = threading.Thread(target=_loop, daemon=True, name="gandalf-heartbeat")
    thread.start()
    return stop_event

import logging, json, time, os, threading
from logging import Logger
from typing import Optional, Dict, Tuple, Any

# --- Error classification helpers ---
def classify_error(err: Any) -> str:
    """Classify errors into categories: network, parse, data_integrity, other."""
    try:
        import socket
        from json import JSONDecodeError
        # Common imports that may not exist in all envs
        try:
            import requests
        except Exception:
            requests = None  # type: ignore
        try:
            import aiohttp
        except Exception:
            aiohttp = None  # type: ignore
    except Exception:
        socket = None  # type: ignore
        JSONDecodeError = ValueError  # type: ignore
        requests = None  # type: ignore
        aiohttp = None  # type: ignore

    # By type
    if isinstance(err, (ConnectionError, TimeoutError)):
        return 'network'
    if socket and isinstance(err, (socket.timeout, OSError)):
        return 'network'
    if requests and isinstance(err, getattr(requests, 'exceptions', ()).__dict__.get('RequestException', tuple())):
        return 'network'
    if aiohttp and isinstance(err, getattr(aiohttp, 'ClientError', tuple())):
        return 'network'
    if isinstance(err, (ValueError,)):
        return 'parse'
    if 'decode' in str(err).lower() or 'json' in str(err).lower():
        return 'parse'
    if isinstance(err, (KeyError, IndexError)):
        return 'data_integrity'
    if 'integrity' in str(err).lower() or 'mismatch' in str(err).lower():
        return 'data_integrity'
    return 'other'

class JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "ts": time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(record.created)),
            "level": record.levelname,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
        }
        # Include structured context if provided via LoggerAdapter extra
        for key in ("component", "game_id", "prediction_version", "category", "suppress", "escalate", "count"):
            if hasattr(record, key):
                base[key] = getattr(record, key)
        # Attach arbitrary context if present
        if hasattr(record, 'context') and isinstance(getattr(record, 'context'), dict):
            base['context'] = getattr(record, 'context')
        if record.exc_info:
            base['exception'] = self.formatException(record.exc_info)
        return json.dumps(base)

_loggers = {}

# Suppression / escalation state
_counters_lock = threading.Lock()
_counters: Dict[Tuple[str, str], Dict[str, Any]] = {}
_window_seconds = 60
_suppression_thresholds = {
    'network': 5,
    'parse': 3,
    'data_integrity': 2,
    'other': 10,
}
_escalation_thresholds = {
    'network': 20,
    'parse': 10,
    'data_integrity': 5,
    'other': 50,
}
_sample_every = 10  # When suppressed, log every Nth occurrence

def _bump_counter(component: str, category: str) -> Tuple[int, bool, bool]:
    """Increase counter and determine suppression/escalation.
    Returns: (count, suppress, escalate_now)
    """
    now = time.time()
    key = (component, category)
    with _counters_lock:
        state = _counters.get(key, {"start": now, "count": 0})
        # Reset window if expired
        if now - state.get("start", now) > _window_seconds:
            state = {"start": now, "count": 0}
        state["count"] += 1
        _counters[key] = state
        count = state["count"]
    sup_th = _suppression_thresholds.get(category, _suppression_thresholds['other'])
    esc_th = _escalation_thresholds.get(category, _escalation_thresholds['other'])
    suppress = count > sup_th and (count % _sample_every != 0)
    escalate_now = (count == esc_th)
    return count, suppress, escalate_now

class ContextAdapter(logging.LoggerAdapter):
    """LoggerAdapter injecting component and prediction_version, with suppression/escalation."""
    def __init__(self, logger: Logger, component: str, prediction_version: str):
        super().__init__(logger, {"component": component, "prediction_version": prediction_version})

    def process(self, msg, kwargs):
        extra = kwargs.get('extra', {})
        # Merge base context
        extra.setdefault('component', self.extra.get('component'))
        extra.setdefault('prediction_version', self.extra.get('prediction_version'))
        # Determine category for suppression
        category = extra.get('category')
        if not category and 'exc_info' in kwargs and kwargs['exc_info']:
            try:
                category = classify_error(kwargs['exc_info'][1])
            except Exception:
                category = 'other'
            extra['category'] = category
        # Apply suppression/escalation only to error/warning/info with category
        if category:
            count, suppress, escalate_now = _bump_counter(extra.get('component') or 'unknown', category)
            extra['count'] = count
            extra['suppress'] = suppress
            extra['escalate'] = escalate_now
            # If suppressed, mark but still pass through; UI may filter on 'suppress'
            # If escalate_now, we emit an extra alert below in log method wrappers
            kwargs['extra'] = extra
            # Attach back
            return msg, kwargs
        kwargs['extra'] = extra
        return msg, kwargs

    # Convenience methods to attach category and optional context
    def event(self, level: str, msg: str, *, category: Optional[str] = None, game_id: Optional[str] = None, context: Optional[dict] = None):
        extra = {}
        if category:
            extra['category'] = category
        if game_id:
            extra['game_id'] = game_id
        if context:
            extra['context'] = context
        log_fn = getattr(self, level.lower(), self.info)
        log_fn(msg, extra=extra)
        # Escalation: emit alert when threshold hit
        try:
            if extra.get('category'):
                key = (self.extra.get('component'), extra['category'])
                with _counters_lock:
                    st = _counters.get(key)
                esc_th = _escalation_thresholds.get(extra['category'], _escalation_thresholds['other'])
                if st and st.get('count') == esc_th:
                    self.error(f"Escalation threshold reached for category={extra['category']}", extra={"category": extra['category']})
        except Exception:
            pass

def get_logger(name: str, path: str, level: str = "INFO", structured: bool = True) -> Logger:
    if name in _loggers:
        return _loggers[name]
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.FileHandler(path, encoding='utf-8')
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    if structured:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    logger.addHandler(handler)
    _loggers[name] = logger
    return logger

def get_structured_adapter(component: str, prediction_version: str,
                           name: str = 'structured', path: str = os.path.join('logs', 'structured_events.log'),
                           level: str = 'INFO') -> ContextAdapter:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base = get_logger(name, path, level=level, structured=True)
    return ContextAdapter(base, component=component, prediction_version=prediction_version)

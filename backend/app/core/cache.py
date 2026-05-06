from cachetools import TTLCache
from threading import Lock
from .config import settings

_prediction_cache: TTLCache = TTLCache(
    maxsize=1000, ttl=settings.PREDICTION_CACHE_TTL
)
_interp_cache: TTLCache = TTLCache(
    maxsize=100, ttl=settings.INTERP_CACHE_TTL
)
_meta_cache: TTLCache = TTLCache(
    maxsize=50, ttl=settings.META_CACHE_TTL
)

_pred_lock = Lock()
_interp_lock = Lock()
_meta_lock = Lock()


def cache_key(*args) -> str:
    return ":".join(str(a) for a in args)


def get_prediction(key: str):
    with _pred_lock:
        return _prediction_cache.get(key)


def set_prediction(key: str, value):
    with _pred_lock:
        _prediction_cache[key] = value


def get_interp(key: str):
    with _interp_lock:
        return _interp_cache.get(key)


def set_interp(key: str, value):
    with _interp_lock:
        _interp_cache[key] = value


def get_meta(key: str):
    with _meta_lock:
        return _meta_cache.get(key)


def set_meta(key: str, value):
    with _meta_lock:
        _meta_cache[key] = value


def clear_prediction_cache():
    with _pred_lock:
        _prediction_cache.clear()
    with _interp_lock:
        _interp_cache.clear()

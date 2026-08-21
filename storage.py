"""Object storage for user uploads — Cloudflare R2, with a local-disk fallback.

Only USER UPLOADS live here. The demo corpus and the evaluation corpus stay on
committed local files and never touch this module.

Backend is chosen by environment, not by a flag: if all four R2_* variables are
set, uploads go to R2; otherwise they go to uploads/ on local disk so the app
still runs without any cloud config. Credentials are read from the environment
only and are never logged.

Keys are POSIX-style paths, e.g. "uploads/<id>/index/pages.json".
"""
import os, shutil, threading

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_ROOT = os.path.join(BASE_DIR, "uploads")

R2_VARS = ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")


def r2_configured():
    return all(os.getenv(v) for v in R2_VARS)


class LocalStorage:
    """Filesystem backend. Used when R2 is not configured."""

    name = "local"

    def __init__(self, root=LOCAL_ROOT):
        self.root = root

    def _path(self, key):
        path = os.path.realpath(os.path.join(self.root, key))
        root = os.path.realpath(self.root)
        if not (path == root or path.startswith(root + os.sep)):
            raise ValueError(f"key escapes the storage root: {key!r}")
        return path

    def put(self, key, data):
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def get(self, key):
        with open(self._path(key), "rb") as f:
            return f.read()

    def exists(self, key):
        return os.path.isfile(self._path(key))

    def list(self, prefix):
        base = self._path(prefix)
        if not os.path.isdir(base):
            return []
        # Relative to the RESOLVED root: _path() resolves symlinks, and on macOS
        # /var -> /private/var, so comparing against the raw root yields ../.. keys.
        root = os.path.realpath(self.root)
        out = []
        for dirpath, _, files in os.walk(base):
            for name in files:
                full = os.path.join(dirpath, name)
                out.append(os.path.relpath(full, root).replace(os.sep, "/"))
        return sorted(out)

    def delete_prefix(self, prefix):
        shutil.rmtree(self._path(prefix), ignore_errors=True)


class R2Storage:
    """Cloudflare R2 over the S3 API."""

    name = "r2"

    def __init__(self):
        import boto3
        from botocore.config import Config
        self.bucket = os.environ["R2_BUCKET"]
        self._client = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",                      # R2 ignores region but boto3 wants one
            config=Config(retries={"max_attempts": 3, "mode": "standard"},
                          signature_version="s3v4"),
        )

    def put(self, key, data):
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get(self, key):
        return self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def exists(self, key):
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def list(self, prefix):
        keys, token = [], None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = self._client.list_objects_v2(**kw)
            keys.extend(o["Key"] for o in resp.get("Contents", []))
            if not resp.get("IsTruncated"):
                return sorted(keys)
            token = resp.get("NextContinuationToken")

    def delete_prefix(self, prefix):
        keys = self.list(prefix)
        for i in range(0, len(keys), 1000):
            batch = [{"Key": k} for k in keys[i:i + 1000]]
            if batch:
                self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})


_backend = None
_lock = threading.Lock()


def backend():
    """The active storage backend, created once per process."""
    global _backend
    if _backend is None:
        with _lock:
            if _backend is None:
                _backend = R2Storage() if r2_configured() else LocalStorage()
    return _backend


def describe():
    b = backend()
    return {"backend": b.name,
            "bucket": getattr(b, "bucket", None),
            "r2_configured": r2_configured()}


# Convenience wrappers so callers never touch the backend object directly.
def put(key, data): return backend().put(key, data)
def get(key): return backend().get(key)
def exists(key): return backend().exists(key)
def list_keys(prefix): return backend().list(prefix)
def delete_prefix(prefix): return backend().delete_prefix(prefix)

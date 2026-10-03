"""Live task context, resolved AFTER the prompt-cache boundary.

A bare ``/skill <name>`` — no typed instruction — still has to know what the
task is. The providers here read that task context (today's diary entry, the
backlog head, the working tree's change count, the session's last user intent)
and render it into a ``[Live context]`` footer.

Two invariants make this safe to run on every skill load:

* **The static prefix is untouched.** Interpolation happens only in the footer,
  which the builder appends *after* ``append_user_instruction``. The bytes the
  cache planner breaks on are identical with and without live context, so a
  skill load cannot invalidate the prompt cache (see ``#81867``).
* **A provider can never break the load.** Every value is read through
  ``try``/``except``; a provider that raises omits the whole block (the model
  sees the plain skill, which is always coherent). Every value passes through
  ``elide`` against a per-provider cap, and the rendered block against
  ``LIVE_CONTEXT_MAX_CHARS`` — a provider that could dump 100 KB is a defect,
  not a feature. Measured precedent: a 176 KB BACKLOG.md once closed the
  model's whole context window.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections import deque
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from agent.compression_marker import elide
from hermes_cli._subprocess_compat import IS_WINDOWS, windows_hide_flags

# Closing marker. The footer is delimited on both sides so the scaffold
# extractors can remove it from an arbitrary span without knowing where the
# user text resumes — the gateway auto-load path appends the user's message
# AFTER the built payload, so a trailing-only strip would eat their words.
LIVE_CONTEXT_END = "[end live context]"

logger = logging.getLogger(__name__)

# Footer header. ``skill_commands`` cuts extracted user instructions on this
# exact marker, so the string is part of that module's scaffold contract.
LIVE_CONTEXT_HEADER = "[Live context — resolved at load time, not part of the skill]"

# Whole-block ceiling, measured against the ~1200-char budget for a skill load
# with two providers active. Past this the footer is noise, not context.
LIVE_CONTEXT_MAX_CHARS = 1200

# Providers re-read at most this often (seconds). A skill load is not a poll
# loop; the values only need to be fresh within a turn. Short TTL because
# HERMES_LAST_USER_INTENT is per-turn by nature.
_CACHE_TTL_S = 60.0

# Per-provider character caps, public so callers (and tests) can assert against the
# same numbers the resolver enforces. A diary file and a backlog are both unbounded
# in practice, so each read is bounded twice: by bytes scanned and by chars kept.
MAX_CHARS_PER_PROVIDER: Dict[str, int] = {
    "HERMES_DIARY_TODAY": 500,
    "HERMES_BACKLOG_HEAD": 400,
    "HERMES_GIT_STATUS": 300,
    "HERMES_LAST_USER_INTENT": 400,
}
_MAX_SOURCE_BYTES = 256_000
_DIARY_TAIL_LINES = 40
_DIARY_HEAD_LINES = 8
_BACKLOG_HEAD_LINES = 30
_BACKLOG_SCAN_LINES = 400
_GIT_TIMEOUT_S = 5
_GIT_SAMPLE_LINES = 5
# Rows scanned backwards for the newest user turn. Small: the newest row is
# almost always within a handful of messages of the tail.
_INTENT_SCAN_MESSAGES = 40

# name -> provider(ctx) -> str. The table is the extension point; there is no
# if-chain, so a new variable is one entry plus one test.
LIVE_CONTEXT_PROVIDERS: Dict[str, Callable[[Dict[str, Any]], str]] = {}

_CACHE: Dict[tuple, tuple] = {}


def _hermes_home() -> Optional[Path]:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        logger.debug("live context: hermes home unavailable", exc_info=True)
        return None


def _cwd() -> Optional[Path]:
    """The agent's working directory, or None when it cannot be resolved.

    ``resolve_agent_cwd`` propagates the OSError of a deleted cwd; a skill load
    must not inherit that.
    """
    try:
        from agent.runtime_cwd import resolve_agent_cwd

        return Path(resolve_agent_cwd())
    except Exception:
        logger.debug("live context: cwd unavailable", exc_info=True)
        return None


def _read_head(path: Path, keep: int, scan: int) -> list:
    """First *keep* non-blank lines, reading at most *scan* lines."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            kept = []
            for line in islice(handle, scan):
                stripped = line.strip()
                if stripped:
                    kept.append(stripped)
                    if len(kept) >= keep:
                        break
            return kept
    except OSError:
        return []


def _read_tail(path: Path, keep: int, max_bytes: int = _MAX_SOURCE_BYTES) -> list:
    """Last *keep* lines, seeking past the front of an oversized file."""
    try:
        size = path.stat().st_size
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
                handle.readline()  # drop the partial line the seek landed in
            return [line.rstrip("\n") for line in deque(handle, maxlen=keep)]
    except OSError:
        return []


def _diary_today(ctx: Dict[str, Any]) -> str:
    """Today's diary entry: its heading plus the tail. Never the whole file."""
    home = ctx.get("hermes_home")
    if home is None:
        return ""
    path = Path(home) / "diary" / f"{ctx['today']}.md"
    head = _read_head(path, _DIARY_HEAD_LINES, scan=_DIARY_HEAD_LINES * 4)
    tail = _read_tail(path, _DIARY_TAIL_LINES)
    if not head and not tail:
        return ""
    seen = set(head)
    body = [line for line in tail if line not in seen]
    return "\n".join(head + (["..."] if body else []) + body)


def _backlog_head(ctx: Dict[str, Any]) -> str:
    """The backlog's opening items, from the project root then the hermes home."""
    for base in (ctx.get("cwd"), ctx.get("hermes_home")):
        if base is None:
            continue
        lines = _read_head(Path(base) / "BACKLOG.md", _BACKLOG_HEAD_LINES, _BACKLOG_SCAN_LINES)
        if lines:
            return "\n".join(lines)
    return ""


def _git_status(ctx: Dict[str, Any]) -> str:
    """How many paths are dirty, plus a sample of their names. Never the diff."""
    cwd = ctx.get("cwd")
    if cwd is None:
        return ""
    kwargs = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {}
    try:
        from agent.delegation_context import delegated_child_subprocess_env

        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=_GIT_TIMEOUT_S,
            check=False, stdin=subprocess.DEVNULL,
            env=delegated_child_subprocess_env(), **kwargs,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Not a repo, git missing, or a repo too slow to scan in 5s (a large
        # btrfs worktree can exceed it). "No value here" is a normal outcome
        # for this provider, not a broken one — the other providers still ship.
        logger.debug("live context: git status unavailable: %s", exc)
        return ""
    lines = [line for line in (completed.stdout or "").splitlines() if line.strip()]
    if not lines:
        return ""
    sample = "\n".join(lines[:_GIT_SAMPLE_LINES])
    remaining = len(lines) - _GIT_SAMPLE_LINES
    return f"{len(lines)} uncommitted path(s):\n{sample}" + (f"\n... (+{remaining} more)" if remaining > 0 else "")


def _last_user_intent(ctx: Dict[str, Any]) -> str:
    """The newest real user turn of this session, truncated.

    Read from the transcript rather than an in-process variable: a skill load
    runs before the current turn's row exists, so the last persisted row IS the
    task the user is carrying into the skill. A stored skill scaffold is run
    through the extraction helper so the footer carries the typed request, not a
    second copy of a skill body.
    """
    session_id = ctx.get("session_id")
    if not session_id:
        return ""
    try:
        from hermes_state_registry import acquire, release_or_close

        db = acquire()
        try:
            rows = db.get_messages(session_id, latest=True, limit=_INTENT_SCAN_MESSAGES)
        finally:
            release_or_close(db)
    except Exception:
        logger.debug("live context: session transcript unavailable", exc_info=True)
        return ""
    for row in reversed(rows or []):
        if not isinstance(row, dict) or row.get("role") != "user":
            continue
        content = row.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            from agent.skill_commands import extract_user_instruction_from_skill_message

            content = extract_user_instruction_from_skill_message(content) or ""
        except Exception:
            logger.debug("live context: scaffold extraction unavailable", exc_info=True)
        content = " ".join(content.split())
        return content if content else ""
    return ""


LIVE_CONTEXT_PROVIDERS.update({
    "HERMES_DIARY_TODAY": _diary_today,
    "HERMES_BACKLOG_HEAD": _backlog_head,
    "HERMES_GIT_STATUS": _git_status,
    "HERMES_LAST_USER_INTENT": _last_user_intent,
})


def live_context_session_ctx(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Resolve the inputs every provider reads. Each is optional by design."""
    return {
        "session_id": (session_id or "").strip() or None,
        "cwd": _cwd(),
        "hermes_home": _hermes_home(),
        # Part of the cache key so a process alive across midnight re-reads
        # today's diary instead of serving yesterday's.
        "today": datetime.now().strftime("%Y-%m-%d"),
    }


def _cached(name: str, ctx: Dict[str, Any], provider: Callable[[Dict[str, Any]], str]) -> str:
    key = (name, str(ctx.get("cwd")), str(ctx.get("session_id")), ctx.get("today"))
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit is not None and now - hit[0] < _CACHE_TTL_S:
        return hit[1]
    value = provider(ctx)
    _CACHE[key] = (now, value)
    return value


def resolve_live_context(
    providers: Optional[Any] = None, session_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Resolve the named providers against *session_ctx*.

    Returns ``{}`` when the block is disabled, when nothing resolved, or when a
    provider raised — a partial footer could describe a *different* task than the
    one in flight, so a broken provider drops the block instead of half-filling
    it. Callers must be able to render a skill with no footer at all.
    """
    names = list(providers) if providers else list(LIVE_CONTEXT_PROVIDERS)
    ctx = session_ctx if isinstance(session_ctx, dict) else live_context_session_ctx()
    resolved: Dict[str, str] = {}
    for name in names:
        provider = LIVE_CONTEXT_PROVIDERS.get(name)
        if provider is None:
            logger.debug("live context: unknown provider %r", name)
            continue
        try:
            value = str(_cached(name, ctx, provider) or "").strip()
        except Exception:
            logger.warning("live context: provider %s failed; omitting block", name, exc_info=True)
            return {}
        if value:
            resolved[name] = elide(value, MAX_CHARS_PER_PROVIDER.get(name, 400))
    return resolved


def render_live_context_block(values: Dict[str, str]) -> str:
    """Render resolved values as the footer block; ``""`` when there are none."""
    if not values:
        return ""
    lines = [LIVE_CONTEXT_HEADER]
    budget = LIVE_CONTEXT_MAX_CHARS - len(LIVE_CONTEXT_HEADER) - len(LIVE_CONTEXT_END) - 2
    for name, value in values.items():
        line = f"{name}: {value}"
        if len(line) > budget:
            line = elide(line, max(budget, 0))
        if len(line) > budget:
            break  # out of budget: stop rather than emit an over-long footer
        lines.append(line)
        budget -= len(line) + 1
    lines.append(LIVE_CONTEXT_END)
    return "\n".join(lines)


def strip_live_context(text: str) -> str:
    """Remove every live-context block from *text*, closing marker included.

    The extractors in ``agent.skill_commands`` recover the user's typed words
    from a built skill message. Without this the footer would ride along into
    memory rows, session titles and previews as if the user had typed it.
    """
    if LIVE_CONTEXT_HEADER not in text:
        return text
    out, rest = [], text
    while True:
        start = rest.find(LIVE_CONTEXT_HEADER)
        if start < 0:
            out.append(rest)
            return "".join(out)
        end = rest.find(LIVE_CONTEXT_END, start)
        if end < 0:  # unterminated block: drop from the header on
            out.append(rest[:start])
            return "".join(out)
        out.append(rest[:start])
        rest = rest[end + len(LIVE_CONTEXT_END):]


def build_live_context_footer(session_ctx: Optional[Dict[str, Any]] = None) -> str:
    """The complete footer, or ``""`` when there is nothing to say."""
    return render_live_context_block(resolve_live_context(session_ctx=session_ctx))


def clear_live_context_cache() -> None:
    """Drop cached provider values (tests; a settings change that moves cwd)."""
    _CACHE.clear()
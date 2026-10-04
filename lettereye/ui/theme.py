"""Look & feel: brand colours from the app icon (cyan → indigo → magenta), cards, typography."""

from nicegui import ui

PRIMARY = "#6366f1"
SECONDARY = "#06b6d4"
ACCENT = "#d946ef"

STATUS = {
    "queued": ("Queued", "grey-6", "schedule"),
    "processing": ("Processing", "indigo-5", "autorenew"),
    "filed": ("Filed", "positive", "task_alt"),
    "review": ("Needs review", "warning", "rate_review"),
    "failed": ("Failed", "negative", "error_outline"),
    "ignored": ("Dismissed", "grey-5", "visibility_off"),
}

OCR_SOURCES = {
    "text_layer": ("PDF text", "text_snippet"),
    "fast_ocr": ("PP-OCRv6", "bolt"),
    "ocr_llm": ("OCR LLM", "auto_awesome"),
}

CSS = """
:root {
  --le-bg: #f5f6fb; --le-card: #ffffff; --le-border: rgba(15, 23, 42, .08); --le-muted: #64748b;
  --le-text: #0f172a; --le-soft: #eef0fb;
  --le-gradient: linear-gradient(135deg, #22d3ee 0%, #6366f1 52%, #d946ef 100%);
}
body.body--dark {
  --le-bg: #0d1017; --le-card: #151924; --le-border: rgba(148, 163, 184, .12); --le-muted: #94a3b8;
  --le-text: #e2e8f0; --le-soft: #1c2131;
}
body { background: var(--le-bg); color: var(--le-text); font-family: 'Inter', 'Segoe UI Variable', 'Segoe UI', system-ui, -apple-system, sans-serif; }
.nicegui-content { padding: 0; }
.le-page { width: 100%; max-width: 1280px; margin: 0 auto; padding: 28px 32px 48px; gap: 20px; }
.le-card { background: var(--le-card); border: 1px solid var(--le-border); border-radius: 18px;
  box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.12); }
.le-card.q-card { box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.12); }
.le-muted { color: var(--le-muted); }
.le-title { font-size: 26px; font-weight: 700; letter-spacing: -.02em; }
.le-subtitle { font-size: 14px; color: var(--le-muted); }
.le-section-title { font-size: 15px; font-weight: 650; letter-spacing: -.01em; }
.le-gradient-text { background: var(--le-gradient); -webkit-background-clip: text; background-clip: text; color: transparent; }
.le-hero { background: var(--le-gradient); color: white; border-radius: 22px; position: relative; overflow: hidden; }
.le-hero::after { content: ''; position: absolute; inset: 0; background: radial-gradient(circle at 85% 20%, rgba(255,255,255,.25), transparent 45%); pointer-events: none; }
.le-stat-value { font-size: 30px; font-weight: 750; letter-spacing: -.03em; line-height: 1.1; }
.le-chip { border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 600; display: inline-flex; align-items: center; gap: 4px; }
.le-drawer { background: linear-gradient(180deg, #0f1222 0%, #161a33 55%, #1d1640 100%) !important; color: #e2e8f0; }
.le-nav-item { border-radius: 12px; padding: 10px 14px; color: #cbd5e1; cursor: pointer; transition: background .15s, color .15s; gap: 12px; }
.le-nav-item:hover { background: rgba(255,255,255,.06); color: white; }
.le-nav-item.active { background: linear-gradient(90deg, rgba(99,102,241,.35), rgba(217,70,239,.18)); color: white; box-shadow: inset 0 0 0 1px rgba(255,255,255,.08); }
.le-nav-badge { margin-left: auto; background: #f59e0b; color: #1f1300; border-radius: 999px; font-size: 11px; font-weight: 700; padding: 1px 8px; }
.le-status-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.le-dot-on { background: #22c55e; box-shadow: 0 0 0 4px rgba(34,197,94,.18); animation: le-pulse 2s infinite; }
.le-dot-off { background: #64748b; }
.le-dot-warn { background: #f59e0b; }
@keyframes le-pulse { 0%,100% { box-shadow: 0 0 0 4px rgba(34,197,94,.18); } 50% { box-shadow: 0 0 0 7px rgba(34,197,94,.06); } }
.le-bar { height: 8px; border-radius: 999px; background: var(--le-soft); overflow: hidden; }
.le-bar > div { height: 100%; border-radius: 999px; background: var(--le-gradient); }
.le-feed-item { border-bottom: 1px solid var(--le-border); padding: 9px 2px; }
.le-feed-item:last-child { border-bottom: none; }
.le-worker-item { border-radius: 14px; padding: 10px 12px; cursor: pointer; border: 1px solid transparent; transition: background .15s; }
.le-worker-item:hover { background: var(--le-soft); }
.le-worker-item.active { background: var(--le-soft); border-color: rgba(99,102,241,.35); }
.le-preview { border-radius: 12px; border: 1px solid var(--le-border); background: white; }
.le-kbd { font-family: ui-monospace, 'Cascadia Code', Consolas, monospace; font-size: 12px; background: var(--le-soft); border-radius: 6px; padding: 1px 6px; }
.le-token { cursor: pointer; }
.le-empty { padding: 48px 16px; text-align: center; }
.q-avatar__content { font-family: inherit; letter-spacing: .02em; }
.q-field--outlined .q-field__control { border-radius: 12px; }
.q-btn { border-radius: 10px; text-transform: none; font-weight: 600; letter-spacing: 0; }
.q-tab { text-transform: none; font-weight: 600; }
.q-table__card { background: var(--le-card) !important; border-radius: 18px; box-shadow: none; border: 1px solid var(--le-border); }
.q-stepper { background: var(--le-card) !important; border-radius: 22px !important; box-shadow: none !important; border: 1px solid var(--le-border); }
.q-stepper__tab { padding: 18px 24px; }
.q-dialog__inner > .q-card { border-radius: 20px; }
.le-ocr { white-space: pre-wrap; font-family: ui-monospace, 'Cascadia Code', Consolas, monospace; font-size: 12.5px; line-height: 1.5; max-height: 420px; overflow: auto; background: var(--le-soft); border-radius: 12px; padding: 14px; margin: 0; }
"""


def apply(dark: bool | None) -> None:
    ui.colors(primary=PRIMARY, secondary=SECONDARY, accent=ACCENT, positive="#16a34a", negative="#dc2626",
              warning="#f59e0b", info="#0ea5e9")
    ui.add_css(CSS)
    ui.dark_mode(dark)

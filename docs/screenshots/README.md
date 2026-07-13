# Dashboard screenshots

Add PNG captures here for the README **Dashboard preview** section.

## Recommended captures

| File | What to show |
|------|----------------|
| `dashboard-overview.png` | KPI sub-bar + investment-theme charts (scroll to top of dashboard) |
| `dashboard-findings.png` | Claude findings panel with urgency badges |
| `dashboard-chat.png` | "Ask me…" chat with a grounded answer and sources |

## Capture tips

- Run the dashboard locally: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python3 src/api.py`
- Open http://127.0.0.1:7860 in Chrome or Safari
- Viewport width ~1400px; use full-window or cropped region screenshots
- Prefer synthetic or anonymized data if anything sensitive appears on screen
- Compress PNGs before commit (aim for &lt;500 KB each)

## README embed (after adding images)

```markdown
## Dashboard preview

![KPI bar and investment-theme charts](docs/screenshots/dashboard-overview.png)
*KPIs and quarterly charts from `analysis.py`.*

![Claude findings by theme](docs/screenshots/dashboard-findings.png)
*Cached briefing from `insights.py`.*

![RAG analyst chat](docs/screenshots/dashboard-chat.png)
*Grounded drill-down via `chat.py` + Chroma retrieval.*
```

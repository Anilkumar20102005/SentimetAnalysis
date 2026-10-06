import { useState, useEffect, useCallback, useRef } from 'react'
import './index.css'

const API_BASE = '/api'

// ── Circular confidence ring ────────────────────────────────────────────────
function ConfidenceRing({ value, sentiment }) {
  const r = 28
  const circ = 2 * Math.PI * r
  const offset = circ - (value / 100) * circ
  const color = sentiment === 'positive' ? '#10b981' : '#ef4444'
  return (
    <div className="confidence-ring-wrap">
      <svg viewBox="0 0 68 68" width="68" height="68">
        <circle cx="34" cy="34" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="6" />
        <circle cx="34" cy="34" r={r} fill="none" stroke={color} strokeWidth="6"
          strokeDasharray={circ} strokeDashoffset={offset}
          strokeLinecap="round" transform="rotate(-90 34 34)"
          style={{ transition: 'stroke-dashoffset 0.7s ease' }} />
      </svg>
      <div className="conf-text">
        <span style={{ color }}>{Math.round(value)}%</span>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.6rem' }}>conf.</span>
      </div>
    </div>
  )
}

// ── Result display ──────────────────────────────────────────────────────────
function ResultPanel({ result, loading, error }) {
  if (loading) {
    return (
      <div className="result-empty">
        <div className="spinner" style={{ width: 40, height: 40 }} />
        <p>Analyzing sentiment…</p>
      </div>
    )
  }
  if (error) {
    return (
      <div className="result-panel">
        <div className="error-banner">⚠️ {error}</div>
      </div>
    )
  }
  if (!result) {
    return (
      <div className="result-empty">
        <div className="result-empty-icon">🔍</div>
        <p>Enter a review and click Analyze to see the prediction</p>
      </div>
    )
  }

  const isPos = result.sentiment === 'positive'
  const conf  = result.confidence * 100

  return (
    <div className="result-panel result-animate">
      {/* Main badge */}
      <div className={`sentiment-badge ${result.sentiment}`}>
        <span className="sentiment-emoji">{isPos ? '😊' : '😤'}</span>
        <div className="sentiment-label">
          <h3>{isPos ? 'POSITIVE' : 'NEGATIVE'}</h3>
          <p>{isPos ? 'Customer had a good experience' : 'Customer had a poor experience'}</p>
        </div>
        <ConfidenceRing value={conf} sentiment={result.sentiment} />
      </div>

      {/* Probability bars */}
      <div className="prob-bars">
        <div className="prob-row">
          <div className="prob-header">
            <span className="prob-name" style={{ color: 'var(--positive)' }}>✅ Positive</span>
            <span className="prob-pct" style={{ color: 'var(--positive)' }}>
              {(result.positive_prob * 100).toFixed(1)}%
            </span>
          </div>
          <div className="prob-bar-track">
            <div className="prob-bar-fill pos" style={{ width: `${result.positive_prob * 100}%` }} />
          </div>
        </div>
        <div className="prob-row">
          <div className="prob-header">
            <span className="prob-name" style={{ color: 'var(--negative)' }}>❌ Negative</span>
            <span className="prob-pct" style={{ color: 'var(--negative)' }}>
              {(result.negative_prob * 100).toFixed(1)}%
            </span>
          </div>
          <div className="prob-bar-track">
            <div className="prob-bar-fill neg" style={{ width: `${result.negative_prob * 100}%` }} />
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="stats-row">
        <div className="stat-chip">
          <div className="stat-val">{conf.toFixed(1)}%</div>
          <div className="stat-label">Confidence</div>
        </div>
        <div className="stat-chip">
          <div className="stat-val">{result.inference_ms}ms</div>
          <div className="stat-label">Latency</div>
        </div>
        <div className="stat-chip">
          <div className="stat-val">{result.processed_text.split(' ').length}</div>
          <div className="stat-label">Tokens</div>
        </div>
      </div>
    </div>
  )
}

// ── History item ─────────────────────────────────────────────────────────────
function HistoryList({ history, onSelect }) {
  if (!history.length) return (
    <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center', padding: '1rem 0' }}>
      No history yet. Analyze a review to get started!
    </p>
  )
  return (
    <div className="history-list">
      {history.map((item, i) => (
        <div key={i} className="history-item" onClick={() => onSelect(item.text)}>
          <div className={`hist-dot ${item.sentiment}`} />
          <div className="hist-text">{item.text}</div>
          <div className={`hist-conf ${item.sentiment}`}>
            {(item.confidence * 100).toFixed(0)}%
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Model Info ────────────────────────────────────────────────────────────────
function ModelInfoCard({ info }) {
  if (!info) return (
    <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
      {info === null ? 'Could not connect to backend. Start the FastAPI server!' : 'Loading model info…'}
    </div>
  )
  return (
    <div className="model-info-grid">
      <div className="model-stat">
        <div className="ms-val">{info.test_accuracy ? `${(info.test_accuracy * 100).toFixed(1)}%` : '—'}</div>
        <div className="ms-label">Test Accuracy</div>
      </div>
      <div className="model-stat">
        <div className="ms-val">{info.test_roc_auc?.toFixed(3) ?? '—'}</div>
        <div className="ms-label">ROC-AUC</div>
      </div>
      <div className="model-stat">
        <div className="ms-val">{info.train_samples ? `${(info.train_samples / 1000).toFixed(0)}K` : '—'}</div>
        <div className="ms-label">Train Samples</div>
      </div>
      <div className="model-stat" style={{ gridColumn: '1 / -1' }}>
        <div className="ms-val" style={{ fontSize: '0.85rem', fontWeight: 600 }}>{info.model_type ?? '—'}</div>
        <div className="ms-label">Architecture</div>
      </div>
      <div className="model-stat" style={{ gridColumn: '1 / -1' }}>
        <div className="ms-val" style={{ fontSize: '0.8rem', fontWeight: 600 }}>
          {info.dataset ? info.dataset.split('(')[0].trim() : '—'}
        </div>
        <div className="ms-label">Dataset</div>
      </div>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [text, setText]         = useState('')
  const [title, setTitle]       = useState('')
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [history, setHistory]   = useState([])
  const [modelInfo, setModelInfo] = useState(undefined)
  const [backendUp, setBackendUp] = useState(true)
  const [examples, setExamples]   = useState([])
  const textareaRef = useRef(null)

  // Fetch model info + examples on mount
  useEffect(() => {
    fetch(`${API_BASE}/model/info`)
      .then(r => r.json())
      .then(d => { setModelInfo(d); setBackendUp(true) })
      .catch(() => { setModelInfo(null); setBackendUp(false) })

    fetch(`${API_BASE}/examples`)
      .then(r => r.json())
      .then(d => setExamples(d.examples || []))
      .catch(() => {})
  }, [])

  const analyze = useCallback(async () => {
    if (!text.trim() || loading) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const resp = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim(), title: title.trim() || undefined }),
      })
      if (!resp.ok) {
        const err = await resp.json()
        throw new Error(err.detail || `Server error ${resp.status}`)
      }
      const data = await resp.json()
      setResult(data)
      setHistory(prev => [{ text: text.trim(), ...data }, ...prev].slice(0, 15))
    } catch (e) {
      setError(e.message || 'An error occurred. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }, [text, title, loading])

  const handleKey = (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) analyze() }
  const clearAll  = () => { setText(''); setTitle(''); setResult(null); setError(null); textareaRef.current?.focus() }
  const loadExample = (ex) => { setText(ex.text); setTitle(ex.label); setResult(null); setError(null) }

  return (
    <>
      {/* Background orbs */}
      <div className="bg-orbs">
        <div className="orb orb-1" /><div className="orb orb-2" /><div className="orb orb-3" />
      </div>

      <div className="layout">
        {/* ── Header ── */}
        <header className="header">
          <div className="header-inner">
            <div className="logo">
              <span className="logo-icon">🔮</span>
              SentiScope
            </div>
            <div className="nav-badge">
              <div className={`dot ${backendUp ? '' : 'offline'}`} />
              {backendUp ? 'API Connected' : 'API Offline'}
            </div>
          </div>
        </header>

        {/* ── Main ── */}
        <main className="main animate-in">
          {/* Hero */}
          <div className="hero">
            <div className="hero-label">
              <span>🤖</span> AI-Powered Analysis
            </div>
            <h1>Product Sentiment<br />Analysis</h1>
            <p>
              Instantly classify Amazon product reviews as <strong>positive</strong> or <strong>negative</strong> using
              a model trained on 60,000 real reviews from the <em>CyranoB/polarity</em> dataset.
            </p>
          </div>

          {/* Main grid */}
          <div className="grid-2">
            {/* Input card */}
            <div className="card">
              <div className="card-title">✏️ Input Review</div>

              {/* Example buttons */}
              <div className="examples-strip">
                <span className="examples-label">Quick Examples</span>
                {examples.map((ex, i) => (
                  <button key={i} className="btn btn-example" onClick={() => loadExample(ex)}>
                    {ex.label}
                  </button>
                ))}
              </div>

              <div className="input-wrap">
                <label className="input-label" htmlFor="title-input">Review Title (optional)</label>
                <input
                  id="title-input"
                  className="input-field"
                  placeholder="e.g. Great product!"
                  value={title}
                  onChange={e => setTitle(e.target.value)}
                  onKeyDown={handleKey}
                />
              </div>

              <div className="input-wrap">
                <label className="input-label" htmlFor="review-textarea">Review Text</label>
                <textarea
                  id="review-textarea"
                  ref={textareaRef}
                  className="textarea"
                  placeholder="Paste a product review here… (Ctrl+Enter to analyze)"
                  value={text}
                  onChange={e => setText(e.target.value)}
                  onKeyDown={handleKey}
                  maxLength={5000}
                />
                <div className="char-count">{text.length} / 5000</div>
              </div>

              <div style={{ display: 'flex', gap: 10 }}>
                <button
                  className="btn btn-primary"
                  onClick={analyze}
                  disabled={!text.trim() || loading || !backendUp}
                >
                  {loading ? <><div className="spinner" /> Analyzing…</> : <><span>⚡</span> Analyze Sentiment</>}
                </button>
                <button className="btn btn-ghost" onClick={clearAll}>Clear</button>
              </div>
              {!backendUp && (
                <div className="error-banner" style={{ marginTop: 12 }}>
                  ⚠️ Backend offline. Run: <code style={{ marginLeft: 6 }}>uvicorn main:app --reload</code> in the backend folder.
                </div>
              )}
            </div>

            {/* Result card */}
            <div className="card">
              <div className="card-title">🎯 Prediction Result</div>
              <ResultPanel result={result} loading={loading} error={error} />
            </div>
          </div>

          {/* Bottom grid */}
          <div className="grid-2">
            {/* Model Info */}
            <div className="card">
              <div className="card-title">📊 Model Information</div>
              <ModelInfoCard info={modelInfo} />
            </div>

            {/* History */}
            <div className="card">
              <div className="card-title" style={{ justifyContent: 'space-between' }}>
                <span>🕐 Analysis History</span>
                {history.length > 0 && (
                  <button className="btn btn-ghost" style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                    onClick={() => setHistory([])}>Clear</button>
                )}
              </div>
              <HistoryList history={history} onSelect={(t) => { setText(t); setResult(null) }} />
            </div>
          </div>
        </main>

        {/* ── Footer ── */}
        <footer className="footer">
          Trained on&nbsp;
          <a href="https://huggingface.co/datasets/CyranoB/polarity" target="_blank" rel="noopener">
            CyranoB/polarity
          </a>
          &nbsp;· Amazon Review Polarity Dataset · Apache 2.0
        </footer>
      </div>
    </>
  )
}

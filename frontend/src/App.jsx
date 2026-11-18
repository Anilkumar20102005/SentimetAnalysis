import { useState, useEffect, useCallback, useRef } from 'react'
import confetti from 'canvas-confetti'
import {
  Sparkles,
  ThumbsUp,
  ThumbsDown,
  Zap,
  Layers,
  Scale,
  Cpu,
  CheckCircle2,
  AlertCircle,
  Copy,
  Check,
  Download,
  Search,
  Trash2,
  Activity,
  FileText,
  Clock
} from 'lucide-react'
import './index.css'

const API_BASE = '/api'

// ── Circular confidence ring ────────────────────────────────────────────────
function ConfidenceRing({ value, sentiment }) {
  const r = 28
  const circ = 2 * Math.PI * r
  const offset = circ - (value / 100) * circ
  const color = sentiment === 'positive' ? 'var(--positive)' : 'var(--negative)'

  return (
    <div className="confidence-ring-wrap" title={`Confidence: ${Math.round(value)}%`}>
      <svg viewBox="0 0 68 68" width="68" height="68">
        <circle cx="34" cy="34" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="6" />
        <circle
          cx="34"
          cy="34"
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 34 34)"
          style={{ transition: 'stroke-dashoffset 0.8s ease' }}
        />
      </svg>
      <div className="conf-text">
        <span style={{ color }}>{Math.round(value)}%</span>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.62rem' }}>CONF</span>
      </div>
    </div>
  )
}

// ── Keywords Section ────────────────────────────────────────────────────────
function KeywordsList({ keywords }) {
  if (!keywords || !keywords.length) return null
  return (
    <div className="keywords-section animate-in">
      <div className="keywords-title">
        <span>Influential Keywords</span>
        <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>TF-IDF + LR Weight</span>
      </div>
      <div className="keywords-pills">
        {keywords.map((kw, i) => (
          <span key={i} className={`keyword-tag ${kw.sentiment}`} title={`Weight: ${kw.weight}`}>
            {kw.sentiment === 'positive' ? '▲' : '▼'} {kw.word}
            <span className="kw-weight">{kw.weight > 0 ? `+${kw.weight}` : kw.weight}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

// ── Single Result Display ───────────────────────────────────────────────────
function ResultPanel({ result, loading, error, onCopy }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    if (!result) return
    const text = `Sentiment: ${result.sentiment.toUpperCase()} (${(result.confidence * 100).toFixed(1)}% confidence)\nPositive: ${(result.positive_prob * 100).toFixed(1)}% | Negative: ${(result.negative_prob * 100).toFixed(1)}%`
    navigator.clipboard?.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
    if (onCopy) onCopy('Analysis copied to clipboard!')
  }

  if (loading) {
    return (
      <div className="result-empty">
        <div className="spinner" style={{ width: 44, height: 44 }} />
        <p style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Analyzing sentiment…</p>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Tokenizing text & running classification</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="result-panel">
        <div className="error-banner">
          <AlertCircle size={20} />
          <div>
            <strong>Analysis Error:</strong> {error}
          </div>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="result-empty">
        <div className="result-empty-icon">
          <Sparkles size={28} />
        </div>
        <p style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Ready to Analyze</p>
        <p>Type a review on the left or select a quick example to see live sentiment detection</p>
      </div>
    )
  }

  const isPos = result.sentiment === 'positive'
  const conf = result.confidence * 100

  return (
    <div className="result-panel result-animate">
      {/* Main badge */}
      <div className={`sentiment-badge ${result.sentiment}`}>
        <div className="sentiment-badge-glow" />
        <div className="sentiment-icon-box">
          {isPos ? <ThumbsUp size={28} color="var(--positive)" /> : <ThumbsDown size={28} color="var(--negative)" />}
        </div>
        <div className="sentiment-label">
          <span className="sentiment-pill-tag">
            {isPos ? 'Polarity: Positive' : 'Polarity: Negative'}
          </span>
          <h3>{isPos ? 'POSITIVE REVIEW' : 'NEGATIVE REVIEW'}</h3>
          <p>{isPos ? 'Customer expressed high satisfaction.' : 'Customer expressed dissatisfaction or issues.'}</p>
        </div>
        <ConfidenceRing value={conf} sentiment={result.sentiment} />
      </div>

      {/* Probability bars */}
      <div className="prob-bars">
        <div className="prob-row">
          <div className="prob-header">
            <span className="prob-name" style={{ color: 'var(--positive-light)' }}>
              <CheckCircle2 size={14} /> Positive Probability
            </span>
            <span className="prob-pct" style={{ color: 'var(--positive-light)' }}>
              {(result.positive_prob * 100).toFixed(1)}%
            </span>
          </div>
          <div className="prob-bar-track">
            <div className="prob-bar-fill pos" style={{ width: `${result.positive_prob * 100}%` }} />
          </div>
        </div>
        <div className="prob-row">
          <div className="prob-header">
            <span className="prob-name" style={{ color: 'var(--negative-light)' }}>
              <AlertCircle size={14} /> Negative Probability
            </span>
            <span className="prob-pct" style={{ color: 'var(--negative-light)' }}>
              {(result.negative_prob * 100).toFixed(1)}%
            </span>
          </div>
          <div className="prob-bar-track">
            <div className="prob-bar-fill neg" style={{ width: `${result.negative_prob * 100}%` }} />
          </div>
        </div>
      </div>

      {/* Keywords section */}
      <KeywordsList keywords={result.keywords} />

      {/* Stats */}
      <div className="stats-row">
        <div className="stat-chip">
          <div className="stat-val">{conf.toFixed(1)}%</div>
          <div className="stat-label">Confidence</div>
        </div>
        <div className="stat-chip">
          <div className="stat-val">{result.inference_ms}ms</div>
          <div className="stat-label">Inference Latency</div>
        </div>
        <div className="stat-chip">
          <div className="stat-val">{result.processed_text ? result.processed_text.split(' ').filter(Boolean).length : 0}</div>
          <div className="stat-label">Features Matched</div>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 4 }}>
        <button className="btn btn-ghost btn-sm" onClick={handleCopy}>
          {copied ? <><Check size={14} /> Copied</> : <><Copy size={14} /> Copy Summary</>}
        </button>
      </div>
    </div>
  )
}

// ── Batch Analysis View ─────────────────────────────────────────────────────
function BatchView({ onNotify }) {
  const [inputText, setInputText] = useState('')
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState(null)
  const [filter, setFilter] = useState('all') // 'all', 'positive', 'negative'

  const sampleBatch = [
    "The camera quality is exceptional and the battery lasts two full days!",
    "Worst experience ever. It broke after two hours of light usage.",
    "Pretty decent value for the money, shipped quickly and well packaged.",
    "Completely useless. Does not work as described and customer support is non-existent.",
    "Super happy with this purchase! Highly recommended for anyone looking for good quality.",
    "Average item. It gets the job done but feels somewhat cheap.",
    "Defective right out of the box. Returning immediately for a full refund."
  ].join('\n')

  const handleRunBatch = async () => {
    const lines = inputText.split('\n').map(l => l.trim()).filter(Boolean)
    if (!lines.length) return
    setLoading(true)
    setData(null)

    try {
      const payload = {
        reviews: lines.slice(0, 50).map(text => ({ text }))
      }
      const res = await fetch(`${API_BASE}/predict/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      if (!res.ok) throw new Error(`Batch request failed: ${res.statusText}`)
      const json = await res.json()
      setData(json)
      if (onNotify) onNotify(`Successfully analyzed ${json.count} reviews!`)
    } catch (e) {
      if (onNotify) onNotify(e.message)
    } finally {
      setLoading(false)
    }
  }

  const exportCSV = () => {
    if (!data?.results?.length) return
    const headers = ['Review', 'Sentiment', 'Confidence', 'Pos_Prob', 'Neg_Prob', 'Latency_ms']
    const rows = data.results.map(r => [
      `"${(r.text || '').replace(/"/g, '""')}"`,
      r.sentiment,
      r.confidence,
      r.positive_prob,
      r.negative_prob,
      r.inference_ms
    ])
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n')
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement('a')
    link.setAttribute('href', encodedUri)
    link.setAttribute('download', `sentiment_batch_${Date.now()}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const filteredResults = data?.results?.filter(r => {
    if (filter === 'positive') return r.sentiment === 'positive'
    if (filter === 'negative') return r.sentiment === 'negative'
    return true
  }) || []

  return (
    <div className="card animate-in">
      <div className="card-header">
        <div className="card-title">
          <Layers size={18} color="var(--accent-light)" /> Batch Sentiment Processor
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn btn-ghost btn-sm" onClick={() => setInputText(sampleBatch)}>
            Load Sample Batch (7 Reviews)
          </button>
          {inputText && (
            <button className="btn btn-ghost btn-sm" onClick={() => setInputText('')}>
              Clear
            </button>
          )}
        </div>
      </div>

      <div className="input-wrap">
        <label className="input-label">Enter Reviews (one per line, up to 50)</label>
        <textarea
          className="textarea"
          rows={5}
          placeholder="Paste multiple product reviews here, one per line..."
          value={inputText}
          onChange={e => setInputText(e.target.value)}
        />
        <div className="input-meta-bar">
          <span>{inputText.split('\n').filter(l => l.trim()).length} reviews detected</span>
          <span>Max 50 items per batch</span>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 10, marginBottom: '1.5rem' }}>
        <button
          className="btn btn-primary"
          style={{ width: 'auto', minWidth: 200 }}
          onClick={handleRunBatch}
          disabled={!inputText.trim() || loading}
        >
          {loading ? <><div className="spinner" /> Processing Batch…</> : <><Zap size={16} /> Run Batch Analysis</>}
        </button>
      </div>

      {data && (
        <div className="animate-in">
          {/* Summary stats */}
          <div className="batch-summary-grid">
            <div className="batch-stat-card">
              <div className="batch-stat-val">{data.summary.total}</div>
              <div className="batch-stat-label">Total Analyzed</div>
            </div>
            <div className="batch-stat-card" style={{ borderColor: 'rgba(16,185,129,0.3)' }}>
              <div className="batch-stat-val" style={{ color: 'var(--positive-light)' }}>
                {data.summary.positive} ({data.summary.positive_pct}%)
              </div>
              <div className="batch-stat-label">Positive Reviews</div>
            </div>
            <div className="batch-stat-card" style={{ borderColor: 'rgba(244,63,94,0.3)' }}>
              <div className="batch-stat-val" style={{ color: 'var(--negative-light)' }}>
                {data.summary.negative} ({data.summary.negative_pct}%)
              </div>
              <div className="batch-stat-label">Negative Reviews</div>
            </div>
            <div className="batch-stat-card">
              <div className="batch-stat-val" style={{ color: 'var(--cyan)' }}>
                {(data.summary.avg_confidence * 100).toFixed(1)}%
              </div>
              <div className="batch-stat-label">Avg Confidence</div>
            </div>
          </div>

          {/* Action bar */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <div style={{ display: 'flex', gap: 6 }}>
              <button
                className={`btn btn-ghost btn-sm ${filter === 'all' ? 'btn-example' : ''}`}
                onClick={() => setFilter('all')}
              >
                All ({data.results.length})
              </button>
              <button
                className={`btn btn-ghost btn-sm ${filter === 'positive' ? 'btn-example' : ''}`}
                onClick={() => setFilter('positive')}
              >
                Positive ({data.summary.positive})
              </button>
              <button
                className={`btn btn-ghost btn-sm ${filter === 'negative' ? 'btn-example' : ''}`}
                onClick={() => setFilter('negative')}
              >
                Negative ({data.summary.negative})
              </button>
            </div>

            <button className="btn btn-ghost btn-sm" onClick={exportCSV}>
              <Download size={14} /> Export CSV
            </button>
          </div>

          {/* Table */}
          <div className="batch-table-container">
            <table className="batch-table">
              <thead>
                <tr>
                  <th style={{ width: '50%' }}>Review Snippet</th>
                  <th>Sentiment</th>
                  <th>Confidence</th>
                  <th>Top Keywords</th>
                </tr>
              </thead>
              <tbody>
                {filteredResults.map((item, idx) => (
                  <tr key={idx}>
                    <td>
                      <div style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                        {item.text}
                      </div>
                    </td>
                    <td>
                      <span className={`badge-sentiment ${item.sentiment}`}>
                        {item.sentiment === 'positive' ? '✓ Positive' : '✕ Negative'}
                      </span>
                    </td>
                    <td style={{ fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                      {(item.confidence * 100).toFixed(1)}%
                    </td>
                    <td>
                      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                        {item.keywords?.slice(0, 3).map((k, ki) => (
                          <span
                            key={ki}
                            className={`keyword-tag ${k.sentiment}`}
                            style={{ fontSize: '0.7rem', padding: '2px 6px' }}
                          >
                            {k.word}
                          </span>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Comparison Mode ─────────────────────────────────────────────────────────
function CompareView({ onNotify }) {
  const [textA, setTextA] = useState("This product exceeded all expectations! Battery life is unbelievable.")
  const [textB, setTextB] = useState("Disappointing build quality. The battery drains within two hours.")
  const [resA, setResA] = useState(null)
  const [resB, setResB] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleCompare = async () => {
    if (!textA.trim() || !textB.trim()) return
    setLoading(true)
    try {
      const [rA, rB] = await Promise.all([
        fetch(`${API_BASE}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: textA.trim() })
        }).then(r => r.json()),
        fetch(`${API_BASE}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: textB.trim() })
        }).then(r => r.json())
      ])
      setResA(rA)
      setResB(rB)
      if (onNotify) onNotify('Comparison completed!')
    } catch (e) {
      if (onNotify) onNotify(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="animate-in">
      <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
        <button
          className="btn btn-primary"
          style={{ width: 'auto', padding: '10px 32px' }}
          onClick={handleCompare}
          disabled={loading}
        >
          {loading ? <><div className="spinner" /> Comparing…</> : <><Scale size={16} /> Compare Sentiment Shift</>}
        </button>
      </div>

      <div className="compare-container">
        {/* Review A */}
        <div className="compare-card">
          <div className="card-title">
            <span style={{ color: 'var(--accent-light)' }}>Variant A</span> Product Review
          </div>
          <textarea
            className="textarea"
            rows={4}
            value={textA}
            onChange={e => setTextA(e.target.value)}
            placeholder="Review Variant A..."
          />
          {resA && (
            <div className="result-panel">
              <div className={`sentiment-badge ${resA.sentiment}`}>
                <div className="sentiment-icon-box">
                  {resA.sentiment === 'positive' ? <ThumbsUp size={24} /> : <ThumbsDown size={24} />}
                </div>
                <div className="sentiment-label">
                  <h3>{resA.sentiment.toUpperCase()}</h3>
                  <p>Confidence: {(resA.confidence * 100).toFixed(1)}%</p>
                </div>
              </div>
              <KeywordsList keywords={resA.keywords} />
            </div>
          )}
        </div>

        {/* Review B */}
        <div className="compare-card">
          <div className="card-title">
            <span style={{ color: 'var(--cyan)' }}>Variant B</span> Product Review
          </div>
          <textarea
            className="textarea"
            rows={4}
            value={textB}
            onChange={e => setTextB(e.target.value)}
            placeholder="Review Variant B..."
          />
          {resB && (
            <div className="result-panel">
              <div className={`sentiment-badge ${resB.sentiment}`}>
                <div className="sentiment-icon-box">
                  {resB.sentiment === 'positive' ? <ThumbsUp size={24} /> : <ThumbsDown size={24} />}
                </div>
                <div className="sentiment-label">
                  <h3>{resB.sentiment.toUpperCase()}</h3>
                  <p>Confidence: {(resB.confidence * 100).toFixed(1)}%</p>
                </div>
              </div>
              <KeywordsList keywords={resB.keywords} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Model Info Card ────────────────────────────────────────────────────────
function ModelInfoCard({ info }) {
  if (!info) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
        Connecting to ML Inference Engine…
      </div>
    )
  }

  return (
    <div className="model-info-grid">
      <div className="model-stat">
        <div className="ms-val">{info.test_accuracy ? `${(info.test_accuracy * 100).toFixed(1)}%` : '86.9%'}</div>
        <div className="ms-label">Validation Accuracy</div>
      </div>
      <div className="model-stat">
        <div className="ms-val">{info.test_roc_auc ? info.test_roc_auc.toFixed(3) : '0.945'}</div>
        <div className="ms-label">ROC-AUC Score</div>
      </div>
      <div className="model-stat">
        <div className="ms-val">{info.train_samples ? `${(info.train_samples / 1000).toFixed(1)}k` : '21.3k'}</div>
        <div className="ms-label">Training Reviews</div>
      </div>
      <div className="model-stat" style={{ gridColumn: '1 / -1' }}>
        <div className="ms-val" style={{ fontSize: '0.88rem', fontWeight: 600 }}>
          {info.model_type || 'TF-IDF Vectorizer + Logistic Regression'}
        </div>
        <div className="ms-label">ML Pipeline Architecture</div>
      </div>
      <div className="model-stat" style={{ gridColumn: '1 / -1' }}>
        <div className="ms-val" style={{ fontSize: '0.82rem', fontWeight: 600 }}>
          {info.dataset || 'Amazon Review Polarity (CyranoB/polarity)'}
        </div>
        <div className="ms-label">Training Corpus Source</div>
      </div>
    </div>
  )
}

// ── History List ────────────────────────────────────────────────────────────
function HistoryList({ history, onSelect }) {
  const [searchTerm, setSearchTerm] = useState('')
  const [filter, setFilter] = useState('all')

  const filtered = history.filter(item => {
    const matchText = (item.text || '').toLowerCase().includes(searchTerm.toLowerCase())
    if (!matchText) return false
    if (filter === 'positive') return item.sentiment === 'positive'
    if (filter === 'negative') return item.sentiment === 'negative'
    return true
  })

  if (!history.length) {
    return (
      <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center', padding: '1.5rem 0' }}>
        No recent analyses yet. Submit a review above to populate your history!
      </p>
    )
  }

  return (
    <div>
      {/* Search & filters */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <div style={{ position: 'relative', flex: 1 }}>
          <input
            className="input-field"
            style={{ padding: '6px 12px 6px 30px', fontSize: '0.8rem' }}
            placeholder="Search history…"
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
          />
          <Search size={14} style={{ position: 'absolute', left: 10, top: 9, color: 'var(--text-muted)' }} />
        </div>
        <button
          className={`btn btn-ghost btn-sm ${filter === 'all' ? 'btn-example' : ''}`}
          onClick={() => setFilter('all')}
        >
          All
        </button>
        <button
          className={`btn btn-ghost btn-sm ${filter === 'positive' ? 'btn-example' : ''}`}
          onClick={() => setFilter('positive')}
        >
          Pos
        </button>
        <button
          className={`btn btn-ghost btn-sm ${filter === 'negative' ? 'btn-example' : ''}`}
          onClick={() => setFilter('negative')}
        >
          Neg
        </button>
      </div>

      <div className="history-list">
        {filtered.map((item, i) => (
          <div key={i} className="history-item" onClick={() => onSelect(item.text)}>
            <div className={`hist-dot ${item.sentiment}`} />
            <div className="hist-text">{item.text}</div>
            <div className={`hist-conf ${item.sentiment}`}>
              {(item.confidence * 100).toFixed(0)}%
            </div>
          </div>
        ))}
        {!filtered.length && (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem', padding: '1rem' }}>
            No matching reviews found.
          </div>
        )}
      </div>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState('single') // 'single', 'batch', 'compare', 'model'
  const [text, setText] = useState('')
  const [title, setTitle] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('sentiment_history')
      return saved ? JSON.parse(saved) : []
    } catch {
      return []
    }
  })
  const [modelInfo, setModelInfo] = useState(null)
  const [backendUp, setBackendUp] = useState(true)
  const [examples, setExamples] = useState([])
  const [toast, setToast] = useState(null)
  const textareaRef = useRef(null)

  // Persist history
  useEffect(() => {
    try {
      localStorage.setItem('sentiment_history', JSON.stringify(history.slice(0, 30)))
    } catch {}
  }, [history])

  const showToast = (msg) => {
    setToast(msg)
    setTimeout(() => setToast(null), 3000)
  }

  // Fetch model info + examples on mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const infoRes = await fetch(`${API_BASE}/model/info`)
        if (infoRes.ok) {
          const info = await infoRes.json()
          setModelInfo(info)
          setBackendUp(true)
        } else {
          setBackendUp(false)
        }
      } catch {
        setBackendUp(false)
      }

      try {
        const exRes = await fetch(`${API_BASE}/examples`)
        if (exRes.ok) {
          const ex = await exRes.json()
          setExamples(ex.examples || [])
        }
      } catch {}
    }

    checkBackend()
    const timer = setInterval(checkBackend, 15000)
    return () => clearInterval(timer)
  }, [])

  // Analyze single review
  const analyze = useCallback(async () => {
    if (!text.trim() || loading) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const resp = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text.trim(),
          title: title.trim() || undefined
        }),
      })

      if (!resp.ok) {
        const err = await resp.json()
        throw new Error(err.detail || `Server error (${resp.status})`)
      }

      const data = await resp.json()
      setResult(data)
      setHistory(prev => [{ text: text.trim(), ...data }, ...prev].slice(0, 30))

      // Trigger celebration confetti for high-confidence positive review
      if (data.sentiment === 'positive' && data.confidence >= 0.85) {
        confetti({
          particleCount: 50,
          spread: 60,
          origin: { y: 0.7 },
          colors: ['#10b981', '#6366f1', '#06b6d4']
        })
      }
    } catch (e) {
      setError(e.message || 'Could not connect to the backend server.')
    } finally {
      setLoading(false)
    }
  }, [text, title, loading])

  const handleKey = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      analyze()
    }
  }

  const clearAll = () => {
    setText('')
    setTitle('')
    setResult(null)
    setError(null)
    textareaRef.current?.focus()
  }

  const loadExample = (ex) => {
    setText(ex.text)
    setTitle(ex.label)
    setResult(null)
    setError(null)
  }

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0

  return (
    <>
      {/* Background orbs */}
      <div className="bg-orbs">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      <div className="layout">
        {/* ── Header ── */}
        <header className="header">
          <div className="header-inner">
            <div className="logo">
              <div className="logo-icon-wrap">
                <Sparkles size={20} color="var(--accent-light)" />
              </div>
              <span className="logo-text">SentiScope AI</span>
              <span className="logo-tag">v2.0 Pro</span>
            </div>

            <div className="header-actions">
              <div className="nav-badge">
                <div className={`dot ${backendUp ? '' : 'offline'}`} />
                {backendUp ? 'FastAPI Model Live' : 'Backend Disconnected'}
              </div>
            </div>
          </div>
        </header>

        {/* ── Main ── */}
        <main className="main animate-in">
          {/* Hero */}
          <div className="hero">
            <div className="hero-label">
              <Cpu size={14} /> Industrial NLP Classifier
            </div>
            <h1>Product Sentiment Intelligence</h1>
            <p>
              Instantly decode customer sentiment from real-world product reviews with keyword explainability,
              batch processing, and precision metrics.
            </p>
          </div>

          {/* Navigation Tabs */}
          <div className="nav-tabs-container">
            <div className="nav-tabs">
              <button
                className={`nav-tab ${activeTab === 'single' ? 'active' : ''}`}
                onClick={() => setActiveTab('single')}
              >
                <Zap size={15} /> Single Review
              </button>
              <button
                className={`nav-tab ${activeTab === 'batch' ? 'active' : ''}`}
                onClick={() => setActiveTab('batch')}
              >
                <Layers size={15} /> Batch Analyzer
              </button>
              <button
                className={`nav-tab ${activeTab === 'compare' ? 'active' : ''}`}
                onClick={() => setActiveTab('compare')}
              >
                <Scale size={15} /> A/B Comparison
              </button>
              <button
                className={`nav-tab ${activeTab === 'model' ? 'active' : ''}`}
                onClick={() => setActiveTab('model')}
              >
                <Activity size={15} /> Model Metrics
              </button>
            </div>
          </div>

          {/* Tab Content */}
          {activeTab === 'single' && (
            <>
              {/* Main 2-column Grid */}
              <div className="grid-2">
                {/* Input Card */}
                <div className="card">
                  <div className="card-header">
                    <div className="card-title">
                      <FileText size={16} color="var(--accent-light)" /> Input Review
                    </div>
                    {text && (
                      <button className="btn btn-ghost btn-sm" onClick={clearAll}>
                        Clear
                      </button>
                    )}
                  </div>

                  {/* Examples strip */}
                  <div className="examples-strip">
                    <span className="examples-label">Preset Test Samples</span>
                    {examples.map((ex, i) => (
                      <button key={i} className="btn-example" onClick={() => loadExample(ex)}>
                        {ex.label}
                      </button>
                    ))}
                  </div>

                  <div className="input-wrap">
                    <label className="input-label" htmlFor="title-input">
                      Review Title (optional)
                    </label>
                    <input
                      id="title-input"
                      className="input-field"
                      placeholder="e.g. Best battery life on the market!"
                      value={title}
                      onChange={e => setTitle(e.target.value)}
                      onKeyDown={handleKey}
                    />
                  </div>

                  <div className="input-wrap">
                    <div className="input-header">
                      <label className="input-label" htmlFor="review-textarea">
                        Review Content
                      </label>
                    </div>
                    <textarea
                      id="review-textarea"
                      ref={textareaRef}
                      className="textarea"
                      placeholder="Paste any customer review here… Press Ctrl + Enter to analyze immediately."
                      value={text}
                      onChange={e => setText(e.target.value)}
                      onKeyDown={handleKey}
                      maxLength={5000}
                    />
                    <div className="input-meta-bar">
                      <span>{wordCount} words · {text.length} characters</span>
                      <span>Shortcut: Ctrl + Enter</span>
                    </div>
                  </div>

                  <div style={{ marginTop: 14 }}>
                    <button
                      className="btn btn-primary"
                      onClick={analyze}
                      disabled={!text.trim() || loading || !backendUp}
                    >
                      {loading ? (
                        <><div className="spinner" /> Classifying Review…</>
                      ) : (
                        <><Zap size={16} /> Run Sentiment Analysis</>
                      )}
                    </button>
                  </div>

                  {!backendUp && (
                    <div className="error-banner" style={{ marginTop: 14 }}>
                      <AlertCircle size={18} />
                      <div>
                        Backend is offline. Start the server with: <code style={{ color: 'white' }}>python run.py</code>
                      </div>
                    </div>
                  )}
                </div>

                {/* Result Card */}
                <div className="card">
                  <div className="card-header">
                    <div className="card-title">
                      <Activity size={16} color="var(--positive-light)" /> Prediction Results
                    </div>
                  </div>
                  <ResultPanel
                    result={result}
                    loading={loading}
                    error={error}
                    onCopy={showToast}
                  />
                </div>
              </div>

              {/* Bottom 2-column Grid: Model Info & History */}
              <div className="grid-2">
                <div className="card">
                  <div className="card-header">
                    <div className="card-title">
                      <Cpu size={16} color="var(--accent-light)" /> Model Specifications
                    </div>
                  </div>
                  <ModelInfoCard info={modelInfo} />
                </div>

                <div className="card">
                  <div className="card-header">
                    <div className="card-title">
                      <Clock size={16} color="var(--cyan)" /> Analysis History ({history.length})
                    </div>
                    {history.length > 0 && (
                      <button
                        className="btn btn-ghost btn-sm"
                        onClick={() => { setHistory([]); showToast('History cleared') }}
                      >
                        <Trash2 size={13} /> Clear
                      </button>
                    )}
                  </div>
                  <HistoryList
                    history={history}
                    onSelect={(t) => {
                      setText(t)
                      setResult(null)
                      window.scrollTo({ top: 120, behavior: 'smooth' })
                    }}
                    onClear={() => setHistory([])}
                  />
                </div>
              </div>
            </>
          )}

          {activeTab === 'batch' && (
            <BatchView onNotify={showToast} />
          )}

          {activeTab === 'compare' && (
            <CompareView onNotify={showToast} />
          )}

          {activeTab === 'model' && (
            <div className="card animate-in" style={{ maxWidth: 800, margin: '0 auto' }}>
              <div className="card-header">
                <div className="card-title">
                  <Cpu size={18} color="var(--accent-light)" /> Comprehensive Architecture & Performance
                </div>
              </div>
              <ModelInfoCard info={modelInfo} />

              <div style={{ marginTop: '2rem', display: 'flex', flexDirection: 'column', gap: 14 }}>
                <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  How the Sentiment Engine Operates
                </h4>
                <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
                  1. <strong>Text Cleaning & Lemmatization:</strong> HTML tags and URLs are stripped, letters converted to lowercase, and tokens lemmatized to their base forms using WordNet with English stopword removal.
                </p>
                <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
                  2. <strong>TF-IDF N-gram Matrix:</strong> Unigrams and bigrams are weighted across a vocabulary of up to 80,000 top features to capture contextual phrasing (e.g., "highly recommended", "terrible battery").
                </p>
                <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
                  3. <strong>Logistic Regression Classifier:</strong> Calibrated probabilities are generated with L2 regularization, delivering sub-millisecond inference and transparent feature coefficient weights.
                </p>
              </div>
            </div>
          )}
        </main>

        {/* Toast Feedback */}
        {toast && (
          <div className="toast">
            <CheckCircle2 size={16} color="var(--positive-light)" />
            <span>{toast}</span>
          </div>
        )}

        {/* ── Footer ── */}
        <footer className="footer">
          Product Sentiment Analysis Platform · Powered by{' '}
          <a href="https://huggingface.co/datasets/CyranoB/polarity" target="_blank" rel="noopener noreferrer">
            Amazon Review Polarity Dataset
          </a>{' '}
          · Scikit-Learn, FastAPI & React 19
        </footer>
      </div>
    </>
  )
}

import { useState, useRef } from 'react'
import './App.css'

const API = 'http://localhost:8000'

const STAGE_LABELS = {
  input:        'Input Photo',
  restored:     'After Restoration',
  edges:        'Canny Edges',
  rectified:    'Tilt Corrected',
  strip:        'Polar Unwrap',
  needle_clean: 'Needle Mask',
}

export default function App() {
  const [imageFile, setImageFile]   = useState(null)
  const [imagePreview, setPreview]  = useState(null)
  const [form, setForm]             = useState({ vmin: '', vmax: '', unit: '', safe_lo: '', safe_hi: '', direction: 'clockwise', scale_type: 'linear' })
  const [result, setResult]         = useState(null)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)
  const [activeStage, setActiveStage] = useState('input')
  const dropRef = useRef()

  const handleFile = (file) => {
    if (!file) return
    setImageFile(file)
    setPreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
  }

  const onDrop = (e) => {
    e.preventDefault()
    dropRef.current?.classList.remove('drag-over')
    handleFile(e.dataTransfer.files[0])
  }

  const onSubmit = async (e) => {
    e.preventDefault()
    if (!imageFile) { setError('Please upload a gauge image first.'); return }
    if (!form.vmin || !form.vmax) { setError('vmin and vmax are required.'); return }

    setLoading(true); setError(null); setResult(null)

    const fd = new FormData()
    fd.append('image', imageFile)
    fd.append('vmin', form.vmin)
    fd.append('vmax', form.vmax)
    fd.append('unit', form.unit)
    fd.append('direction', form.direction)
    fd.append('scale_type', form.scale_type)
    if (form.safe_lo !== '') fd.append('safe_lo', form.safe_lo)
    if (form.safe_hi !== '') fd.append('safe_hi', form.safe_hi)

    try {
      const res = await fetch(`${API}/read`, { method: 'POST', body: fd })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setResult(data)
      setActiveStage('input')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const statusColor = result
    ? result.status === 'reliable' ? 'var(--green)'
    : result.status === 'unreliable' ? 'var(--yellow)'
    : 'var(--red)'
    : 'var(--text-muted)'

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">⊙</span>
            <span className="logo-text">DialReader</span>
          </div>
          <p className="header-sub">Automated Analog Gauge Reading · Classical Computer Vision Pipeline</p>
        </div>
      </header>

      <main className="main">
        {/* ── Left column: form ── */}
        <section className="card form-card">
          <h2 className="section-title">📸 Gauge Input</h2>

          {/* Drop zone */}
          <div
            ref={dropRef}
            className={`dropzone ${imagePreview ? 'has-image' : ''}`}
            onClick={() => document.getElementById('file-input').click()}
            onDragOver={(e) => { e.preventDefault(); dropRef.current?.classList.add('drag-over') }}
            onDragLeave={() => dropRef.current?.classList.remove('drag-over')}
            onDrop={onDrop}
          >
            {imagePreview
              ? <img src={imagePreview} alt="gauge" className="preview-img" />
              : (
                <div className="drop-placeholder">
                  <span className="drop-icon">🖼</span>
                  <p>Drop gauge image here</p>
                  <span className="drop-hint">or click to browse · JPG / PNG</span>
                </div>
              )
            }
          </div>
          <input id="file-input" type="file" accept="image/*" hidden onChange={e => handleFile(e.target.files[0])} />

          {/* Scale Parameters */}
          <form onSubmit={onSubmit} className="param-form">
            <h3 className="param-heading">Scale Parameters</h3>

            <div className="row2">
              <label className="field">
                <span>Min value (vmin) <em>*</em></span>
                <input type="number" placeholder="e.g. 0" value={form.vmin} onChange={e => setForm({...form, vmin: e.target.value})} required />
              </label>
              <label className="field">
                <span>Max value (vmax) <em>*</em></span>
                <input type="number" placeholder="e.g. 10" value={form.vmax} onChange={e => setForm({...form, vmax: e.target.value})} required />
              </label>
            </div>

            <label className="field">
              <span>Unit (optional)</span>
              <input type="text" placeholder="e.g. bar, psi, °C, km/h" value={form.unit} onChange={e => setForm({...form, unit: e.target.value})} />
            </label>

            <div className="row2">
              <label className="field">
                <span>Direction</span>
                <select value={form.direction} onChange={e => setForm({...form, direction: e.target.value})}>
                  <option value="clockwise">Clockwise</option>
                  <option value="anticlockwise">Anticlockwise</option>
                </select>
              </label>
              <label className="field">
                <span>Scale type</span>
                <select value={form.scale_type} onChange={e => setForm({...form, scale_type: e.target.value})}>
                  <option value="linear">Linear</option>
                  <option value="log">Logarithmic</option>
                </select>
              </label>
            </div>

            <div className="divider" />
            <h3 className="param-heading">Safe Range <span className="optional">(for warning alert)</span></h3>

            <div className="row2">
              <label className="field">
                <span>Safe lower limit</span>
                <input type="number" placeholder="optional" value={form.safe_lo} onChange={e => setForm({...form, safe_lo: e.target.value})} />
              </label>
              <label className="field">
                <span>Safe upper limit</span>
                <input type="number" placeholder="optional" value={form.safe_hi} onChange={e => setForm({...form, safe_hi: e.target.value})} />
              </label>
            </div>

            {error && <div className="error-box">⚠ {error}</div>}

            <button type="submit" className="btn-run" disabled={loading}>
              {loading ? <><span className="spinner" /> Analysing…</> : '▶  Run Pipeline'}
            </button>
          </form>
        </section>

        {/* ── Right column: results ── */}
        <section className="right-col">

          {/* Reading result card */}
          <div className="card result-card">
            <h2 className="section-title">📊 Reading</h2>

            {!result && !loading && (
              <div className="empty-state">Upload a gauge image and fill in the scale to get a reading.</div>
            )}
            {loading && (
              <div className="empty-state">
                <span className="spinner big" /> Running 8-stage pipeline…
              </div>
            )}

            {result && (
              <>
                {/* Status pill */}
                <div className="status-row">
                  <span className="status-pill" style={{ background: statusColor + '22', color: statusColor, border: `1px solid ${statusColor}55` }}>
                    {result.status === 'reliable' ? '✅' : result.status === 'unreliable' ? '⚠️' : '❌'} {result.status.toUpperCase()}
                  </span>
                  <span className="confidence">Confidence: {(result.confidence * 100).toFixed(0)}%</span>
                </div>

                {/* Big value */}
                {result.ok && (
                  <div className="reading-value">
                    <span className="value-num">{parseFloat(result.value).toFixed(3)}</span>
                    <span className="value-unit">{result.unit}</span>
                    <span className="value-uncertainty">± {parseFloat(result.uncertainty).toFixed(3)}</span>
                  </div>
                )}

                {/* Range warning */}
                {result.range_warning && (
                  <div className="warning-banner">
                    {result.range_warning}
                  </div>
                )}

                {/* In range */}
                {result.in_range === true && (
                  <div className="ok-banner">
                    ✅ Within safe range
                  </div>
                )}

                {/* Failure reasons */}
                {result.reasons.length > 0 && (
                  <div className="reasons">
                    <strong>Quality flags:</strong>
                    <ul>{result.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
                  </div>
                )}

                {/* Timing */}
                <div className="timing-row">
                  {Object.entries(result.timings).filter(([k]) => k !== 'total').map(([k, v]) => (
                    <div className="timing-chip" key={k}><span>{k}</span><strong>{v} ms</strong></div>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Pipeline stages image viewer */}
          {result && result.stage_images && Object.keys(result.stage_images).length > 0 && (
            <div className="card stages-card">
              <h2 className="section-title">🔬 Pipeline Stages</h2>
              <div className="stage-tabs">
                {Object.keys(result.stage_images).map(k => (
                  <button
                    key={k}
                    className={`stage-tab ${activeStage === k ? 'active' : ''}`}
                    onClick={() => setActiveStage(k)}
                  >
                    {STAGE_LABELS[k] ?? k}
                  </button>
                ))}
              </div>
              <div className="stage-img-wrap">
                <img
                  key={activeStage}
                  src={`data:image/png;base64,${result.stage_images[activeStage]}`}
                  alt={activeStage}
                  className="stage-img"
                />
                <span className="stage-label">{STAGE_LABELS[activeStage] ?? activeStage}</span>
              </div>
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        DialReader · IPCV Course-Based Project · Classical CV Pipeline · No deep learning
      </footer>
    </div>
  )
}

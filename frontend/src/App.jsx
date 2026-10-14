import { useEffect, useRef, useState } from 'react'
import './App.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const EMPTY_FORM = {
  gauge_type: 'pressure_gauge', vmin: '', vmax: '', unit: '', direction: 'clockwise', scale_type: 'linear',
  n_ticks: '', safe_lo: '', safe_hi: '',
}

const fmt = (x, digits = 4) => (x === null || x === undefined ? '–' : Number(x).toPrecision(digits).replace(/\.?0+$/, ''))

export default function App() {
  const [types, setTypes]           = useState([])
  const [samples, setSamples]       = useState([])
  const [canParse, setCanParse]     = useState(false)
  const [apiDown, setApiDown]       = useState(false)
  const [imageFile, setImageFile]   = useState(null)
  const [imagePreview, setPreview]  = useState(null)
  const [sample, setSample]         = useState(null)       // the example photo in use, with its true value
  const [form, setForm]             = useState(EMPTY_FORM)
  const [scaleSource, setScaleSource] = useState('manual')
  const [result, setResult]         = useState(null)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)
  const [step, setStep]             = useState(0)
  const dropRef = useRef()

  const preset = types.find(t => t.key === form.gauge_type)
  const set = (changes) => setForm(f => ({ ...f, ...changes }))

  // Presets and example photos from the API.
  useEffect(() => {
    Promise.all([fetch(`${API}/gauges`).then(r => r.json()), fetch(`${API}/samples`).then(r => r.json())])
      .then(([g, s]) => {
        setTypes(g.types)
        setCanParse(g.scale_from_photo)
        setSamples(s.samples)
        const first = g.types[0]
        if (first) applyPreset(first)
      })
      .catch(() => setApiDown(true))
  }, [])

  function applyPreset(p, range) {
    const [lo, hi] = range || p.ranges[0] || ['', '']
    setForm(f => ({ ...f, gauge_type: p.key, vmin: String(lo), vmax: String(hi), unit: p.unit,
                    scale_type: p.scale_type, direction: p.direction, n_ticks: '' }))
  }

  const onGaugeType = (key) => {
    const p = types.find(t => t.key === key)
    if (p) applyPreset(p)
  }

  const handleFile = (file) => {
    if (!file) return
    setImageFile(file)
    setPreview(URL.createObjectURL(file))
    setSample(null)
    setResult(null)
    setError(null)
  }

  const loadSample = async (name) => {
    const s = samples.find(x => x.file === name)
    if (!s) return
    const blob = await fetch(`${API}/samples/${s.file}`).then(r => r.blob())
    setImageFile(new File([blob], s.file, { type: 'image/jpeg' }))
    setPreview(URL.createObjectURL(blob))
    setSample(s)
    setResult(null)
    setError(null)
    // An example photo comes with its exact scale, tick count included.
    setForm(f => ({ ...f, gauge_type: s.gauge_type, vmin: String(s.scale.vmin), vmax: String(s.scale.vmax),
                    unit: types.find(t => t.key === s.gauge_type)?.unit || s.scale.unit,
                    scale_type: s.scale.scale_type, direction: s.scale.direction, n_ticks: String(s.scale.n_ticks) }))
  }

  const onDrop = (e) => {
    e.preventDefault()
    dropRef.current?.classList.remove('drag-over')
    handleFile(e.dataTransfer.files[0])
  }

  const run = async (override = {}) => {
    const f = { ...form, ...override }
    if (!imageFile) { setError('Upload a gauge photo or pick an example first.'); return }
    if (scaleSource === 'manual' && (f.vmin === '' || f.vmax === '')) {
      setError('Enter the minimum and maximum printed on the dial.'); return
    }
    setLoading(true); setError(null); setResult(null)

    const fd = new FormData()
    fd.append('image', imageFile)
    fd.append('gauge_type', f.gauge_type)
    fd.append('scale_source', scaleSource)
    for (const k of ['vmin', 'vmax', 'n_ticks', 'safe_lo', 'safe_hi']) if (f[k] !== '') fd.append(k, f[k])
    fd.append('unit', f.unit)
    fd.append('direction', f.direction)
    fd.append('scale_type', f.scale_type)

    try {
      const res = await fetch(`${API}/read`, { method: 'POST', body: fd })
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setResult(data)
      setStep(Math.max(0, data.stages.length - 1))      // open on the final reading
    } catch (err) {
      setError(err.message === 'Failed to fetch' ? `Cannot reach the API at ${API}. Is it running?` : err.message)
    } finally {
      setLoading(false)
    }
  }

  const pickRange = (r) => {
    set({ vmin: String(r[0]), vmax: String(r[1]), n_ticks: '' })
    run({ vmin: String(r[0]), vmax: String(r[1]), n_ticks: '' })
  }

  const statusColor = !result ? 'var(--text-muted)'
    : result.status === 'reliable' ? 'var(--green)' : result.status === 'unreliable' ? 'var(--yellow)' : 'var(--red)'
  const stage = result?.stages?.[step]
  const sc = result?.scale_check
  const error_fs = result?.ok && sample
    ? 100 * Math.abs(result.value - sample.true_value) / Math.abs(sample.scale.vmax - sample.scale.vmin) : null

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <img src="/favicon.svg" alt="" className="logo-icon" />
            <span className="logo-text">DialReader</span>
          </div>
          <p className="header-sub">Reads an analogue gauge from one photo, with an error bar, and says when it is unsure</p>
        </div>
      </header>

      {apiDown && (
        <div className="api-down">Cannot reach the DialReader API at {API}. Start it with <code>uvicorn api:app --port 8000</code> and reload.</div>
      )}

      <main className="main">
        {/* ── Left column: photo and scale ── */}
        <section className="card form-card">
          <h2 className="section-title">1 · Photo</h2>

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
                  <p>Drop a gauge photo here</p>
                  <span className="drop-hint">or click to browse · JPG / PNG · one round dial, single needle</span>
                </div>
              )}
          </div>
          <input id="file-input" type="file" accept="image/*" hidden onChange={e => handleFile(e.target.files[0])} />

          {samples.length > 0 && (
            <label className="field sample-pick">
              <span>…or try an example photo</span>
              <select value={sample?.file || ''} onChange={e => loadSample(e.target.value)}>
                <option value="" disabled>Choose an example</option>
                {samples.map(s => <option key={s.file} value={s.file}>{s.label}</option>)}
              </select>
            </label>
          )}

          <form onSubmit={e => { e.preventDefault(); run() }} className="param-form">
            <div className="divider" />
            <h2 className="section-title">2 · Gauge and scale</h2>

            <label className="field">
              <span>Gauge type</span>
              <select value={form.gauge_type} onChange={e => onGaugeType(e.target.value)}>
                {types.map(t => <option key={t.key} value={t.key}>{t.label}</option>)}
              </select>
            </label>

            {canParse && (
              <div className="segmented">
                <button type="button" className={scaleSource === 'manual' ? 'on' : ''} onClick={() => setScaleSource('manual')}>
                  Enter the scale
                </button>
                <button type="button" className={scaleSource === 'photo' ? 'on' : ''} onClick={() => setScaleSource('photo')}>
                  Read it from the photo (AI)
                </button>
              </div>
            )}
            {scaleSource === 'photo' && (
              <p className="hint">A vision-language model reads the printed numbers; the result is only used if it lines
                up with the tick marks the pipeline found. The values below are the fallback.</p>
            )}

            {preset?.ranges?.length > 0 && (
              <div className="chips">
                <span className="chips-label">Common ranges</span>
                {preset.ranges.map(r => (
                  <button type="button" key={r.join()} className={`chip ${String(r[0]) === form.vmin && String(r[1]) === form.vmax ? 'on' : ''}`}
                          onClick={() => set({ vmin: String(r[0]), vmax: String(r[1]), n_ticks: '' })}>
                    {r[0]} – {r[1]}
                  </button>
                ))}
              </div>
            )}

            <div className="row2">
              <label className="field">
                <span>Lowest number on the dial <em>*</em></span>
                <input type="number" step="any" value={form.vmin} onChange={e => set({ vmin: e.target.value })} />
              </label>
              <label className="field">
                <span>Highest number on the dial <em>*</em></span>
                <input type="number" step="any" value={form.vmax} onChange={e => set({ vmax: e.target.value })} />
              </label>
            </div>

            <div className="row2">
              <label className="field">
                <span>Unit</span>
                <input type="text" list="unit-options" value={form.unit} onChange={e => set({ unit: e.target.value })} />
                <datalist id="unit-options">
                  {[preset?.unit, ...(preset?.units || [])].filter(Boolean).map(u => <option key={u} value={u} />)}
                </datalist>
              </label>
              <label className="field">
                <span>Scale</span>
                <select value={form.scale_type} onChange={e => set({ scale_type: e.target.value })}>
                  <option value="linear">Linear (even steps)</option>
                  <option value="log">Logarithmic</option>
                </select>
              </label>
            </div>

            <label className="field">
              <span>Numbers increase</span>
              <div className="segmented">
                <button type="button" className={form.direction === 'clockwise' ? 'on' : ''} onClick={() => set({ direction: 'clockwise' })}>
                  ↻ Clockwise
                </button>
                <button type="button" className={form.direction === 'anticlockwise' ? 'on' : ''} onClick={() => set({ direction: 'anticlockwise' })}>
                  ↺ Anticlockwise
                </button>
              </div>
            </label>

            {preset?.note && <p className="hint">{preset.note}</p>}
            <p className="hint">The range is cross-checked against the major ticks; the direction cannot be, so check it on the dial.</p>

            <details className="more">
              <summary>Tick count and safe range (optional)</summary>
              <label className="field">
                <span>Number of tick marks, if you counted them</span>
                <input type="number" min="2" step="1" value={form.n_ticks} onChange={e => set({ n_ticks: e.target.value })} />
              </label>
              <div className="row2">
                <label className="field">
                  <span>Safe lower limit</span>
                  <input type="number" step="any" value={form.safe_lo} onChange={e => set({ safe_lo: e.target.value })} />
                </label>
                <label className="field">
                  <span>Safe upper limit</span>
                  <input type="number" step="any" value={form.safe_hi} onChange={e => set({ safe_hi: e.target.value })} />
                </label>
              </div>
            </details>

            {error && <div className="error-box">{error}</div>}

            <button type="submit" className="btn-run" disabled={loading || apiDown}>
              {loading ? <><span className="spinner" /> Reading the dial…</> : 'Read the dial'}
            </button>
          </form>
        </section>

        {/* ── Right column: reading and the pipeline ── */}
        <section className="right-col">
          <div className="card result-card">
            <h2 className="section-title">Reading</h2>

            {!result && !loading && <div className="empty-state">Choose a photo and its scale, then read the dial.</div>}
            {loading && <div className="empty-state"><span className="spinner big" /> Running the pipeline…</div>}

            {result && (
              <>
                <div className="status-row">
                  <span className="status-pill" style={{ background: statusColor + '22', color: statusColor, border: `1px solid ${statusColor}55` }}>
                    {result.status === 'reliable' ? 'Reliable' : result.status === 'unreliable' ? 'Unreliable: check by eye' : 'No reading'}
                  </span>
                  <span className="gauge-tag">{result.gauge_type}</span>
                  <span className="confidence">Confidence {(result.confidence * 100).toFixed(0)}%</span>
                </div>

                {result.ok && (
                  <div className="reading-value">
                    <span className="value-num">{fmt(result.value)}</span>
                    <span className="value-unit">{result.unit}</span>
                    <span className="value-uncertainty">± {fmt(result.uncertainty, 2)}</span>
                  </div>
                )}

                {sample && result.ok && (
                  <div className="truth">
                    True value <strong>{fmt(sample.true_value)} {result.unit}</strong> · error {error_fs.toFixed(2)}% of full scale
                  </div>
                )}

                {result.scale_used && (
                  <div className="scale-line">
                    Scale used: {fmt(result.scale_used.vmin)} to {fmt(result.scale_used.vmax)} {result.scale_used.unit},
                    {' '}{result.scale_used.scale_type}, {result.scale_used.direction} · {result.scale_source}
                  </div>
                )}
                {result.scale_parser && !result.scale_parser.accepted && (
                  <div className="reasons">The AI scale reading was rejected: {result.scale_parser.reason}.</div>
                )}

                {sc?.checked && (
                  <div className={sc.ok ? 'check-ok' : 'check-bad'}>
                    {sc.ok ? '✓ ' : '✗ '}{sc.message}
                    {!sc.ok && sc.suggestions.length > 0 && (
                      <div className="chips">
                        <span className="chips-label">Ranges that fit</span>
                        {sc.suggestions.map(r => (
                          <button type="button" key={r.join()} className="chip" onClick={() => pickRange(r)}>
                            {r[0]} – {r[1]}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {result.range_warning && <div className="warning-banner">⚠ {result.range_warning}</div>}
                {result.in_range === true && <div className="ok-banner">Within the safe range</div>}

                {result.reasons.length > 0 && (
                  <div className="reasons">
                    <strong>Why it is flagged:</strong>
                    <ul>{result.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
                  </div>
                )}

                {result.checks.length > 0 && (
                  <details className="more">
                    <summary>Quality checks ({result.checks.filter(c => c.ok).length} of {result.checks.length} passed)</summary>
                    <ul className="checks">
                      {result.checks.map(c => (
                        <li key={c.name} className={c.ok ? 'pass' : 'fail'}>
                          <span>{c.ok ? '✓' : '✗'}</span> {c.text.replace(/^./, s => s.toUpperCase())}
                          <em>{c.score.toFixed(2)} × threshold</em>
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </>
            )}
          </div>

          {result?.stages?.length > 0 && (
            <div className="card stages-card">
              <h2 className="section-title">How it was read</h2>
              <div className="stage-tabs">
                {result.stages.map((s, i) => (
                  <button key={s.key} className={`stage-tab ${i === step ? 'active' : ''}`} onClick={() => setStep(i)}>
                    {s.title}
                  </button>
                ))}
              </div>
              <div className="stage-img-wrap">
                <img key={stage.key} src={`data:image/png;base64,${stage.image}`} alt={stage.title} className="stage-img" />
              </div>
              <p className="stage-caption"><strong>{stage.title}.</strong> {stage.caption}</p>
              <div className="stage-nav">
                <button className="stage-tab" disabled={step === 0} onClick={() => setStep(step - 1)}>← Previous</button>
                <span>{step + 1} / {result.stages.length}</span>
                <button className="stage-tab" disabled={step === result.stages.length - 1} onClick={() => setStep(step + 1)}>Next →</button>
              </div>
              <div className="timing-row">
                <span className="timing-total">Total {result.timings.total} ms</span>
              </div>
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        DialReader · Image Processing and Computer Vision project · every measurement is classical image processing ·
        not for safety-critical decisions
      </footer>
    </div>
  )
}

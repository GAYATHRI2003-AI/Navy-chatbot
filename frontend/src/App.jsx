import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Shield,
  Target,
  Activity,
  Radio,
  AlertTriangle,
  FileText,
  Map as MapIcon,
  Search,
  Upload,
  Cpu,
  Zap,
  Mic,
  MessageSquare,
  Terminal as TerminalIcon,
  Maximize2,
  ChevronRight,
  Info,
  Clock
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';

const API_BASE = 'http://localhost:8030';

function App() {
  const [query, setQuery] = useState('');
  const [logs, setLogs] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [activeView, setActiveView] = useState('tactical');
  const [identificationResult, setIdentificationResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [events, setEvents] = useState([]);
  const [systemStatus, setSystemStatus] = useState('OFFLINE');
  const [inputLibrary, setInputLibrary] = useState({ audio: [], video: [], images: [] });
  const [activeBase, setActiveBase] = useState(localStorage.getItem('uata_active_base') || '');
  const [bases, setBases] = useState({});

  const scrollRef = useRef(null);

  useEffect(() => {
    fetchMetrics();
    fetchEvents();
    fetchHealth();
    fetchInputLibrary();
    fetchBases();
    const interval = setInterval(() => {
      fetchEvents();
      fetchMetrics();
      fetchHealth();
      fetchInputLibrary();
      fetchBases(true); // Background sync
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchBases = async (isBackground = false) => {
    try {
      const res = await axios.get(`${API_BASE}/list_bases`);
      const basesData = res.data.bases || {};
      const serverActiveBase = res.data.active_base;

      setBases(basesData);

      if (!isBackground) {
        setActiveBase(current => {
          if (current) return current;
          const initial = serverActiveBase || (Object.keys(basesData).length > 0 ? Object.keys(basesData).sort()[0] : '');
          if (initial) localStorage.setItem('uata_active_base', initial);
          return initial;
        });
      }
    } catch (e) {
      console.error("Base fetching error:", e);
    }
  };

  const handleBaseChange = async (e) => {
    const newBase = e.target.value;
    setActiveBase(newBase);
    localStorage.setItem('uata_active_base', newBase);
    try {
      const formData = new FormData();
      formData.append('base_name', newBase);
      await axios.post(`${API_BASE}/set_base`, formData);
      addLog(`Operational Base transferred to: ${newBase}`, 'system');
    } catch (err) {
      addLog('Failed to change operational base.', 'error');
    }
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const fetchInputLibrary = async () => {
    try {
      const res = await axios.get(`${API_BASE}/list_inputs`);
      setInputLibrary(res.data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await axios.get(`${API_BASE}/metrics`);
      setMetrics(res.data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchEvents = async () => {
    try {
      const res = await axios.get(`${API_BASE}/events`);
      setEvents(res.data.events || []);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchHealth = async () => {
    try {
      const res = await axios.get(`${API_BASE}/health`);
      setSystemStatus(res.data.status === 'online' ? 'OPERATIONAL' : 'PARTIAL');
    } catch (e) {
      setSystemStatus('OFFLINE');
    }
  };

  const handleQuery = async (e, forcedQuery = null) => {
    if (e) e.preventDefault();
    const currentQuery = forcedQuery || query;
    if (!currentQuery.trim()) return;

    if (!forcedQuery) setQuery('');
    addLog(`> ${currentQuery}`, 'user');
    setIsProcessing(true);

    try {
      let res;
      if (currentQuery.startsWith('/')) {
        const parts = currentQuery.split(' ');
        const cmd = parts[0];
        const arg = parts.slice(1).join(' ');

        const formData = new FormData();
        formData.append('command', cmd);
        formData.append('argument', arg);
        res = await axios.post(`${API_BASE}/command`, formData, { timeout: 120000 });
        addLog(res.data.response, 'ai');

        if (res.data.pdf_path) {
          const pdfUrl = `${API_BASE}/${res.data.pdf_path}`;
          window.open(pdfUrl, '_blank');
          addLog("Mission SITREP generated and opened in secure tactical viewer.", "system");
        }
      } else {
        const formData = new FormData();
        formData.append('query', currentQuery);
        res = await axios.post(`${API_BASE}/query`, formData, { timeout: 120000 });
        addLog(res.data.response, 'ai');
      }
    } catch (err) {
      addLog(`Error contacting tactical server.`, 'error');
    } finally {
      setIsProcessing(false);
    }
  };

  const addLog = (text, type) => {
    setLogs(prev => [...prev, { text, type, time: new Date().toLocaleTimeString() }]);
  };

  const processStoredFile = async (category, filename) => {
    const fullPath = `sensor_input/${category}/${filename}`;
    setIsProcessing(true);
    addLog(`Processing archived asset: ${filename}...`, 'system');

    try {
      const formData = new FormData();
      formData.append('path', fullPath);
      const res = await axios.post(`${API_BASE}/identify_path`, formData);
      setIdentificationResult(res.data);
      addLog(res.data.response, 'ai');
      setActiveView('tactical');
    } catch (err) {
      addLog(`Failed to process archived asset.`, 'error');
    } finally {
      setIsProcessing(false);
    }
  };

  const onDrop = async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setIsProcessing(true);
    addLog(`Uploading tactical file: ${file.name}...`, 'system');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post(`${API_BASE}/identify`, formData);
      setIdentificationResult(res.data);
      addLog(res.data.response, 'ai');

      if (file.type.startsWith('image/')) {
        setActiveView('tactical');
      }
    } catch (err) {
      addLog(`Failed to process identification request.`, 'error');
    } finally {
      setIsProcessing(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  const commandSamples = [
    { label: 'Emergency SOP', cmd: '/emergency fire', icon: <AlertTriangle size={14} /> },
    { label: 'Maritime Compliance', cmd: '/comply check transit through 9-degree channel', icon: <Shield size={14} /> },
    { label: 'Tactical Recon', cmd: '/tactical analyze central target', icon: <Target size={14} /> },
    { label: 'Mission Report', cmd: '/report', icon: <FileText size={14} /> },
    { label: 'Acoustic Profiler', cmd: '/listen', icon: <Activity size={14} /> }
  ];

  return (
    <div className="grid-layout">
      {/* Header */}
      <header className="glass-panel">
        <div className="logo">
          <Shield size={32} />
          <span>UATA / Command Center</span>
        </div>
        <div style={{ display: 'flex', gap: '2rem' }}>
          <div className="stat-card">
            <span className="stat-label">Model Status</span>
            <span className="stat-value" style={{
              color: systemStatus === 'OPERATIONAL' ? '#00e676' :
                systemStatus === 'PARTIAL' ? '#ffeb3b' : '#ff5252'
            }}>{systemStatus || 'UNKNOWN'}</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">AI Logic</span>
            <span className="stat-value">DeepSeek/CLIP</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Targeting</span>
            <span className="stat-value" style={{ color: 'var(--secondary)' }}>ACTIVE</span>
          </div>
          <div className="stat-card" style={{ minWidth: '150px' }}>
            <span className="stat-label">Command Base</span>
            <select
              value={activeBase}
              onChange={handleBaseChange}
              style={{
                background: 'rgba(0, 255, 136, 0.05)',
                border: '1px solid var(--primary)',
                borderRadius: '4px',
                padding: '4px 8px',
                color: 'var(--primary)',
                fontSize: '0.9rem',
                fontWeight: '800',
                cursor: 'pointer',
                outline: 'none',
                marginTop: '4px'
              }}
            >
              <option value="" disabled>Select Base</option>
              {Object.keys(bases).sort().map(base => (
                <option key={base} value={base} style={{ background: '#050a14', color: '#fff' }}>{base}</option>
              ))}
            </select>
          </div>
        </div>
      </header>

      {/* Sidebar: Navigation & Command Dossier */}
      <aside className="sidebar">
        <div className="glass-panel" style={{ padding: '1.2rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ fontSize: '0.8rem', color: 'var(--secondary)', letterSpacing: '1px' }}>COMMAND DOSSIER</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
            {commandSamples.map((sample, idx) => (
              <button
                key={idx}
                className="command-pill"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '10px',
                  fontSize: '0.75rem',
                  textAlign: 'left',
                  background: 'rgba(255,255,255,0.03)',
                  border: '1px solid rgba(255,255,255,0.05)'
                }}
                onClick={() => handleQuery(null, sample.cmd)}
              >
                <span style={{ color: 'var(--primary)' }}>{sample.icon}</span>
                <span>{sample.label}</span>
                <ChevronRight size={12} style={{ marginLeft: 'auto', opacity: 0.3 }} />
              </button>
            ))}
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '1.2rem', flex: 1, display: 'flex', flexDirection: 'column', gap: '1rem', overflowY: 'auto' }}>
          <h3 style={{ fontSize: '0.8rem', color: 'var(--secondary)', letterSpacing: '1px' }}>STRATEGIC VULT</h3>

          {/* Audio Input Folder */}
          <div className="folder-section">
            <div className="stat-label" style={{ display: 'flex', alignItems: 'center', gap: '5px', marginBottom: '8px' }}>
              <Mic size={14} color="var(--primary)" /> AUDIO FEED
            </div>
            <div className="file-list">
              {inputLibrary.audio.map((f, i) => (
                <div key={i} className="file-item" onClick={() => processStoredFile('audio', f)}>
                  <Activity size={10} /> {f}
                </div>
              ))}
              {inputLibrary.audio.length === 0 && <div className="empty-msg">No audio sensors detected</div>}
            </div>
          </div>

          {/* Video Input Folder */}
          <div className="folder-section">
            <div className="stat-label" style={{ display: 'flex', alignItems: 'center', gap: '5px', marginBottom: '8px' }}>
              <Maximize2 size={14} color="var(--primary)" /> VIDEO RECON
            </div>
            <div className="file-list">
              {inputLibrary.video.map((f, i) => (
                <div key={i} className="file-item" onClick={() => processStoredFile('video', f)}>
                  <Zap size={10} /> {f}
                </div>
              ))}
              {inputLibrary.video.length === 0 && <div className="empty-msg">No active video leads</div>}
            </div>
          </div>

          {/* Image Input Folder */}
          <div className="folder-section">
            <div className="stat-label" style={{ display: 'flex', alignItems: 'center', gap: '5px', marginBottom: '8px' }}>
              <Target size={14} color="var(--primary)" /> SONAR ARCHIVE
            </div>
            <div className="file-list">
              {inputLibrary.images.map((f, i) => (
                <div key={i} className="file-item" onClick={() => processStoredFile('images', f)}>
                  <MapIcon size={10} /> {f}
                </div>
              ))}
            </div>
          </div>

          <div style={{ marginTop: 'auto', padding: '10px', background: 'rgba(0,255,136,0.05)', borderRadius: '8px', border: '1px solid var(--primary)' }}>
            <div style={{ color: 'var(--primary)', fontSize: '0.7rem', fontWeight: 800 }}>INPUTS CLASSIFIED</div>
            <div style={{ fontSize: '0.55rem', opacity: 0.7 }}>Sensors are classified into Tactical Folders automatically.</div>
          </div>
        </div>
      </aside>

      {/* Main Tactical View */}
      <main className="main-view glass-panel">
        <div className="scanline"></div>
        <div style={{ padding: '1.2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '1.2rem' }}>
            <Target className="pulse" style={{ color: '#00ff88' }} size={24} />
            INTELLIGENCE WORKSTATION
          </h2>
          <div style={{ display: 'flex', gap: '8px' }}>
            <button style={{ fontSize: '0.7rem' }} onClick={() => handleQuery(null, "/metrics")}><Cpu size={14} /> Metrics</button>
            <button style={{ fontSize: '0.7rem', borderColor: 'var(--secondary)' }} onClick={() => handleQuery(null, "/report")}><FileText size={14} /> REPORT</button>
          </div>
        </div>

        <div style={{ flex: 1, padding: '1.5rem', display: 'flex', gap: '1.5rem', overflowY: 'auto' }}>
          {identificationResult ? (
            <>
              {/* Prediction Visual */}
              <div style={{ flex: 1.5, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  {/* Original Input */}
                  <div style={{ borderRadius: '12px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)', background: '#000' }}>
                    <div style={{ background: 'rgba(0,0,0,0.5)', padding: '5px', textAlign: 'center', fontSize: '0.6rem', borderBottom: '1px solid rgba(255,255,255,0.1)', color: 'var(--secondary)' }}>ORIGINAL SENSOR INPUT</div>
                    <img
                      src={`${API_BASE}/${identificationResult.file_path}`}
                      alt="Input"
                      style={{ width: '100%', height: '280px', objectFit: 'contain' }}
                      onError={(e) => { e.target.style.display = 'none'; }}
                    />
                  </div>
                  {/* Annotated Output */}
                  <div style={{ position: 'relative', borderRadius: '12px', overflow: 'hidden', border: '2px solid var(--primary)', background: '#000' }}>
                    <div style={{ background: 'var(--primary)', color: '#000', padding: '5px', textAlign: 'center', fontSize: '0.6rem', fontWeight: 800 }}>TACTICAL ANNOTATION</div>
                    <img
                      src={`${API_BASE}/${identificationResult.output_image || identificationResult.file_path}`}
                      alt="Tactical Scan"
                      style={{ width: '100%', height: '280px', objectFit: 'contain' }}
                      onError={(e) => { e.target.src = `${API_BASE}/${identificationResult.file_path}`; }}
                    />
                    <div style={{ position: 'absolute', top: '35px', left: '10px', background: 'rgba(0,255,136,0.9)', color: '#000', padding: '2px 8px', borderRadius: '4px', fontSize: '0.7rem', fontWeight: 800 }}>
                      {identificationResult.predicted_class}
                    </div>
                  </div>
                </div>

                <div className="glass-panel" style={{ padding: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', gap: '1rem' }}>
                    <div style={{ border: '1px solid var(--primary)', padding: '5px 12px', borderRadius: '4px' }}>
                      <div style={{ fontSize: '0.55rem', opacity: 0.5 }}>CLASSIFICATION</div>
                      <div style={{ fontSize: '0.85rem', color: 'var(--primary)', fontWeight: 800 }}>{identificationResult.predicted_class}</div>
                    </div>
                    <div style={{ border: '1px solid var(--secondary)', padding: '5px 12px', borderRadius: '4px' }}>
                      <div style={{ fontSize: '0.55rem', opacity: 0.5 }}>CONFIDENCE</div>
                      <div style={{ fontSize: '0.85rem', color: 'var(--secondary)', fontWeight: 800 }}>{identificationResult.confidence}</div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '1rem', fontSize: '0.7rem', opacity: 0.6 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}><div style={{ width: 8, height: 8, background: 'red' }}></div> Heatmap</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}><div style={{ width: 8, height: 8, border: '1px solid lime' }}></div> Bounding</div>
                  </div>
                </div>

                <div className="glass-panel neon-border" style={{ padding: '1.2rem', background: 'rgba(0,255,136,0.03)', flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '0.8rem', color: 'var(--primary)' }}>
                    <Info size={18} />
                    <h3 style={{ fontSize: '0.85rem', fontWeight: 800 }}>INTEL ADVISORY</h3>
                  </div>
                  <div className="intel-advisory-content" style={{ fontSize: '0.85rem', lineHeight: '1.6', color: '#e0f2f1' }}>
                    {identificationResult.response.split('\n').map((line, i) => (
                      <p key={i} style={{ marginBottom: '6px' }}>{line}</p>
                    ))}
                  </div>
                  <button style={{ marginTop: '1rem', width: '100%', padding: '10px', background: 'transparent', border: '1px solid rgba(255,255,255,0.1)' }} onClick={() => setIdentificationResult(null)}>RESET WORKSTATION</button>
                </div>
              </div>
            </>
          ) : (
            <div {...getRootProps()} style={{
              flex: 1,
              border: '2px dashed rgba(255,255,255,0.15)',
              borderRadius: '24px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              background: isDragActive ? 'rgba(0,255,136,0.03)' : 'transparent',
              borderColor: isDragActive ? 'var(--primary)' : 'rgba(255,255,255,0.15)'
            }}>
              <input {...getInputProps()} />
              <motion.div animate={{ y: isDragActive ? -20 : 0 }}>
                <Upload size={64} style={{ marginBottom: '1.5rem', color: 'var(--primary)', opacity: 0.5 }} />
              </motion.div>
              <h3 style={{ fontSize: '1.4rem', marginBottom: '0.8rem', fontWeight: 800 }}>SENSORY DATA UPLOAD</h3>
              <p style={{ color: 'var(--text-secondary)', maxWidth: '400px', fontSize: '0.9rem', lineHeight: '1.5' }}>
                Deploy tactical scans, sonar hydrophone recordings, or video hydro-recon clips for **YOLO-SONAR** processing.
              </p>
              <div style={{ marginTop: '2rem', display: 'flex', gap: '10px' }}>
                <span className="badge">SONAR</span>
                <span className="badge">Acoustic</span>
                <span className="badge">AIS</span>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Right Sidebar: Intel Gallery */}
      <aside className="right-sidebar glass-panel intel-gallery">
        <div style={{ padding: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Radio size={18} color="var(--secondary)" className="pulse" />
          <h3 style={{ fontSize: '0.8rem', letterSpacing: '1px' }}>INTEL GALLERY</h3>
        </div>
        <div style={{ flex: 1, padding: '1rem', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {events.length > 0 ? events.slice().reverse().map((e, i) => (
            <div key={i} className="intel-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '0.6rem' }}>
                <span style={{ color: 'var(--secondary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Clock size={10} /> {e.time}
                </span>
                <span className="event-tag">VERIFIED</span>
              </div>
              <div style={{ fontSize: '0.75rem', opacity: 0.9, lineHeight: '1.4' }}>{e.event}</div>
            </div>
          )) : (
            <div style={{ padding: '2rem', textAlign: 'center', opacity: 0.3 }}>
              <Search size={32} style={{ marginBottom: '1rem', margin: 'auto' }} />
              <p style={{ fontSize: '0.75rem' }}>Awaiting sensor verification...</p>
            </div>
          )}
        </div>
      </aside>

      {/* Bottom Bar: Terminal */}
      <footer className="bottom-bar glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ flex: 1, padding: '1rem', overflowY: 'auto', backgroundColor: 'rgba(0,0,0,0.4)', borderBottom: '1px solid rgba(255,255,255,0.05)' }} ref={scrollRef}>
          {logs.length === 0 && (
            <div style={{ opacity: 0.4, fontStyle: 'italic', fontSize: '0.8rem', fontFamily: 'monospace' }}>
              [SYSTEM] Tactical Command Node Initialized. Models Loaded. Awaiting Pilot Instruction...
            </div>
          )}
          {logs.map((log, i) => (
            <div key={i} className="log-entry" style={{ border: 'none', padding: '2px 0', fontFamily: 'monospace' }}>
              <span className="log-time">[{log.time}]</span>
              <span style={{
                color: log.type === 'user' ? 'var(--secondary)' :
                  log.type === 'error' ? 'var(--danger)' :
                    log.type === 'system' ? 'var(--primary)' : '#fff',
                fontWeight: log.type === 'ai' ? 'normal' : 'bold'
              }}>
                {log.text}
              </span>
            </div>
          ))}
          {isProcessing && (
            <div className="log-entry" style={{ border: 'none', padding: '2px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="log-time">[{new Date().toLocaleTimeString()}]</span>
              <span style={{ color: 'var(--primary)', fontFamily: 'monospace' }} className="pulse">&gt; Neural Pulse Active... Analysing Metadata...</span>
            </div>
          )}
        </div>
        <form onSubmit={handleQuery} style={{ padding: '0.6rem', display: 'flex', gap: '0.6rem', background: 'rgba(5, 10, 20, 0.9)' }}>
          <div style={{ position: 'relative', flex: 1 }}>
            <TerminalIcon size={18} style={{ position: 'absolute', left: '15px', top: '50%', transform: 'translateY(-50%)', opacity: 0.4 }} />
            <input
              type="text"
              placeholder="Query the DeepSeek Advisor or issue a directive..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              style={{ paddingLeft: '45px', height: '48px', border: '1px solid rgba(255,255,255,0.05)', fontSize: '0.9rem' }}
            />
          </div>
          <button type="submit" style={{ height: '48px', padding: '0 1.8rem', background: 'var(--primary)', color: '#000', fontWeight: 800 }}>
            COMMAND
          </button>
        </form>
      </footer>

      <style dangerouslySetInnerHTML={{
        __html: `
        .pulse { animation: pulse 2s infinite; }
        .badge { background: rgba(0,255,136,0.1); color: var(--primary); padding: 4px 10px; border-radius: 4px; font-size: 0.6rem; font-weight: 800; border: 1px solid rgba(0,255,136,0.2); }
        .intel-card { padding: 12px; background: rgba(255,255,255,0.02); border-left: 2px solid var(--secondary); border-radius: 4px; border-top-right-radius: 8px; border-bottom-right-radius: 8px; transition: background 0.2s; }
        .intel-card:hover { background: rgba(255,255,255,0.04); }
        .event-tag { background: var(--primary); color: #000; padding: 2px 6px; border-radius: 3px; font-weight: 900; }
        .command-pill { transition: all 0.2s ease !important; border-radius: 6px !important; }
        .command-pill:hover { background: rgba(0,255,136,0.1) !important; border-color: var(--primary) !important; transform: translateX(5px); }
        .asset-card { padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px; border-left: 2px solid var(--secondary); }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(0, 255, 136, 0.2); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(0, 255, 136, 0.4); }
      `}} />
    </div>
  );
}

export default App;

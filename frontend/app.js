// Fixed API base pointing to your deployed Heroku backend
const API_BASE = "https://docsense-60db96460d1e.herokuapp.com";

// Elements (may be null depending on the page)
const providerEl = document.getElementById('provider');
const apiKeyEl = document.getElementById('apiKey');
const temperatureEl = document.getElementById('temperature');
const temperatureValEl = document.getElementById('temperatureVal');
const maxTokensEl = document.getElementById('maxTokens');
const maxTokensValEl = document.getElementById('maxTokensVal');
const fileInputEl = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const embedBtn = document.getElementById('embedBtn');
const resetBtn = document.getElementById('resetBtn');
const uploadStatusEl = document.getElementById('uploadStatus');
const urlsEl = document.getElementById('urls');
const urlsBtn = document.getElementById('urlsBtn');
const urlStatusEl = document.getElementById('urlStatus');
const messagesEl = document.getElementById('messages');
const promptEl = document.getElementById('prompt');
const sendBtn = document.getElementById('sendBtn');
const queryStatusEl = document.getElementById('queryStatus');

// Helpers for persisted settings
function getSetting(key, fallback) {
  const v = localStorage.getItem(key);
  return v !== null ? v : fallback;
}
function setSetting(key, value) {
  try { localStorage.setItem(key, String(value)); } catch (_) { /* ignore */ }
}

// Initialize settings from storage if inputs exist
const initialProvider = getSetting('provider', 'groq');
const initialApiKey = getSetting('apiKey', '');
const initialTemperature = parseFloat(getSetting('temperature', '0.7'));
const initialMaxTokens = parseInt(getSetting('maxTokens', '512'), 10);

if (providerEl) {
  providerEl.value = initialProvider;
  providerEl.addEventListener('change', () => setSetting('provider', providerEl.value));
}
if (apiKeyEl) {
  apiKeyEl.value = initialApiKey;
  apiKeyEl.addEventListener('input', () => setSetting('apiKey', apiKeyEl.value));
}
if (temperatureEl) {
  // set default from storage
  temperatureEl.value = String(initialTemperature);
  if (temperatureValEl) temperatureValEl.textContent = String(temperatureEl.value);
  temperatureEl.addEventListener('input', () => {
    if (temperatureValEl) temperatureValEl.textContent = temperatureEl.value;
    setSetting('temperature', temperatureEl.value);
  });
}
if (maxTokensEl) {
  maxTokensEl.value = String(initialMaxTokens);
  if (maxTokensValEl) maxTokensValEl.textContent = String(maxTokensEl.value);
  maxTokensEl.addEventListener('input', () => {
    if (maxTokensValEl) maxTokensValEl.textContent = maxTokensEl.value;
    setSetting('maxTokens', maxTokensEl.value);
  });
}

function addMessage(role, content) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;
  const roleEl = document.createElement('div');
  roleEl.className = 'role';
  roleEl.textContent = role === 'user' ? 'You' : 'Assistant';
  const contentEl = document.createElement('div');
  contentEl.className = 'content';
  contentEl.textContent = content;
  wrap.appendChild(roleEl);
  wrap.appendChild(contentEl);
  if (messagesEl) {
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
}

async function uploadFiles() {
  if (uploadStatusEl) uploadStatusEl.textContent = 'Uploading...';
  try {
    if (!fileInputEl) return;
    const files = fileInputEl.files;
    if (!files || files.length === 0) {
      if (uploadStatusEl) uploadStatusEl.textContent = 'Please select files to upload.';
      return;
    }
    const form = new FormData();
    for (const f of files) form.append('files', f);

    const resp = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: form
    });
    const data = await resp.json();
    if (resp.ok) {
      if (uploadStatusEl) uploadStatusEl.textContent = `Uploaded: ${data.saved.join(', ')}`;
    } else {
      if (uploadStatusEl) uploadStatusEl.textContent = `Upload failed.`;
    }
  } catch (e) {
    if (uploadStatusEl) uploadStatusEl.textContent = `Upload error: ${e}`;
  }
}

async function createEmbeddings() {
  if (uploadStatusEl) uploadStatusEl.textContent = 'Creating embeddings...';
  try {
    const resp = await fetch(`${API_BASE}/embed`, { method: 'POST' });
    const data = await resp.json();
    if (data.status === 'ok') {
      if (uploadStatusEl) uploadStatusEl.textContent = `Embeddings created. Chunks: ${data.chunks}`;
    } else if (data.status === 'cached') {
      if (uploadStatusEl) uploadStatusEl.textContent = data.message;
    } else {
      if (uploadStatusEl) uploadStatusEl.textContent = data.message || 'Embedding failed.';
    }
  } catch (e) {
    if (uploadStatusEl) uploadStatusEl.textContent = `Embedding error: ${e}`;
  }
}

async function ingestUrls() {
  if (urlStatusEl) urlStatusEl.textContent = 'Fetching...';
  try {
    if (!urlsEl) return;
    const lines = (urlsEl.value || '').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
    if (lines.length === 0) {
      if (urlStatusEl) urlStatusEl.textContent = 'Please enter one or more valid URLs.';
      return;
    }
    const resp = await fetch(`${API_BASE}/urls`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ urls: lines })
    });
    const data = await resp.json();
    const failed = (data.failed || []).join(', ');
    if (urlStatusEl) urlStatusEl.textContent = `Embedded: ${data.embedded}. ${failed ? 'Failed: ' + failed : ''}`;
  } catch (e) {
    if (urlStatusEl) urlStatusEl.textContent = `URL ingestion error: ${e}`;
  }
}

async function sendQuery() {
  if (queryStatusEl) queryStatusEl.textContent = '';
  if (!promptEl) return;
  const prompt = (promptEl.value || '').trim();
  if (!prompt) return;
  if (prompt.length > 1000) {
    if (queryStatusEl) queryStatusEl.textContent = 'Query too long. Please keep it under 1000 characters.';
    return;
  }
  const provider = providerEl ? providerEl.value : getSetting('provider', 'groq');
  const apiKey = (apiKeyEl ? apiKeyEl.value : getSetting('apiKey', '')).trim();
  const temperature = parseFloat(temperatureEl ? temperatureEl.value : getSetting('temperature', '0.7'));
  const maxTokens = parseInt(maxTokensEl ? maxTokensEl.value : getSetting('maxTokens', '512'), 10);

  if (!apiKey) {
    if (queryStatusEl) queryStatusEl.textContent = 'Please enter a valid API key.';
    return;
  }

  addMessage('user', prompt);
  promptEl.value = '';

  try {
    const resp = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider, api_key: apiKey, prompt, temperature, max_tokens: maxTokens })
    });
    const data = await resp.json();
    if (data.error) {
      if (queryStatusEl) queryStatusEl.textContent = data.error;
      addMessage('assistant', `Error: ${data.error}`);
      return;
    }
  const answer = data.answer || '(no answer)';
  addMessage('assistant', answer);
    if (queryStatusEl) queryStatusEl.textContent = `Latency: ${data.latency_sec ? data.latency_sec.toFixed(2) : '?'}s`;
  } catch (e) {
    if (queryStatusEl) queryStatusEl.textContent = `Query error: ${e}`;
    addMessage('assistant', `Error: ${e}`);
  }
}

async function resetAll() {
  try {
    await fetch(`${API_BASE}/reset`, { method: 'POST' });
    if (messagesEl) messagesEl.innerHTML = '';
    if (uploadStatusEl) uploadStatusEl.textContent = 'State reset. You can upload and embed again.';
    if (urlStatusEl) urlStatusEl.textContent = '';
    if (queryStatusEl) queryStatusEl.textContent = '';
    if (fileInputEl) fileInputEl.value = '';
    if (urlsEl) urlsEl.value = '';
  } catch (e) {
    if (uploadStatusEl) uploadStatusEl.textContent = `Reset error: ${e}`;
  }
}

// Bind only when elements exist on the page
if (uploadBtn) uploadBtn.addEventListener('click', uploadFiles);
if (embedBtn) embedBtn.addEventListener('click', createEmbeddings);
if (urlsBtn) urlsBtn.addEventListener('click', ingestUrls);
if (sendBtn) sendBtn.addEventListener('click', sendQuery);
if (resetBtn) resetBtn.addEventListener('click', resetAll);

if (promptEl) {
  promptEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuery();
    }
  });
}

// Fixed API base pointing to your deployed Heroku backend
const API_BASE = "https://docsense-60db96460d1e.herokuapp.com";

// Elements
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

// UI bindings
temperatureEl.addEventListener('input', () => {
  temperatureValEl.textContent = temperatureEl.value;
});
maxTokensEl.addEventListener('input', () => {
  maxTokensValEl.textContent = maxTokensEl.value;
});

function addMessage(role, content, contexts) {
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

  if (contexts && contexts.length) {
    const details = document.createElement('div');
    details.className = 'details';
    const title = document.createElement('div');
    title.textContent = `Context (${contexts.length})`;
    details.appendChild(title);
    contexts.forEach((c, i) => {
      const p = document.createElement('p');
      p.textContent = c;
      details.appendChild(p);
    });
    wrap.appendChild(details);
  }

  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function uploadFiles() {
  uploadStatusEl.textContent = 'Uploading...';
  try {
    const files = fileInputEl.files;
    if (!files || files.length === 0) {
      uploadStatusEl.textContent = 'Please select files to upload.';
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
      uploadStatusEl.textContent = `Uploaded: ${data.saved.join(', ')}`;
    } else {
      uploadStatusEl.textContent = `Upload failed.`;
    }
  } catch (e) {
    uploadStatusEl.textContent = `Upload error: ${e}`;
  }
}

async function createEmbeddings() {
  uploadStatusEl.textContent = 'Creating embeddings...';
  try {
    const resp = await fetch(`${API_BASE}/embed`, { method: 'POST' });
    const data = await resp.json();
    if (data.status === 'ok') {
      uploadStatusEl.textContent = `Embeddings created. Chunks: ${data.chunks}`;
    } else if (data.status === 'cached') {
      uploadStatusEl.textContent = data.message;
    } else {
      uploadStatusEl.textContent = data.message || 'Embedding failed.';
    }
  } catch (e) {
    uploadStatusEl.textContent = `Embedding error: ${e}`;
  }
}

async function ingestUrls() {
  urlStatusEl.textContent = 'Fetching...';
  try {
    const lines = (urlsEl.value || '').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
    if (lines.length === 0) {
      urlStatusEl.textContent = 'Please enter one or more valid URLs.';
      return;
    }
    const resp = await fetch(`${API_BASE}/urls`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ urls: lines })
    });
    const data = await resp.json();
    const failed = (data.failed || []).join(', ');
    urlStatusEl.textContent = `Embedded: ${data.embedded}. ${failed ? 'Failed: ' + failed : ''}`;
  } catch (e) {
    urlStatusEl.textContent = `URL ingestion error: ${e}`;
  }
}

async function sendQuery() {
  queryStatusEl.textContent = '';
  const prompt = (promptEl.value || '').trim();
  if (!prompt) return;
  if (prompt.length > 1000) {
    queryStatusEl.textContent = 'Query too long. Please keep it under 1000 characters.';
    return;
  }
  const provider = providerEl.value;
  const apiKey = apiKeyEl.value.trim();
  const temperature = parseFloat(temperatureEl.value);
  const maxTokens = parseInt(maxTokensEl.value, 10);

  if (!apiKey) {
    queryStatusEl.textContent = 'Please enter a valid API key.';
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
      queryStatusEl.textContent = data.error;
      addMessage('assistant', `Error: ${data.error}`);
      return;
    }
    const answer = data.answer || '(no answer)';
    addMessage('assistant', answer, data.contexts || []);
    queryStatusEl.textContent = `Latency: ${data.latency_sec ? data.latency_sec.toFixed(2) : '?'}s`;
  } catch (e) {
    queryStatusEl.textContent = `Query error: ${e}`;
    addMessage('assistant', `Error: ${e}`);
  }
}

async function resetAll() {
  try {
    await fetch(`${API_BASE}/reset`, { method: 'POST' });
    messagesEl.innerHTML = '';
    uploadStatusEl.textContent = 'State reset. You can upload and embed again.';
    urlStatusEl.textContent = '';
    queryStatusEl.textContent = '';
    fileInputEl.value = '';
    urlsEl.value = '';
  } catch (e) {
    uploadStatusEl.textContent = `Reset error: ${e}`;
  }
}

// Bind
uploadBtn.addEventListener('click', uploadFiles);
embedBtn.addEventListener('click', createEmbeddings);
urlsBtn.addEventListener('click', ingestUrls);
sendBtn.addEventListener('click', sendQuery);
resetBtn.addEventListener('click', resetAll);

promptEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
});

/* ============================================================
   Multi-Agent Research Assistant — Frontend Application
   ============================================================ */

(() => {
  'use strict';

  // ── Configuration ──────────────────────────────────────────
  const API_BASE = 'http://localhost:8000';

  // ── State ──────────────────────────────────────────────────
  const state = {
    currentThreadId: null,
    eventSource: null,
    isResearching: false,
    eventCount: 0,
    reportMarkdown: '',
    sessions: [],
  };

  // ── DOM References ─────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const dom = {
    // Sidebar
    sidebar: $('#sidebar'),
    sidebarToggle: $('#sidebarToggle'),
    sidebarOverlay: $('#sidebarOverlay'),
    mobileMenuBtn: $('#mobileMenuBtn'),
    sessionList: $('#sessionList'),
    btnNewResearch: $('#btnNewResearch'),
    apiStatus: $('#apiStatus'),
    usageValue: $('#usageValue'),
    usageFill: $('#usageFill'),

    // Query section
    querySection: $('#querySection'),
    queryInput: $('#queryInput'),
    maxIterations: $('#maxIterations'),
    btnStartResearch: $('#btnStartResearch'),

    // Research section
    researchSection: $('#researchSection'),
    bannerQuery: $('#bannerQuery'),
    btnStopResearch: $('#btnStopResearch'),
    activeQueryBanner: $('#activeQueryBanner'),
    pipelineVisualizer: $('#pipelineVisualizer'),
    activityFeed: $('#activityFeed'),
    feedCount: $('#feedCount'),

    // Report
    reportContainer: $('#reportContainer'),
    reportBody: $('#reportBody'),
    btnCopyReport: $('#btnCopyReport'),
    btnDownloadReport: $('#btnDownloadReport'),

    // HITL Modal
    hitlModal: $('#hitlModal'),
    hitlReason: $('#hitlReason'),
    conflictsSection: $('#conflictsSection'),
    conflictsList: $('#conflictsList'),
    feedbackInput: $('#feedbackInput'),
    btnSubmitFeedback: $('#btnSubmitFeedback'),
    btnSkipFeedback: $('#btnSkipFeedback'),

    // Toast
    toastContainer: $('#toastContainer'),
  };

  // ── Initialization ─────────────────────────────────────────
  function init() {
    bindEvents();
    checkApiHealth();
    loadSessions();
    setInterval(checkApiHealth, 30000);
  }

  // ── Event Bindings ─────────────────────────────────────────
  function bindEvents() {
    // Sidebar toggle
    dom.sidebarToggle.addEventListener('click', () => {
      dom.sidebar.classList.toggle('collapsed');
    });

    // Mobile menu
    dom.mobileMenuBtn.addEventListener('click', () => {
      dom.sidebar.classList.add('mobile-open');
      dom.sidebarOverlay.classList.add('visible');
    });

    dom.sidebarOverlay.addEventListener('click', closeMobileSidebar);

    // New research
    dom.btnNewResearch.addEventListener('click', resetToQueryView);

    // Query input
    dom.queryInput.addEventListener('input', handleQueryInput);
    dom.queryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!dom.btnStartResearch.disabled) startResearch();
      }
    });

    // Auto-resize textarea
    dom.queryInput.addEventListener('input', autoResizeTextarea);

    // Start research
    dom.btnStartResearch.addEventListener('click', startResearch);

    // Stop research
    dom.btnStopResearch.addEventListener('click', stopResearch);

    // Example prompts
    $$('.prompt-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        dom.queryInput.value = chip.dataset.prompt;
        autoResizeTextarea.call(dom.queryInput);
        handleQueryInput();
        dom.queryInput.focus();
      });
    });

    // Report actions
    dom.btnCopyReport.addEventListener('click', copyReport);
    dom.btnDownloadReport.addEventListener('click', downloadReport);

    // HITL Modal
    dom.btnSubmitFeedback.addEventListener('click', submitFeedback);
    dom.btnSkipFeedback.addEventListener('click', () => {
      submitFeedbackToApi('continue without changes');
    });
  }

  // ── Auto-resize Textarea ──────────────────────────────────
  function autoResizeTextarea() {
    const el = dom.queryInput;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 160) + 'px';
  }

  // ── Query Input Handler ───────────────────────────────────
  function handleQueryInput() {
    const hasText = dom.queryInput.value.trim().length > 0;
    dom.btnStartResearch.disabled = !hasText;
  }

  // ── API Health Check ──────────────────────────────────────
  async function checkApiHealth() {
    const statusDot = dom.apiStatus.querySelector('.status-dot');
    const statusText = dom.apiStatus.querySelector('.status-text');

    statusDot.className = 'status-dot checking';
    statusText.textContent = 'Checking...';

    try {
      const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        statusDot.className = 'status-dot online';
        statusText.textContent = 'API Online';
        const data = await res.json().catch(() => null);
        if (data && data.budget_remaining !== undefined) {
          const pct = Math.round((data.budget_remaining / (data.budget_total || 100)) * 100);
          dom.usageValue.textContent = `${pct}% left`;
          dom.usageFill.style.width = `${pct}%`;
        } else {
          dom.usageValue.textContent = 'Active';
          dom.usageFill.style.width = '100%';
        }
      } else {
        throw new Error('Not OK');
      }
    } catch {
      statusDot.className = 'status-dot offline';
      statusText.textContent = 'API Offline';
      dom.usageValue.textContent = '—';
      dom.usageFill.style.width = '0%';
    }
  }

  // ── Sessions ──────────────────────────────────────────────
  async function loadSessions() {
    try {
      const res = await fetch(`${API_BASE}/sessions`);
      if (!res.ok) throw new Error('Failed to load sessions');
      state.sessions = await res.json();
      renderSessions();
    } catch {
      // Silently fail — sessions panel stays empty
    }
  }

  function renderSessions() {
    if (!state.sessions.length) {
      dom.sessionList.innerHTML = `
        <div class="session-empty">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
          </svg>
          <p>No sessions yet</p>
        </div>`;
      return;
    }

    dom.sessionList.innerHTML = state.sessions
      .map((s) => {
        const time = formatRelativeTime(s.timestamp);
        const statusClass = s.status || 'completed';
        const isActive = s.thread_id === state.currentThreadId;
        return `
        <div class="session-item ${isActive ? 'active' : ''}" data-thread-id="${s.thread_id}">
          <div class="session-item-icon">🔬</div>
          <div class="session-item-text">
            <div class="session-item-query" title="${escapeHtml(s.query)}">${escapeHtml(truncate(s.query, 50))}</div>
            <div class="session-item-time">${time}</div>
          </div>
          <div class="session-item-status ${statusClass}"></div>
        </div>`;
      })
      .join('');

    // Bind session click events
    dom.sessionList.querySelectorAll('.session-item').forEach((item) => {
      item.addEventListener('click', () => {
        const threadId = item.dataset.threadId;
        if (threadId !== state.currentThreadId) {
          // Could load historical session — for now just show toast
          showToast('Session replay coming soon!', 'info');
        }
      });
    });
  }

  // ── Start Research ────────────────────────────────────────
  async function startResearch() {
    const query = dom.queryInput.value.trim();
    if (!query) return;

    const maxIterations = parseInt(dom.maxIterations.value, 10);

    dom.btnStartResearch.disabled = true;

    try {
      const res = await fetch(`${API_BASE}/research`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, max_iterations: maxIterations }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: 'Failed to start research' }));
        throw new Error(errData.detail || 'Request failed');
      }

      const data = await res.json();
      state.currentThreadId = data.thread_id;
      state.isResearching = true;
      state.eventCount = 0;
      state.reportMarkdown = '';

      // Switch to research view
      showResearchView(query);

      // Start SSE stream
      connectStream(data.thread_id);

      // Reload sessions
      loadSessions();

      showToast('Research started!', 'success');
    } catch (err) {
      showToast(err.message, 'error');
      dom.btnStartResearch.disabled = false;
    }
  }

  // ── Stop Research ─────────────────────────────────────────
  function stopResearch() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    state.isResearching = false;
    markResearchComplete();
    showToast('Research stopped', 'info');
  }

  // ── View Management ───────────────────────────────────────
  function showResearchView(query) {
    dom.querySection.classList.add('hidden');
    dom.researchSection.classList.remove('hidden');
    dom.bannerQuery.textContent = query;
    dom.activityFeed.innerHTML = '';
    dom.reportContainer.classList.add('hidden');
    dom.reportBody.innerHTML = '';
    dom.activeQueryBanner.classList.remove('banner-complete');
    dom.feedCount.textContent = '0 events';

    // Reset pipeline
    $$('.pipeline-agent').forEach((el) => {
      el.classList.remove('active', 'completed');
    });

    closeMobileSidebar();
  }

  function resetToQueryView() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    state.isResearching = false;
    state.currentThreadId = null;

    dom.querySection.classList.remove('hidden');
    dom.researchSection.classList.add('hidden');
    dom.queryInput.value = '';
    dom.btnStartResearch.disabled = true;
    autoResizeTextarea();
    dom.queryInput.focus();

    closeMobileSidebar();
  }

  function closeMobileSidebar() {
    dom.sidebar.classList.remove('mobile-open');
    dom.sidebarOverlay.classList.remove('visible');
  }

  // ── SSE Stream Connection ─────────────────────────────────
  function connectStream(threadId) {
    if (state.eventSource) {
      state.eventSource.close();
    }

    const url = `${API_BASE}/research/${threadId}/stream`;
    const es = new EventSource(url);
    state.eventSource = es;

    es.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        handleStreamEvent(parsed);
      } catch {
        // Skip malformed events
      }
    };

    es.onerror = () => {
      if (state.isResearching) {
        // EventSource auto-reconnects — but after complete, close it
        // We'll handle completion via the 'complete' event
      }
    };
  }

  // ── Stream Event Handler ──────────────────────────────────
  function handleStreamEvent(event) {
    const { type, data } = event;

    state.eventCount++;
    dom.feedCount.textContent = `${state.eventCount} event${state.eventCount !== 1 ? 's' : ''}`;

    switch (type) {
      case 'status':
        handleStatusEvent(data);
        break;
      case 'search_result':
        handleSearchResultEvent(data);
        break;
      case 'content':
        handleContentEvent(data);
        break;
      case 'analysis':
        handleAnalysisEvent(data);
        break;
      case 'report':
        handleReportEvent(data);
        break;
      case 'human_review':
        handleHumanReviewEvent(data);
        break;
      case 'error':
        handleErrorEvent(data);
        break;
      case 'complete':
        handleCompleteEvent();
        break;
      default:
        addActivityEvent('status', '⚙️', 'System', JSON.stringify(data));
    }

    // Auto-scroll feed
    const feed = dom.activityFeed;
    feed.scrollTop = feed.scrollHeight;
  }

  // ── Event Handlers ────────────────────────────────────────
  function handleStatusEvent(data) {
    const agentName = data.agent || data.node || 'System';
    const message = data.message || data.status || '';

    updatePipeline(agentName);
    addActivityEvent('status', getAgentEmoji(agentName), agentName, message);
  }

  function handleSearchResultEvent(data) {
    const results = data.results || data.search_results || [];
    const count = results.length;

    let detailsHtml = '';
    if (results.length) {
      detailsHtml = '<div class="search-results-chips">';
      results.forEach((r) => {
        detailsHtml += `<a class="result-chip" href="${escapeHtml(r.url)}" target="_blank" rel="noopener" title="${escapeHtml(r.snippet || r.content || '')}">
          <span class="result-chip-icon">🔗</span>
          ${escapeHtml(truncate(r.title || r.url, 40))}
        </a>`;
      });
      detailsHtml += '</div>';
    }

    addActivityEvent('search', '🔍', 'Searcher', `Found ${count} result${count !== 1 ? 's' : ''}`, detailsHtml);
    updatePipeline('searcher');
  }

  function handleContentEvent(data) {
    const pages = data.pages || data.scraped_content || [];
    let detailsHtml = '<div class="event-details">';
    pages.forEach((p) => {
      detailsHtml += `<div class="event-details-item">
        <a href="${escapeHtml(p.url)}" target="_blank" rel="noopener">${escapeHtml(truncate(p.title || p.url, 60))}</a>
      </div>`;
    });
    detailsHtml += '</div>';

    addActivityEvent('content', '📄', 'Extractor', `Extracted content from ${pages.length} page${pages.length !== 1 ? 's' : ''}`, detailsHtml);
  }

  function handleAnalysisEvent(data) {
    // Support both flat format (data.findings) and nested format (data.analysis.key_findings)
    const analysis = data.analysis || {};
    const findings = data.findings || analysis.key_findings || [];
    const conflicts = data.conflicts || analysis.conflicts || [];
    const gaps = data.gaps || analysis.knowledge_gaps || [];

    let detailsHtml = '<div class="analysis-cards">';
    detailsHtml += `<div class="analysis-card">
      <div class="analysis-card-title findings">Findings</div>
      <div class="analysis-card-count">${findings.length}</div>
    </div>`;
    detailsHtml += `<div class="analysis-card">
      <div class="analysis-card-title conflicts">Conflicts</div>
      <div class="analysis-card-count">${conflicts.length}</div>
    </div>`;
    detailsHtml += `<div class="analysis-card">
      <div class="analysis-card-title gaps">Gaps</div>
      <div class="analysis-card-count">${gaps.length}</div>
    </div>`;
    detailsHtml += '</div>';

    addActivityEvent('analysis', '🧪', 'Analyzer', 'Analysis complete', detailsHtml);
    updatePipeline('analyzer');
  }

  function handleReportEvent(data) {
    const content = data.content || data.report || '';
    state.reportMarkdown = content;

    dom.reportContainer.classList.remove('hidden');
    dom.reportBody.innerHTML = renderMarkdown(content);

    addActivityEvent('complete', '📝', 'Writer', 'Research report generated');
    updatePipeline('writer');
  }

  function handleHumanReviewEvent(data) {
    const reason = data.reason || 'The research agents need your input to proceed.';
    const conflicts = data.conflicts || [];

    dom.hitlReason.textContent = reason;

    if (conflicts.length) {
      dom.conflictsSection.classList.remove('hidden');
      dom.conflictsList.innerHTML = conflicts
        .map((c) => `<li>${escapeHtml(typeof c === 'string' ? c : JSON.stringify(c))}</li>`)
        .join('');
    } else {
      dom.conflictsSection.classList.add('hidden');
    }

    dom.feedbackInput.value = '';
    dom.hitlModal.classList.remove('hidden');

    addActivityEvent('review', '👤', 'Reviewer', 'Human review requested');
    updatePipeline('reviewer');
  }

  function handleErrorEvent(data) {
    const message = data.message || 'An unknown error occurred';
    addActivityEvent('error', '❌', 'Error', message);
    showToast(message, 'error');
  }

  function handleCompleteEvent() {
    state.isResearching = false;
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    markResearchComplete();
    addActivityEvent('complete', '✅', 'System', 'Research complete!');
    showToast('Research complete!', 'success');
    loadSessions();
  }

  function markResearchComplete() {
    dom.activeQueryBanner.classList.add('banner-complete');
    dom.btnStopResearch.classList.add('hidden');
  }

  // ── Pipeline Visualizer ───────────────────────────────────
  const agentMap = {
    planner: 'agentPlanner',
    searcher: 'agentSearcher',
    search: 'agentSearcher',
    analyzer: 'agentAnalyzer',
    analysis: 'agentAnalyzer',
    writer: 'agentWriter',
    report: 'agentWriter',
    critic: 'agentCritic',
  };

  function updatePipeline(agentName) {
    const key = agentName.toLowerCase().replace(/[^a-z]/g, '');
    const elementId = agentMap[key];
    if (!elementId) return;

    // Mark all previously active as completed
    $$('.pipeline-agent.active').forEach((el) => {
      el.classList.remove('active');
      el.classList.add('completed');
    });

    // Set current agent as active
    const el = $(`#${elementId}`);
    if (el) {
      el.classList.remove('completed');
      el.classList.add('active');
    }
  }

  // ── Activity Feed ─────────────────────────────────────────
  function addActivityEvent(type, icon, agent, message, detailsHtml = '') {
    const time = new Date().toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });

    const eventEl = document.createElement('div');
    eventEl.className = 'activity-event';
    eventEl.innerHTML = `
      <div class="event-icon ${type}">${icon}</div>
      <div class="event-body">
        <div class="event-header">
          <span class="event-agent">${escapeHtml(agent)}</span>
          <span class="event-time">${time}</span>
        </div>
        <div class="event-message">${escapeHtml(message)}</div>
        ${detailsHtml}
      </div>
    `;

    dom.activityFeed.appendChild(eventEl);
  }

  // ── HITL Feedback ─────────────────────────────────────────
  function submitFeedback() {
    const feedback = dom.feedbackInput.value.trim();
    if (!feedback) {
      showToast('Please enter your feedback', 'error');
      return;
    }
    submitFeedbackToApi(feedback);
  }

  async function submitFeedbackToApi(feedback) {
    if (!state.currentThreadId) return;

    dom.btnSubmitFeedback.disabled = true;
    dom.btnSkipFeedback.disabled = true;

    try {
      const res = await fetch(`${API_BASE}/research/${state.currentThreadId}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ feedback }),
      });

      if (!res.ok) throw new Error('Failed to submit feedback');

      dom.hitlModal.classList.add('hidden');
      addActivityEvent('status', '💬', 'You', `Feedback: "${truncate(feedback, 80)}"`);
      showToast('Feedback submitted — resuming research...', 'success');

      // Reconnect the SSE stream to pick up events from the resumed graph.
      // The old connection may still be alive (heartbeating), but reconnecting
      // ensures we don't miss events if it timed out.
      connectStream(state.currentThreadId);
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      dom.btnSubmitFeedback.disabled = false;
      dom.btnSkipFeedback.disabled = false;
    }
  }

  // ── Report Actions ────────────────────────────────────────
  function copyReport() {
    if (!state.reportMarkdown) return;
    navigator.clipboard
      .writeText(state.reportMarkdown)
      .then(() => showToast('Report copied to clipboard', 'success'))
      .catch(() => showToast('Failed to copy', 'error'));
  }

  function downloadReport() {
    if (!state.reportMarkdown) return;
    const blob = new Blob([state.reportMarkdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research-report-${state.currentThreadId || 'output'}.md`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Report downloaded', 'success');
  }

  // ── Markdown Renderer ─────────────────────────────────────
  function renderMarkdown(md) {
    if (!md) return '';

    let html = md;

    // Normalize line endings
    html = html.replace(/\r\n/g, '\n');

    // Code blocks (``` ... ```)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
      return `<pre><code class="language-${escapeHtml(lang)}">${escapeHtml(code.trim())}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');

    // Headers
    html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Horizontal rules
    html = html.replace(/^---+$/gm, '<hr>');

    // Blockquotes
    html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
    // Merge consecutive blockquotes
    html = html.replace(/<\/blockquote>\n<blockquote>/g, '\n');

    // Bold and italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
    html = html.replace(/_(.+?)_/g, '<em>$1</em>');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // Unordered lists
    html = html.replace(/^(?:[-*+] .+\n?)+/gm, (match) => {
      const items = match
        .trim()
        .split('\n')
        .map((line) => `<li>${line.replace(/^[-*+] /, '')}</li>`)
        .join('');
      return `<ul>${items}</ul>`;
    });

    // Ordered lists
    html = html.replace(/^(?:\d+\. .+\n?)+/gm, (match) => {
      const items = match
        .trim()
        .split('\n')
        .map((line) => `<li>${line.replace(/^\d+\. /, '')}</li>`)
        .join('');
      return `<ol>${items}</ol>`;
    });

    // Paragraphs: wrap lines that aren't already in block elements
    const blockTags = ['<h', '<ul', '<ol', '<pre', '<blockquote', '<hr', '<table'];
    html = html
      .split('\n\n')
      .map((block) => {
        const trimmed = block.trim();
        if (!trimmed) return '';
        const isBlock = blockTags.some((tag) => trimmed.startsWith(tag));
        return isBlock ? trimmed : `<p>${trimmed.replace(/\n/g, '<br>')}</p>`;
      })
      .join('\n');

    return html;
  }

  // ── Toast Notifications ───────────────────────────────────
  function showToast(message, type = 'info') {
    const icons = { success: '✓', error: '✕', info: 'ℹ' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
      <span class="toast-icon">${icons[type] || icons.info}</span>
      <span>${escapeHtml(message)}</span>
    `;
    dom.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.classList.add('leaving');
      toast.addEventListener('animationend', () => toast.remove());
    }, 4000);
  }

  // ── Utility Functions ─────────────────────────────────────
  function escapeHtml(str) {
    if (typeof str !== 'string') return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function truncate(str, len) {
    if (!str) return '';
    return str.length > len ? str.slice(0, len) + '…' : str;
  }

  function getAgentEmoji(name) {
    const map = {
      planner: '📋',
      searcher: '🔍',
      search: '🔍',
      analyzer: '🧪',
      analysis: '🧪',
      writer: '✍️',
      report: '✍️',
      reviewer: '👁️',
      review: '👁️',
      system: '⚙️',
    };
    return map[name.toLowerCase()] || '🤖';
  }

  function formatRelativeTime(timestamp) {
    if (!timestamp) return '';
    const now = Date.now();
    const then = new Date(timestamp).getTime();
    const diff = now - then;

    const seconds = Math.floor(diff / 1000);
    if (seconds < 60) return 'Just now';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d ago`;
    return new Date(timestamp).toLocaleDateString();
  }

  // ── Boot ──────────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

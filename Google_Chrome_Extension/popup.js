// ───────────────────────────────────────────────────────────
// popup.js — UI logic & client-side latency tables
// ───────────────────────────────────────────────────────────

/* helper — pretty-print metrics */
function logTable(response, overallStart) {
  if (!response) return;

  const t_total               = Date.now() - overallStart;
  const t_background_to_popup = Date.now() - response.metrics.sentAt;
  const m                     = response.metrics;

  console.table({
    'T_content_to_background (ms)': (m.t_content_to_background ?? 0).toFixed(2),
    'T_background_to_backend (ms)': m.t_background_to_backend.toFixed(2),
    'T_backend_to_background (ms)': m.t_backend_to_background.toFixed(2),
    'T_background_to_popup (ms)':   t_background_to_popup.toFixed(2),
    'T_total (ms)':                 t_total.toFixed(2),
  });

  return t_total;
}

/* ───────────────────────── Truthfulness button ─────────────────────────── */
document.getElementById('checkText').addEventListener('click', () => {
  const text = document.getElementById('textInput').value.trim();
  if (!text) {
    document.getElementById('result').textContent = 'Please enter some text.';
    return;
  }

  const overallStart = Date.now();

  chrome.runtime.sendMessage({ action: 'checkUserText', text }, response => {
    if (chrome.runtime.lastError || !response) {
      document.getElementById('result').textContent = 'Error.';
      console.error('Truth error:', chrome.runtime.lastError?.message);
      return;
    }

    const t_total = logTable(response, overallStart);
    document.getElementById('result').textContent =
      `Result: ${response.result}  ⏱ ${t_total.toFixed(1)} ms`;
  });
});

/* ───────────────────────── URL-only button ─────────────────────────────── */
document.getElementById('checkURLOnly').addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
    const { url } = tabs[0];

    if (/^(chrome|chrome-extension|about|edge):\/\//.test(url)) {
      document.getElementById('result').textContent = 'Cannot scan Chrome internal pages.';
      return;
    }

    const overallStart = Date.now();

    chrome.runtime.sendMessage({ action: 'checkUserURLOnly', url }, response => {
      if (chrome.runtime.lastError || !response) {
        document.getElementById('result').textContent = 'Error analyzing URL.';
        console.error('URL-only error:', chrome.runtime.lastError?.message);
        return;
      }

      const t_total = logTable(response, overallStart);
      document.getElementById('result').textContent =
        `URL Result: ${response.result}  ⏱ ${t_total.toFixed(1)} ms`;
    });
  });
});

/* ───────────────────────── URL + HTML button ───────────────────────────── */
document.getElementById('checkURLAndHTML').addEventListener('click', () => {
  chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
    const { id: tabId, url } = tabs[0];

    if (/^(chrome|chrome-extension|about|edge):\/\//.test(url)) {
      document.getElementById('result').textContent = 'Cannot scan Chrome internal pages.';
      return;
    }

    const overallStart = Date.now();          // master clock

    /* send message (with fallback injection) */
    const pingContent = () => chrome.tabs.sendMessage(
      tabId,
      { action: 'checkURLAndHTML', overallStart },
      response => {
        if (chrome.runtime.lastError || !response) {
          console.warn('Content script not present:', chrome.runtime.lastError?.message);
          injectThenRetry();
          return;
        }
        const t_total = logTable(response, overallStart);
        document.getElementById('result').textContent =
          `Result: ${response.result}  ⏱ ${t_total.toFixed(1)} ms`;
      }
    );

    const injectThenRetry = () => {
      chrome.scripting.executeScript(
        { target: { tabId }, files: ['content.js'] },
        () => {
          if (chrome.runtime.lastError) {
            console.error('Script injection failed:', chrome.runtime.lastError.message);
            document.getElementById('result').textContent = 'Error: Unable to inject content script.';
            return;
          }
          setTimeout(pingContent, 300); // give listener time to register
        }
      );
    };

    pingContent();
  });
});

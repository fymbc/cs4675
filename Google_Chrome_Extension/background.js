chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === "checkUserText") {
    const inputText = request.text;
    const fetchStart = Date.now();
    // Send the truth text to your LLM model backend (API call)
    fetch('http://localhost:8000/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ claim: inputText })  // claim to backend
    })
    .then(response => response.json())
    .then(data => {
      const fetchEnd            = Date.now();
      const t_bg_to_backend     = fetchEnd - fetchStart;
      const t_backend_to_bg     = t_bg_to_backend;

      sendResponse({
        result: data.ensemble_result,
        metrics: {
          t_content_to_background: 0,               // no content script
          t_background_to_backend: t_bg_to_backend,
          t_backend_to_background: t_backend_to_bg,
          sentAt: Date.now(),                       // → popup
       },
      });
    })
    .catch(error => {
      console.error('Error:', error);
      sendResponse({ result: "Error occurred. Please try again." });
    });

    return true; 
  }
  if (request.action === 'checkUserURLOnly') {
    const { url }   = request;
    const fetchStart = Date.now();

    fetch('http://localhost:8000/analyze-url', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ url })
    })
      .then(res => res.json())
      .then(data => {
        const fetchEnd            = Date.now();
        const t_bg_to_backend     = fetchEnd - fetchStart;
        const t_backend_to_bg     = t_bg_to_backend;

        sendResponse({
          url   : data.url,
          result: data.ensemble_result,
          per_model_raw: data.per_model_raw,
          metrics: {
            t_content_to_background: 0,                // no content script
            t_background_to_backend: t_bg_to_backend,
            t_backend_to_background: t_backend_to_bg,
            sentAt: Date.now()
          }
        });
      })
      .catch(err => {
        console.error('URL-only error:', err);
        sendResponse({ error: 'Error occurred. Please try again.' });
      });

    return true;
  }
  if (request.action === 'checkUserURL') {
    const t_received_in_bg   = Date.now();
    const t_content_to_bg    = t_received_in_bg - request.extractionDone;

    const fetchStart = Date.now();
    fetch('http://localhost:8000/analyze-url-html', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ url: request.url, html: request.html })
    })
    .then(res => res.json())
    .then(data => {
      const fetchEnd = Date.now();
      const t_bg_to_backend   = fetchEnd - fetchStart;   // RTT
      const t_backend_to_bg   = t_bg_to_backend;         // cannot split ⇡
      const sentAt            = Date.now();              // for popup calc

      sendResponse({
        url:            data.url,
        result:         data.ensemble_result,
        per_model_raw:  data.per_model_raw,
        metrics: {
          // NB lower-case keys to avoid clash with chrome internals
          t_content_to_background: t_content_to_bg,
          t_background_to_backend: t_bg_to_backend,
          t_backend_to_background: t_backend_to_bg,
          sentAt                      // timestamp when reply leaves BG
        }
      });
    })
    .catch(err => {
      console.error('Error:', err);
      sendResponse({ error: 'Error occurred. Please try again.' });
    });

    return true; // async
  }
});


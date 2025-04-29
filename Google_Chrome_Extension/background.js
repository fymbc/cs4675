chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === "checkUserText") {
    const inputText = request.text;
    
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
      // KEY IS "result" and is "TRUE" or "FALSE"
      sendResponse({ result: data.ensemble_result });
    })
    .catch(error => {
      console.error('Error:', error);
      sendResponse({ result: "Error occurred. Please try again." });
    });

    return true;  // To indicate asynchronous response
  }
  if (request.action === "checkUserURLOnly") {
    // Send just the URL to the backend
    fetch("http://localhost:8000/analyze-url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: request.url })
    })
    .then(res => res.json())
    .then(data => {
      sendResponse({
        url: data.url,
        result: data.ensemble_result,
        per_model_raw: data.per_model_raw
      });
    })
    .catch(err => {
      console.error("Error:", err);
      sendResponse({ error: "Error occurred. Please try again." });
    });
  
    return true; // Async response
  }
  if (request.action === "checkUserURL") {
    fetch("http://localhost:8000/analyze-url-html", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: request.url, html: request.html })
    })
    .then(res => res.json())
    .then(data => {
      // data has { url, ensemble_result, per_model_raw }
      sendResponse({
        url: data.url,
        result: data.ensemble_result,
        per_model_raw: data.per_model_raw
      });
    })
    .catch(err => {
      console.error("Error:", err);
      sendResponse({ error: "Error occurred. Please try again." });
    });

    // tell Chrome we’ll call sendResponse asynchronously
    return true;
  }
});


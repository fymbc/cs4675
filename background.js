chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === "checkUserText") {
    const inputText = request.text;
    
    // Send the truth text to your LLM model backend (API call)
    fetch('http://your-api-endpoint.com/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ claim: inputText })  // claim to backend
    })
    .then(response => response.json())
    .then(data => {
      // KEY IS "result" and is "TRUE" or "FALSE"
      sendResponse({ result: data.result });
    })
    .catch(error => {
      console.error('Error:', error);
      sendResponse({ result: "Error occurred. Please try again." });
    });

    return true;  // To indicate asynchronous response
  }
  
  if (request.action === "checkUserURL") {
    const url = request.url;
    const htmlContent = request.htmlContent;

    // Send the URLHTML content to LLM model backend (API call)
    fetch('http://your-api-endpoint.com/analyze-url-html', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url: url, html: htmlContent })  // Sending both URL and HTML
    })
    .then(response => response.json())
    .then(data => {
      // Assuming the response contains "result" key with "PHISHING" or "LEGITIMATE"
      sendResponse({ result: data.result });
    })
    .catch(error => {
      console.error('Error:', error);
      sendResponse({ result: "Error occurred. Please try again." });
    });

    return true;  // To indicate asynchronous response
  }
});


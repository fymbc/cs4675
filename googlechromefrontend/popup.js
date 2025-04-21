document.getElementById('checkText').addEventListener('click', function() {
  const text = document.getElementById('textInput').value.trim();

  if (text === "") {
    document.getElementById('result').textContent = "Please enter some text.";
    return;
  }

  // Send the entered text to the background script for analysis
  chrome.runtime.sendMessage({ action: 'checkUserText', text: text }, function(response) {
    // Show the result of the truthfulness detection
    document.getElementById('result').textContent = `Result: ${response.result}`;
  });
});

document.getElementById('checkURLOnly').addEventListener('click', function () {
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    const tab = tabs[0];
    const url = tab.url;

    // Skip protected pages
    if (url.startsWith("chrome://") || url.startsWith("chrome-extension://") || url.startsWith("about:") || url.startsWith("edge://")) {
      document.getElementById('result').textContent = "Cannot scan Chrome internal pages.";
      return;
    }

    // Send only the URL to the background script
    chrome.runtime.sendMessage({ action: 'checkUserURLOnly', url: url }, function (response) {
      if (chrome.runtime.lastError || !response) {
        document.getElementById('result').textContent = "Error analyzing URL.";
        console.error("URL-only error:", chrome.runtime.lastError?.message);
        return;
      }

      console.log("✅ URL-only analysis result:", response);
      document.getElementById('result').textContent = `URL Result: ${response.result}`;
    });
  });
});

document.getElementById('checkURLAndHTML').addEventListener('click', function () {
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    const tab = tabs[0];
    const tabId = tab.id;

    // Don't try to inject into restricted Chrome pages
    if (tab.url.startsWith("chrome://") || tab.url.startsWith("chrome-extension://") || tab.url.startsWith("about:") || tab.url.startsWith("edge://")) {
      document.getElementById('result').textContent = "Cannot scan Chrome internal pages.";
      return;
    }

    // Attempt to send message to content script
    chrome.tabs.sendMessage(tabId, { action: "checkURLAndHTML" }, function (response) {
      if (chrome.runtime.lastError) {
        console.warn("Content script not injected, trying to inject manually...");

        // Try injecting it manually using scripting API
        chrome.scripting.executeScript({
          target: { tabId: tabId },
          files: ['content.js']
        }, () => {
          if (chrome.runtime.lastError) {
            console.error("Script injection failed:", chrome.runtime.lastError.message);
            document.getElementById('result').textContent = "Error: Unable to inject content script.";
            return;
          }

          // wait before retrying message (give content.js time to initialize)
          setTimeout(() => {
            console.log("⏳ Retrying message after injection...");
            chrome.tabs.sendMessage(tabId, { action: "checkURLAndHTML" }, function (response2) {
              if (chrome.runtime.lastError || !response2) {
                console.error("❌ Message failed after injection:", chrome.runtime.lastError?.message);
                document.getElementById('result').textContent = "Error: Could not reach content script.";
                return;
              }
          
              console.log("✅ Got response from injected content script", response2);
              document.getElementById('result').textContent = `Result: ${response2.result}`;
            });
          }, 300); // Delay for listener registration
        });
      } else {
        // Content script already present and responded
        document.getElementById('result').textContent = `Result: ${response.result}`;
      }
    });
  });
});


  
  
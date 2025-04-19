// content.js
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === "checkURL") {
      let url = window.location.href;
      // Send the URL to the background script or backend
      chrome.runtime.sendMessage({action: "analyzeURL", url: url}, function(response) {
        console.log("Analysis Result: ", response.result);
      });
    }
  });  
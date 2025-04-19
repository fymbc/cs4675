// background.js
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === "checkUserURL") {
      // Randomly decide whether the URL is phishing or not
      let isPhishing = Math.random() < 0.5;  // 50% chance for phishing or legitimate
      let result = isPhishing ? "Phishing" : "Legitimate";
      
      sendResponse({ result: result });
    }
  
    if (request.action === "userFeedback") {
      // Log user feedback (this could be used for future updates of the model)
      console.log("User feedback:", request.feedback);
  
      // Optionally, process feedback, update model or store it
    }
  
    return true;  // Indicate that the response will be sent asynchronously
  });
  
// popup.js
document.getElementById('checkLink').addEventListener('click', function() {
    const url = document.getElementById('urlInput').value.trim();
  
    if (url === "") {
      document.getElementById('result').textContent = "Please enter a valid URL.";
      return;
    }
  
    // Send the entered URL to the background script for analysis
    chrome.runtime.sendMessage({ action: 'checkUserURL', url: url }, function(response) {
      // Show the result of phishing detection
      document.getElementById('result').textContent = `Result: ${response.result}`;
  
      // Show the feedback options
      document.getElementById('feedback').style.display = 'block';
    });
  });
  
  // Handle feedback responses
  document.getElementById('feedbackYes').addEventListener('click', function() {
    // Show thank you message and hide feedback
    document.getElementById('thankYouMessage').style.display = 'block';
    document.getElementById('feedback').style.display = 'none';
    
    // Optionally, send feedback to the background script (this could be logged, stored, etc.)
    chrome.runtime.sendMessage({ action: 'userFeedback', feedback: 'yes' });
  });
  
  document.getElementById('feedbackNo').addEventListener('click', function() {
    // Show sorry message and hide feedback
    document.getElementById('sorryMessage').style.display = 'block';
    document.getElementById('feedback').style.display = 'none';
    
    // Optionally, send feedback to the background script (this could be logged, stored, etc.)
    chrome.runtime.sendMessage({ action: 'userFeedback', feedback: 'no' });
  });
  
  
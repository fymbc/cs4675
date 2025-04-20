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

document.getElementById('checkURLAndHTML').addEventListener('click', function() {
  // Send a message to the content script to analyze the URL and HTML content
  chrome.runtime.sendMessage({ action: "checkURLAndHTML" }, function(response) {
      // The result will be handled by the content script
      document.getElementById('result').textContent = `Result: ${response.result}`;
  });
});
  
  
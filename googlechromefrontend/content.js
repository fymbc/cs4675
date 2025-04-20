chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === "checkURLAndHTML") {
    let url = window.location.href;
    let htmlContent = document.documentElement.outerHTML;  // Extract entire HTML content of the page

    // Send the URL and HTML content to the background script for analysis
    chrome.runtime.sendMessage({action: "checkUserURL", url: url, htmlContent: htmlContent}, function(response) {
      console.log("URL + HTML Analysis Result: ", response.result);
    });
  }
});

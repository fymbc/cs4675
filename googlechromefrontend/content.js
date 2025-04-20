console.log("✅ content.js loaded and waiting for message");

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  if (request.action === "checkURLAndHTML") {
    console.log("📥 Received checkURLAndHTML message");

    const url = window.location.href;
    const html = document.documentElement.outerHTML;

    // Send data to background.js for analysis
    chrome.runtime.sendMessage({
      action: "checkUserURL",
      url: url,
      html: html
    }, function (response) {
      console.log("📤 Sent to background, received response:", response);
      sendResponse(response); // ✅ SEND BACK TO POPUP
    });

    return true; // ✅ Tell Chrome this will be async
  }
});

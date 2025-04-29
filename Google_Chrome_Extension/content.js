// ───────────────────────────────────────────────────────────
// content.js — runs in the web page, extracts URL + HTML
// ───────────────────────────────────────────────────────────
console.log('✅ content.js loaded');

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action !== 'checkURLAndHTML') return;

  console.log('📥 Received checkURLAndHTML');

  const url             = window.location.href;
  const html            = document.documentElement.outerHTML;
  const extractionDone  = Date.now();                  // finished gathering

  chrome.runtime.sendMessage(
    {
      action        : 'checkUserURL',
      url,
      html,
      // propagate timings
      overallStart  : request.overallStart,
      extractionDone
    },
    response => sendResponse(response)                 // bubble back to popup
  );

  return true; // async
});

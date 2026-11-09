chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action !== "extract-comments") {
    return;
  }

  const comments = Array.from(
    document.querySelectorAll("#comments #content-text")
  )
    .filter(element => element.getClientRects().length > 0)
    .map(element => element.innerText.trim())
    .filter(text => text.length > 0 && text.length <= 20000)
    .slice(0, 50);

  sendResponse({ comments });
});
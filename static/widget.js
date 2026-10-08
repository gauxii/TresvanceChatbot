/**
 * Tresvance Assistant — embeddable chat widget.
 *
 * Usage (drop into any page, e.g. tresvance.com):
 *   <script src="https://YOUR-DEPLOY-HOST/static/widget.js" data-api-base="https://YOUR-DEPLOY-HOST"></script>
 *
 * data-api-base is optional — if omitted, the widget calls back to the host
 * this script was loaded from.
 */
(function () {
  var scriptEl = document.currentScript;
  var apiBase = (scriptEl && scriptEl.getAttribute("data-api-base")) ||
    (scriptEl ? new URL(scriptEl.src, window.location.href).origin : "");

  var SESSION_KEY = "tv_widget_session_id";
  var sessionId = localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = (window.crypto && crypto.randomUUID)
      ? crypto.randomUUID()
      : "sess-" + Date.now() + "-" + Math.random().toString(36).slice(2);
    localStorage.setItem(SESSION_KEY, sessionId);
  }

  var STYLE = [
    ".tv-widget-btn{position:fixed;bottom:20px;right:20px;width:56px;height:56px;",
    "border-radius:50%;background:#4f6df5;color:#fff;border:none;cursor:pointer;",
    "box-shadow:0 4px 14px rgba(0,0,0,.25);z-index:2147483000;display:flex;",
    "align-items:center;justify-content:center;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;}",
    ".tv-widget-btn svg{width:26px;height:26px;fill:#fff;}",
    ".tv-widget-panel{position:fixed;bottom:88px;right:20px;width:360px;max-width:92vw;",
    "height:520px;max-height:76vh;background:#0f1115;color:#e8e9ed;border-radius:16px;",
    "box-shadow:0 10px 40px rgba(0,0,0,.35);display:none;flex-direction:column;overflow:hidden;",
    "z-index:2147483000;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;}",
    ".tv-widget-panel.tv-open{display:flex;}",
    ".tv-widget-header{padding:14px 16px;background:#171a21;border-bottom:1px solid #262a33;",
    "display:flex;align-items:center;justify-content:space-between;}",
    ".tv-widget-header h1{font-size:14px;margin:0;font-weight:600;}",
    ".tv-widget-header p{font-size:11px;color:#9aa0ab;margin:2px 0 0;}",
    ".tv-widget-close{background:none;border:none;color:#9aa0ab;cursor:pointer;font-size:18px;",
    "line-height:1;padding:4px;}",
    ".tv-widget-messages{flex:1;overflow-y:auto;padding:14px;display:flex;",
    "flex-direction:column;gap:9px;}",
    ".tv-widget-msg{max-width:82%;padding:9px 12px;border-radius:13px;font-size:13px;",
    "line-height:1.4;white-space:pre-wrap;}",
    ".tv-widget-msg.tv-bot{align-self:flex-start;background:#1f232c;border-bottom-left-radius:4px;}",
    ".tv-widget-msg.tv-user{align-self:flex-end;background:#4f6df5;color:#fff;border-bottom-right-radius:4px;}",
    ".tv-widget-typing{align-self:flex-start;color:#9aa0ab;font-size:12px;padding:0 12px;}",
    ".tv-widget-chip{align-self:flex-start;background:none;border:1px solid #4f6df5;",
    "color:#9fb0ff;padding:7px 12px;border-radius:999px;font-size:12px;cursor:pointer;",
    "margin-top:2px;}",
    ".tv-widget-chip:hover{background:rgba(79,109,245,.12);}",
    ".tv-widget-form{display:flex;gap:7px;padding:10px;background:#171a21;border-top:1px solid #262a33;}",
    ".tv-widget-input{flex:1;background:#1f232c;border:1px solid #2c313c;color:#e8e9ed;",
    "border-radius:9px;padding:9px 11px;font-size:13px;outline:none;min-width:0;}",
    ".tv-widget-input:focus{border-color:#4f6df5;}",
    ".tv-widget-send{background:#4f6df5;border:none;color:#fff;padding:0 15px;",
    "border-radius:9px;font-size:13px;cursor:pointer;}",
    ".tv-widget-send:disabled{opacity:.5;cursor:default;}",
  ].join("");

  var styleTag = document.createElement("style");
  styleTag.textContent = STYLE;
  document.head.appendChild(styleTag);

  var btn = document.createElement("button");
  btn.className = "tv-widget-btn";
  btn.setAttribute("aria-label", "Open Tresvance Assistant chat");
  btn.innerHTML =
    '<svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.03 2 11c0 2.38 1.19 4.52 3.14 6.11-.1.92-.42 2.15-1.14 3.49-.1.19.02.42.23.44 1.9.17 3.46-.5 4.46-1.13.74.17 1.52.26 2.31.26 5.52 0 10-4.03 10-9S17.52 2 12 2z"/></svg>';

  var panel = document.createElement("div");
  panel.className = "tv-widget-panel";
  panel.innerHTML =
    '<div class="tv-widget-header">' +
    '<div><h1>Tresvance Assistant</h1><p>Services, projects &amp; contact info</p></div>' +
    '<button class="tv-widget-close" aria-label="Close chat" type="button">&times;</button>' +
    "</div>" +
    '<div class="tv-widget-messages"></div>' +
    '<form class="tv-widget-form">' +
    '<input class="tv-widget-input" type="text" placeholder="Type your question…" autocomplete="off" />' +
    '<button class="tv-widget-send" type="submit">Send</button>' +
    "</form>";

  document.body.appendChild(btn);
  document.body.appendChild(panel);

  var messagesEl = panel.querySelector(".tv-widget-messages");
  var form = panel.querySelector(".tv-widget-form");
  var input = panel.querySelector(".tv-widget-input");
  var sendBtn = panel.querySelector(".tv-widget-send");
  var closeBtn = panel.querySelector(".tv-widget-close");

  var greeted = false;

  function addMessage(text, who) {
    var div = document.createElement("div");
    div.className = "tv-widget-msg tv-" + who;
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  function openPanel() {
    panel.classList.add("tv-open");
    if (!greeted) {
      addMessage(
        "Hi! I'm Tresvance's assistant.",
        "bot"
      );
      addLeadChip();
      greeted = true;
    }
    input.focus();
  }

  function closePanel() {
    panel.classList.remove("tv-open");
  }

  btn.addEventListener("click", function () {
    if (panel.classList.contains("tv-open")) {
      closePanel();
    } else {
      openPanel();
    }
  });
  closeBtn.addEventListener("click", closePanel);

  function sendMessage(text) {
    if (!text) return;
    addMessage(text, "user");
    sendBtn.disabled = true;

    var typingEl = document.createElement("div");
    typingEl.className = "tv-widget-typing";
    typingEl.textContent = "Tresvance assistant is typing…";
    messagesEl.appendChild(typingEl);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    fetch(apiBase + "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    })
      .then(function (res) {
        return res.json();
      })
      .then(function (data) {
        typingEl.remove();
        addMessage(data.answer, "bot");
      })
      .catch(function () {
        typingEl.remove();
        addMessage("Sorry, something went wrong reaching the assistant.", "bot");
      })
      .finally(function () {
        sendBtn.disabled = false;
        input.focus();
      });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var text = input.value.trim();
    if (!text) return;
    input.value = "";
    sendMessage(text);
  });

  function addLeadChip() {
    var chip = document.createElement("button");
    chip.type = "button";
    chip.className = "tv-widget-chip";
    chip.textContent = "💬 Talk to our team about a project";
    chip.addEventListener("click", function () {
      chip.remove();
      sendMessage("I'd like to talk to your team about a project.");
    });
    messagesEl.appendChild(chip);
  }
})();

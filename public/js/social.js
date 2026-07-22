(function () {
  "use strict";

  function getClientId() {
    var key = "blog_client_id";
    var id = localStorage.getItem(key);
    if (!id) {
      id = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2))
        .replace(/[^A-Za-z0-9_-]/g, "");
      localStorage.setItem(key, id);
    }
    return id;
  }

  var HEART_OUTLINE = "♡"; // ♡
  var HEART_FILLED = "♥"; // ♥

  function initLikeButton(root) {
    var postKey = root.dataset.postKey;
    var apiBase = root.dataset.apiBase;
    var button = root.querySelector("#like-button");
    var countEl = root.querySelector("#like-count");
    var iconEl = root.querySelector("#like-icon");
    if (!postKey || !apiBase || !button || !countEl) return;

    var clientId = getClientId();
    var cacheKey = "liked:" + postKey;

    function markLiked() {
      button.disabled = true;
      button.classList.add("liked");
      if (iconEl) iconEl.textContent = HEART_FILLED;
      localStorage.setItem(cacheKey, "1");
    }

    if (localStorage.getItem(cacheKey) === "1") {
      markLiked();
    }

    fetch(apiBase + "/api/likes?post=" + encodeURIComponent(postKey) + "&client_id=" + encodeURIComponent(clientId))
      .then(function (res) { return res.json(); })
      .then(function (data) {
        countEl.textContent = String(data.count || 0);
        if (data.liked) markLiked();
      })
      .catch(function () { /* backend unreachable — leave default state */ });

    button.addEventListener("click", function () {
      if (button.disabled) return;

      // Optimistic update: reflect the like immediately, let the server
      // request settle in the background — no need to wait on it.
      var optimisticCount = (parseInt(countEl.textContent, 10) || 0) + 1;
      countEl.textContent = String(optimisticCount);
      markLiked();

      fetch(apiBase + "/api/likes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ post: postKey, client_id: clientId }),
      })
        .then(function (res) { return res.json(); })
        .then(function (data) {
          // Reconcile with the authoritative server count once it arrives.
          if (typeof data.count === "number") countEl.textContent = String(data.count);
        })
        .catch(function () { /* server will catch up eventually; keep the optimistic state */ });
    });
  }

  function initShareButton(root) {
    var button = root.querySelector("#share-button");
    var confirm = root.querySelector("#share-confirm");
    if (!button) return;
    button.addEventListener("click", function () {
      navigator.clipboard.writeText(window.location.href).then(function () {
        if (!confirm) return;
        confirm.hidden = false;
        setTimeout(function () { confirm.hidden = true; }, 2000);
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var root = document.getElementById("social-actions");
    if (!root) return;
    initLikeButton(root);
    initShareButton(root);
  });
})();

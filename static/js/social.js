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

  function initLikeButton(root) {
    var postKey = root.dataset.postKey;
    var apiBase = root.dataset.apiBase;
    var button = root.querySelector("#like-button");
    var countEl = root.querySelector("#like-count");
    if (!postKey || !apiBase || !button || !countEl) return;

    var clientId = getClientId();
    var cacheKey = "liked:" + postKey;

    function setLiked(count) {
      countEl.textContent = String(count);
      button.disabled = true;
      button.classList.add("liked");
      localStorage.setItem(cacheKey, "1");
    }

    if (localStorage.getItem(cacheKey) === "1") {
      button.disabled = true;
      button.classList.add("liked");
    }

    fetch(apiBase + "/api/likes?post=" + encodeURIComponent(postKey) + "&client_id=" + encodeURIComponent(clientId))
      .then(function (res) { return res.json(); })
      .then(function (data) {
        countEl.textContent = String(data.count || 0);
        if (data.liked) setLiked(data.count);
      })
      .catch(function () { /* backend unreachable — leave default state */ });

    button.addEventListener("click", function () {
      if (button.disabled) return;
      button.disabled = true;
      fetch(apiBase + "/api/likes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ post: postKey, client_id: clientId }),
      })
        .then(function (res) { return res.json(); })
        .then(function (data) {
          if (typeof data.count === "number") setLiked(data.count);
          else button.disabled = false;
        })
        .catch(function () { button.disabled = false; });
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

(function () {
  "use strict";

  var indexData = null;
  var indexPromise = null;

  function loadIndex() {
    if (!indexPromise) {
      indexPromise = fetch("/index.json")
        .then(function (res) { return res.json(); })
        .then(function (data) { indexData = data; return data; })
        .catch(function () { return []; });
    }
    return indexPromise;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function matches(post, query) {
    if (post.title && post.title.toLowerCase().includes(query)) return true;
    if (post.description && post.description.toLowerCase().includes(query)) return true;
    return (post.tags || []).some(function (t) { return t.toLowerCase().includes(query); });
  }

  function renderResults(results, query) {
    var container = document.getElementById("searchResults");
    container.innerHTML = "";
    if (!query) return;
    if (results.length === 0) {
      var empty = document.createElement("div");
      empty.className = "nav-search-empty";
      empty.textContent = "No matches.";
      container.appendChild(empty);
      return;
    }
    results.slice(0, 8).forEach(function (post) {
      var a = document.createElement("a");
      a.className = "nav-search-result";
      a.href = post.url;
      var tags = (post.tags || []).join(", ");
      a.innerHTML =
        '<div class="nav-search-result-title">' + escapeHtml(post.title) + "</div>" +
        '<div class="nav-search-result-meta">' + escapeHtml(post.date || "") +
        (tags ? " · " + escapeHtml(tags) : "") + "</div>";
      container.appendChild(a);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var toggle = document.getElementById("searchToggle");
    var box = document.getElementById("searchBox");
    var input = document.getElementById("searchInput");
    if (!toggle || !box || !input) return;

    function openBox() {
      box.hidden = false;
      loadIndex();
      input.focus();
    }
    function closeBox() {
      box.hidden = true;
    }

    toggle.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();
      if (box.hidden) openBox();
      else closeBox();
    });

    input.addEventListener("click", function (e) { e.stopPropagation(); });

    input.addEventListener("input", function () {
      var query = input.value.trim().toLowerCase();
      loadIndex().then(function (data) {
        var results = query ? data.filter(function (p) { return matches(p, query); }) : [];
        renderResults(results, query);
      });
    });

    document.addEventListener("click", closeBox);
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") closeBox();
    });
  });
})();

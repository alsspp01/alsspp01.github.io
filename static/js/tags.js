(function () {
  "use strict";

  var selectedTags = new Set();
  var logic = "AND";
  var indexData = null;

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function matchesLogic(post) {
    var postTags = post.tags || [];
    var tags = Array.from(selectedTags);
    if (logic === "AND") {
      return tags.every(function (t) { return postTags.includes(t); });
    }
    return tags.some(function (t) { return postTags.includes(t); });
  }

  function renderResults() {
    var container = document.getElementById("tagsFilterResults");
    if (!container) return;
    container.innerHTML = "";
    if (selectedTags.size === 0 || !indexData) return;

    var results = indexData.filter(matchesLogic);
    if (results.length === 0) {
      var empty = document.createElement("p");
      empty.className = "tags-filter-empty";
      empty.textContent = "No posts match.";
      container.appendChild(empty);
      return;
    }
    results.forEach(function (post) {
      var item = document.createElement("div");
      item.className = "post-title";
      var html =
        '<a href="' + post.url + '" class="post-link">' + escapeHtml(post.title) + "</a>" +
        '<div class="flex-break"></div>' +
        '<span class="post-date">' + escapeHtml(post.date || "") + "</span>";
      if (post.description) {
        html += '<div class="flex-break"></div><span class="post-list-subtitle">' +
          escapeHtml(post.description) + "</span>";
      }
      item.innerHTML = html;
      container.appendChild(item);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var list = document.getElementById("tagsSelectableList");
    if (!list) return;

    list.querySelectorAll("a[data-tag]").forEach(function (link) {
      link.addEventListener("click", function (e) {
        e.preventDefault();
        var tag = link.dataset.tag;
        var li = link.closest("li");
        if (selectedTags.has(tag)) {
          selectedTags.delete(tag);
          li.classList.remove("selected");
        } else {
          selectedTags.add(tag);
          li.classList.add("selected");
        }
        renderResults();
      });
    });

    document.querySelectorAll(".tags-logic-option").forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (logic === btn.dataset.logic) return;
        logic = btn.dataset.logic;
        document.querySelectorAll(".tags-logic-option").forEach(function (b) {
          b.classList.toggle("active", b === btn);
        });
        renderResults();
      });
    });

    fetch("/index.json")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        indexData = data;
        renderResults();
      });
  });
})();

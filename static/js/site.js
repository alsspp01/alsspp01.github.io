// Site-level behavior layered on top of the gokarna theme (theme files stay untouched).
(function () {
  // First visit with no saved language: show English unless the browser prefers Korean.
  // Runs before the theme's lang-toggle reads the preference on DOMContentLoaded.
  try {
    if (!localStorage.getItem('lang-preference')) {
      var langs = navigator.languages || [navigator.language || ''];
      var prefersKo = langs.some(function (l) { return /^ko\b/i.test(l); });
      if (!prefersKo) localStorage.setItem('lang-preference', 'en');
    }
  } catch (e) { /* storage blocked: theme falls back to its default */ }

  function closeMenu() {
    var menu = document.querySelector('.nav-hamburger-list');
    if (menu) menu.classList.add('visibility-hidden');
  }

  // Close the hamburger menu on a click outside it, on Escape, or after choosing a link.
  document.addEventListener('click', function (e) {
    var menu = document.querySelector('.nav-hamburger-list');
    var toggle = document.getElementById('hamburger-menu-toggle');
    if (!menu || menu.classList.contains('visibility-hidden')) return;
    if (toggle && toggle.contains(e.target)) return;
    if (menu.contains(e.target) && !e.target.closest('a')) return;
    closeMenu();
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') closeMenu();
  });

  // Mirror the theme's dark stylesheet state as a class so site.css can adjust colors.
  function syncDark() {
    var css = document.getElementById('dark-theme');
    document.documentElement.classList.toggle('is-dark', !!css && !css.disabled);
  }
  document.addEventListener('click', function (e) {
    if (e.target.closest && e.target.closest('.dark-theme-toggle')) setTimeout(syncDark, 0);
  });

  // Back-to-top button on every page.
  document.addEventListener('DOMContentLoaded', function () {
    syncDark();
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'site-top-btn';
    btn.setAttribute('aria-label', 'Back to top');
    btn.innerHTML = '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M12 5l-7 7 1.4 1.4L11 8.8V20h2V8.8l4.6 4.6L19 12z" fill="currentColor"/></svg>';
    btn.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    document.body.appendChild(btn);
    var onScroll = function () {
      btn.classList.toggle('is-visible', window.scrollY > 300);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  });
})();

/* Keyboard shortcuts for the dashboard.
   Space = start / pause     1-4 = Live, Analytics, System, Presentation */
(function () {
  function typing(el) {
    var t = el && el.tagName;
    return t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT' || (el && el.isContentEditable) ||
           (el && el.closest && el.closest('.city-select'));
  }
  document.addEventListener('keydown', function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey || typing(document.activeElement)) { return; }
    if (e.key === ' ' || e.code === 'Space') {
      var ptext = document.getElementById('ptext');
      var running = ptext && /running/.test(ptext.textContent);
      var btn = document.getElementById(running ? 'btn-pause' : 'btn-start');
      if (btn) { btn.click(); e.preventDefault(); }
    } else if (e.key >= '1' && e.key <= '4') {
      var inputs = document.querySelectorAll('#view input');
      var i = parseInt(e.key, 10) - 1;
      if (inputs[i]) { inputs[i].click(); }
    }
  });
})();

function toggleSearch() {
  const overlay = $('#search-overlay');
  if (overlay.hasClass('active')) {
    overlay.removeClass('active');
  } else {
    overlay.addClass('active');
    setTimeout(() => $('#search-input').focus(), 50);
  }
}

function doSearch(q) {
  if (q.length < 2) { $('#search-results').empty(); return; }
  const uidParam = currentUser ? `&user_id=${currentUser.id}` : '';
  $.getJSON(`${API}/api/items/search?q=${encodeURIComponent(q)}&limit=15${uidParam}`, function(data) {
    let html = '';
    for (const it of data.items) {
      const userBadge = it.user_rating
        ? `<span class="user-rating">Your ★${it.user_rating.toFixed(0)}</span>`
        : '';
      html += `<div class="search-item" onclick="openItem('${it.external_id}'); toggleSearch();">
        <span class="domain-icon">${it.domain === 'movie' ? '🎬' : '🎮'}</span>
        <span class="title">${esc(it.title)}</span>
        ${userBadge}
        ${it.avg_rating ? `<span class="rating">★${it.avg_rating.toFixed(1)}</span>` : ''}
      </div>`;
    }
    $('#search-results').html(html);
  });
}

$('#search-overlay').on('click', function(e) {
  if ($(e.target).is('#search-overlay')) toggleSearch();
});

$(document).on('keydown', function(e) {
  if (e.key === 'Escape') {
    $('#search-overlay').removeClass('active');
  }
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
    e.preventDefault();
    toggleSearch();
  }
});

function loadItemDetail(externalId) {
  $('#page-detail').html('<div class="spinner"></div>');
  const uidParam = currentUser ? `?user_id=${currentUser.id}` : '';
  $.getJSON(`${API}/api/items/${externalId}${uidParam}`, function(item) {
    const isMovie = item.domain === 'movie';
    const tagBg = isMovie ? 'background:rgba(37,99,235,0.9);color:#dbeafe' : 'background:rgba(220,38,38,0.9);color:#ede9fe';
    const img = item.image_url
      ? `<img src="${item.image_url}" alt="">`
      : `<span style="font-size:48px;color:#374151">${isMovie ? '🎬' : '🎮'}</span>`;

    const backTarget = currentUser ? `navigateTo('/recs/${currentUser.id}')` : `navigateTo('/')`;

    let html = `<div class="detail-wrap">
      <div style="margin-bottom:16px">
        <button class="btn" onclick="${backTarget}">← Back</button>
      </div>
      <div class="detail-header">
        <div class="detail-poster">${img}</div>
        <div class="detail-info">
          <h1>${esc(item.title)}</h1>
          <span class="domain-tag" style="${tagBg}">${item.domain.toUpperCase()}</span>
          <div class="detail-meta">
            ${item.avg_rating ? `<span>★ ${item.avg_rating.toFixed(1)}</span>` : ''}
            ${item.rating_count ? `<span>${item.rating_count} ratings</span>` : ''}
          </div>
          ${currentUser ? renderStarRating(externalId, item.user_rating || 0) : ''}
          ${item.description ? `<p class="desc">${esc(item.description)}</p>` : ''}
        </div>
      </div>`;

    if (item.similar_games && item.similar_games.length > 0) {
      const gameCards = item.similar_games.map(s => ({...s, rating_count: 0, external_id: s.external_id}));
      html += `<div class="rec-section">
        <div class="rec-header">
          <div>
            <div style="display:flex;align-items:center;gap:8px">
              <h2>Similar Games</h2><span class="model-tag">SBERT Content Similarity</span>
            </div>
            <div class="subtitle">Games with similar themes and descriptions</div>
          </div>
        </div>
        ${renderLane(gameCards, 'game')}
      </div>`;
    }

    html += '</div>';
    $('#page-detail').html(html);
  }).fail(function() {
    $('#page-detail').html('<div class="detail-wrap"><p style="color:#ef4444">Item not found</p><button class="btn" onclick="navigateTo(\'/\')">Go Home</button></div>');
  });
}

function renderStarRating(externalId, currentRating) {
  let stars = '';
  for (let i = 1; i <= 5; i++) {
    const cls = i <= currentRating ? 'active' : '';
    stars += `<span class="star ${cls}" onclick="submitRating('${externalId}', ${i})">★</span>`;
  }
  return `<div class="star-rating" id="stars-${externalId}">${stars}</div>`;
}

function submitRating(externalId, rating) {
  if (!currentUser) return;
  $(`#stars-${externalId} .star`).each(function(i) {
    $(this).toggleClass('active', i < rating);
  });
  $.ajax({
    url: `${API}/api/ratings`,
    method: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({ user_id: currentUser.id, external_id: externalId, rating: rating }),
  });
}

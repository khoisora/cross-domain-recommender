function loadRecommendations(userId) {
  $('#page-recs').html('<div class="spinner"></div>');

  $.when(
    $.getJSON(`${API}/api/users/${userId}`),
    $.getJSON(`${API}/api/recommendations/${userId}`)
  ).done(function(userData, recData) {
    const user = userData[0];
    const data = recData[0];
    if (!currentUser || currentUser.id !== userId) {
      setCurrentUser({ id: userId, name: user.name || 'User' });
    }
    $('#nav-username').text(user.name || '');

    let html = '<div class="container" style="padding-top:16px;padding-bottom:40px">';

    if (data.segment) {
      const seg = data.segment;
      const segLabel = ({cold_start:'Cold Start', one_shot:'One-Shot', warm:'Warm'})[seg.segment] || seg.segment;
      html += `<div class="segment-banner ${esc(seg.segment)}">
        <span class="seg-pill ${esc(seg.segment)}">${esc(segLabel)}</span>
        <span class="seg-text">${esc(seg.explainer)}</span>
        <span class="seg-counts">${seg.movie_count} movies · ${seg.game_count} games rated</span>
      </div>`;
    }

    if (user.ratings && user.ratings.length > 0) {
      const ratedMovies = user.ratings.filter(r => r.domain === 'movie');
      const ratedGames = user.ratings.filter(r => r.domain === 'game');
      html += `<div class="rated-section">
        <div class="rated-header" onclick="$(this).find('.chevron').toggleClass('open'); $('#rated-content').slideToggle(200)">
          <span style="font-size:15px">⭐</span>
          <span style="font-size:14px;font-weight:600;color:#e5e5e5">Your Rated Items</span>
          <span style="background:#1e1e2e;padding:1px 8px;border-radius:10px;font-size:10px;color:#9ca3af">${user.ratings.length}</span>
          <span class="chevron" style="margin-left:auto">▶</span>
        </div>
        <div id="rated-content" style="margin-top:8px;display:none">`;
      if (ratedMovies.length > 0) {
        const movieCards = ratedMovies.map(r => ({
          external_id: r.external_id || r.item_id, title: r.title, domain: 'movie',
          image_url: r.image_url || '', avg_rating: null, rating_count: 0,
          _user_rating: r.rating
        }));
        html += renderLane(movieCards, 'movie');
      }
      if (ratedGames.length > 0) {
        const gameCards = ratedGames.map(r => ({
          external_id: r.external_id || r.item_id, title: r.title, domain: 'game',
          image_url: r.image_url || '', avg_rating: null, rating_count: 0,
          _user_rating: r.rating
        }));
        html += renderLane(gameCards, 'game');
      }
      html += '</div></div>';
    }

    for (const row of data.rows) {
      html += renderRecRow(row);
    }
    html += '</div>';
    $('#page-recs').html(html);
  }).fail(function() {
    $('#page-recs').html('<div class="container" style="padding:40px;text-align:center"><p style="color:#ef4444">Failed to load recommendations</p></div>');
  });
}

function renderRecRow(row) {
  const games = row.items.filter(i => i.domain === 'game' || !i.domain);
  const tag = row.model_tag ? `<span class="model-tag">${esc(row.model_tag)}</span>` : '';

  let html = `<div class="rec-section">
    <div class="rec-header">
      <div>
        <div style="display:flex;align-items:center;gap:8px">
          <h2>${esc(row.title)}</h2>${tag}
        </div>
        <div class="subtitle">${esc(row.subtitle)}</div>
      </div>
    </div>`;

  if (games.length > 0) html += renderLane(games, 'game');

  html += '</div>';
  return html;
}

function renderLane(items, domain) {
  const isMovie = domain === 'movie';
  const watermark = isMovie ? '🎬' : '🎮';
  const label = isMovie ? 'Movies' : 'Games';
  const cls = isMovie ? 'movie' : 'game';

  let cards = '';
  for (const it of items) {
    cards += renderCard(it);
  }

  return `<div class="domain-lane ${cls}">
    <span class="lane-watermark">${watermark}</span>
    <div class="lane-label ${cls}">
      ${label}
      <span class="lane-count ${cls}">${items.length}</span>
    </div>
    <div class="card-scroll-wrap">
      <button class="scroll-btn left" onclick="scrollLane(this, -1)">‹</button>
      <div class="card-scroll">${cards}</div>
      <button class="scroll-btn right" onclick="scrollLane(this, 1)">›</button>
    </div>
  </div>`;
}

function renderCard(it) {
  const cls = it.domain === 'movie' ? 'movie' : 'game';
  const img = it.image_url
    ? `<img src="${it.image_url}" alt="" loading="lazy">`
    : `<span class="placeholder">${it.domain === 'movie' ? '🎬' : '🎮'}</span>`;
  const userRating = it._user_rating
    ? `<span class="user-rating-badge">★${it._user_rating}</span>` : '';

  return `<div class="item-card ${cls}" data-id="${it.external_id}">
    <div class="img-wrap">
      ${img}
      <span class="domain-badge ${cls}">${it.domain}</span>
      ${userRating}
    </div>
    <div class="card-body">
      <div class="card-title">${esc(it.title)}</div>
      ${it.avg_rating ? `<div class="card-rating">★ ${it.avg_rating.toFixed(1)}</div>` : ''}
    </div>
  </div>`;
}


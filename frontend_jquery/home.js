const GROUP_META = {
  cold_start:       { label: 'Cold Start',       desc: 'Zero game history — EMCDR+cooc provides cold-start recommendations from movie taste alone' },
  few_target:       { label: 'Few Target',        desc: '1-3 game ratings — CDR models blend transferred movie preferences with sparse game history' },
  balanced:         { label: 'Balanced',          desc: 'Active in both movies & games — LightGCN graph + cooc performs best on these users' },
  new:              { label: 'New',               desc: 'Last 10 registered users — see recommendations build up as fold-in kicks in after the first ratings' },
};

let activeGroup = 'cold_start';
let groupedUsers = {};

function createUser() {
  const name = $('#new-user-name').val().trim();
  if (!name) return;
  $('#create-user-btn').prop('disabled', true).text('Creating...');
  $.ajax({
    url: `${API}/api/users`,
    method: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({ name: name }),
    success: function(user) {
      selectUser(user);
    },
    error: function() {
      $('#create-user-btn').prop('disabled', false).text('Start');
    }
  });
}

function loadHome() {
  const html = `
    <div class="hero">
      <h1>CrossRec</h1>
      <p>Game recommendations from your movie taste — a cross-domain transfer demo</p>
      <p class="sub">Powered by LightGCN, EMCDR, PTUPCDR, SBERT, and Co-occurrence Reranking</p>
    </div>
    <div class="container">
      <div style="max-width:440px;margin:0 auto 36px;padding:20px 24px;border-radius:14px;border:1px solid #1e1e2e;background:#12121a">
        <div style="font-size:15px;font-weight:600;color:#e5e5e5;margin-bottom:4px">New here? Jump right in</div>
        <div style="font-size:12px;color:#6b7280;margin-bottom:12px">Create a user to get popularity-based recommendations instantly. Rate items to unlock personalized models.</div>
        <div style="display:flex;gap:8px">
          <input type="text" id="new-user-name" placeholder="Enter your name..."
            style="flex:1;padding:9px 14px;border-radius:8px;border:1px solid #374151;background:#0a0a14;color:white;font-size:14px;outline:none"
            onkeydown="if(event.key==='Enter')createUser()">
          <button id="create-user-btn" class="btn" onclick="createUser()"
            style="background:linear-gradient(135deg,#4f46e5,#7c3aed);border:none;color:white;padding:9px 20px;font-weight:600">Start</button>
        </div>
      </div>
      <div style="text-align:center;font-size:13px;color:#6b7280;margin-bottom:16px">— or pick a sample user —</div>
      <div class="group-tabs" id="group-tabs"></div>
      <div class="group-desc" id="group-desc"></div>
      <div class="user-grid" id="user-grid"></div>
    </div>`;
  $('#page-home').html(html);

  $.getJSON(`${API}/api/user-groups`, function(data) {
    groupedUsers = data.groups;
    allUsers = [];
    for (const [group, users] of Object.entries(groupedUsers)) {
      for (const u of users) {
        u._group = group;
        allUsers.push(u);
      }
    }
    renderGroupTabs();
    renderUsers();
  });
}

function renderGroupTabs() {
  let html = '';
  for (const [key, meta] of Object.entries(GROUP_META)) {
    const users = groupedUsers[key] || [];
    const cls = key === activeGroup ? 'active' : '';
    html += `<button class="group-tab ${cls}" onclick="switchGroup('${key}')">${meta.label}<span class="count">${users.length}</span></button>`;
  }
  $('#group-tabs').html(html);
  $('#group-desc').text(GROUP_META[activeGroup]?.desc || '');
}

function switchGroup(key) {
  activeGroup = key;
  renderGroupTabs();
  renderUsers();
}

function renderUsers() {
  const filtered = (groupedUsers[activeGroup] || []).slice(0, 30);
  let html = '';
  for (const u of filtered) {
    html += `
      <div class="user-card" onclick='selectUser(${JSON.stringify(u).replace(/'/g,"&#39;")})'>
        <div class="user-avatar">${u.avatar || '?'}</div>
        <div class="user-info">
          <h3>${esc(u.name)}</h3>
          <p>${esc(u.taste_summary || '')}</p>
          <div class="meta">${u.total_ratings} ratings</div>
        </div>
      </div>`;
  }
  $('#user-grid').html(html || '<p style="color:#6b7280;text-align:center">No users in this group</p>');
}

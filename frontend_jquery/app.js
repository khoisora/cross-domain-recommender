const API = window.location.origin;
const CURRENT_USER_KEY = 'crossrec.currentUser';
let currentUser = null;
let allUsers = [];

try {
  const saved = localStorage.getItem(CURRENT_USER_KEY);
  if (saved) currentUser = JSON.parse(saved);
} catch (e) {}

function setCurrentUser(user) {
  currentUser = user;
  try {
    if (user) localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(user));
    else localStorage.removeItem(CURRENT_USER_KEY);
  } catch (e) {}
}

// ── HASH ROUTING ────────────────────────────────────────
function navigateTo(path) {
  window.location.hash = path;
}

function handleRoute() {
  const hash = window.location.hash.slice(1) || '/';
  const parts = hash.split('/').filter(Boolean);

  if (parts[0] === 'recs' && parts[1]) {
    const userId = parseInt(parts[1]);
    showPage('recs');
    const user = allUsers.find(u => u.id === userId);
    if (user) {
      setCurrentUser(user);
      $('#nav-username').text(user.name);
    }
    loadRecommendations(userId);
  } else if (parts[0] === 'item' && parts[1]) {
    const externalId = decodeURIComponent(parts[1]);
    showPage('detail');
    loadItemDetail(externalId);
  } else {
    setCurrentUser(null);
    showPage('home');
    loadHome();
  }
}

window.addEventListener('hashchange', handleRoute);

function showPage(page) {
  $('#page-home, #page-recs, #page-detail').addClass('hidden');
  $(`#page-${page}`).removeClass('hidden');
  $('#nav').toggleClass('hidden', page === 'home');
  window.scrollTo(0, 0);
}

function selectUser(user) {
  setCurrentUser(user);
  $('#nav-username').text(user.name);
  navigateTo(`/recs/${user.id}`);
}

function openItem(externalId) {
  navigateTo(`/item/${encodeURIComponent(externalId)}`);
}

// ── UTILS ───────────────────────────────────────────────
function esc(str) {
  if (!str) return '';
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function scrollLane(btn, direction) {
  const scrollEl = $(btn).siblings('.card-scroll')[0];
  if (scrollEl) {
    scrollEl.scrollBy({ left: direction * 510, behavior: 'smooth' });
  }
}

// ── ITEM CARD CLICKS ──────────────────────────────────
$(document).on('click', '.item-card', function(e) {
  const id = $(this).data('id');
  if (id) openItem(id);
});

// ── INIT ────────────────────────────────────────────────
$(function() {
  $('#nav-logo').on('click', function() {
    if (currentUser) {
      navigateTo(`/recs/${currentUser.id}`);
    } else {
      navigateTo('/');
    }
  });
  $('#btn-search').on('click', toggleSearch);
  handleRoute();
});

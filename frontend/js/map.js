/**
 * DermaAI — Dermatologist Map (Leaflet.js)
 * Renders mock dermatologist data on an interactive map
 * and populates the card list.
 */

const API_BASE = window.API_BASE || 'http://127.0.0.1:5000';

let leafletMap = null;

const RISK_COLORS = {
    Low: '#22c55e',
    Medium: '#eab308',
    High: '#ef4444',
};

function createMarkerIcon(color) {
    return L.divIcon({
        className: '',
        html: `<div style="
      width: 16px; height: 16px;
      background: ${color};
      border: 2.5px solid white;
      border-radius: 50%;
      box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    "></div>`,
        iconSize: [16, 16],
        iconAnchor: [8, 8],
    });
}

function renderDermCard(doc) {
    const card = document.createElement('div');
    card.className = 'derm-card';
    card.innerHTML = `
    <div class="derm-name">${doc.name}</div>
    <div class="derm-addr">📌 ${doc.address}</div>
    <div class="derm-meta">
      <span class="derm-rating">★ ${doc.rating}</span>
      <span class="derm-dist">${doc.distance}</span>
      <span class="derm-open ${doc.open ? 'open' : 'closed'}">${doc.open ? 'Open Now' : 'Closed'}</span>
    </div>
    <div class="derm-phone" style="margin-top:8px;font-size:0.75rem;">${doc.phone}</div>
  `;
    return card;
}

window.loadDermatologists = async function () {
    const dermList = document.getElementById('derm-list');
    dermList.innerHTML = '<div class="derm-loading">Loading …</div>';

    try {
        const resp = await fetch(`${API_BASE}/api/dermatologists`);
        const data = await resp.json();
        const docs = data.results || [];

        // ── Render cards ──────────────────────────────────────
        dermList.innerHTML = '';
        docs.forEach(doc => {
            const card = renderDermCard(doc);
            dermList.appendChild(card);
        });

        // ── Init / reset map ──────────────────────────────────
        if (leafletMap) {
            leafletMap.remove();
            leafletMap = null;
        }

        const center = docs.length
            ? [docs[0].lat, docs[0].lng]
            : [12.9716, 77.5946]; // Bangalore default

        leafletMap = L.map('map', {
            center: center,
            zoom: 13,
            zoomControl: true,
            attributionControl: false,
        });

        // Dark tile layer (OpenStreetMap + muted style)
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            maxZoom: 19,
            attribution: '© OpenStreetMap contributors © CARTO',
        }).addTo(leafletMap);

        // Attribution small
        L.control.attribution({ position: 'bottomright', prefix: '' })
            .addAttribution('© OSM / CARTO')
            .addTo(leafletMap);

        // Add markers
        docs.forEach((doc, i) => {
            const marker = L.marker([doc.lat, doc.lng], {
                icon: createMarkerIcon('#64dcff'),
            }).addTo(leafletMap);

            marker.bindPopup(`
        <div style="font-family:Inter,sans-serif;min-width:180px;color:#e2eaf5;background:#10182e;padding:4px;">
          <strong style="font-size:0.88rem;">${doc.name}</strong><br/>
          <span style="font-size:0.75rem;color:#7a8aa8;">${doc.address}</span><br/>
          <span style="color:#eab308;font-size:0.75rem;">★ ${doc.rating}</span>
          &nbsp;<span style="font-size:0.72rem;color:#7a8aa8;">${doc.distance}</span>
        </div>
      `, {
                className: 'derm-popup',
                maxWidth: 240,
            });

            // Sync card hover ↔ map
            const card = dermList.children[i];
            if (card) {
                card.addEventListener('click', () => {
                    leafletMap.setView([doc.lat, doc.lng], 15, { animate: true });
                    marker.openPopup();
                });
                marker.on('click', () => {
                    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                });
            }
        });

    } catch (err) {
        dermList.innerHTML = `<div class="derm-loading" style="color:#f87171;">
      Failed to load dermatologists. Is the server running?
    </div>`;
    }
};

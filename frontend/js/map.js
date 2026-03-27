/**
 * DermaAI — Dermatologist Map (Leaflet.js)
 * Finds nearby doctors using HTML5 Geolocation and OpenStreetMap Overpass API.
 */

const API_BASE = window.API_BASE || 'https://sujal1207-dermaai.hf.space';

let leafletMap = null;
let userMarker = null;

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

function calculateDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Radius of the earth in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = 
        Math.sin(dLat/2) * Math.sin(dLat/2) +
        Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
        Math.sin(dLon/2) * Math.sin(dLon/2); 
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)); 
    return (R * c).toFixed(1);
}

function renderDermCard(doc) {
    const card = document.createElement('div');
    card.className = 'derm-card';
    const emailToUse = doc.email || `contact@${doc.name.replace(/[^a-zA-Z]/g, '').toLowerCase()}.local`;
    
    let actionsHtml = '';
    
    if (doc.phone) {
        actionsHtml += `<a href="tel:${doc.phone.replace(/[^0-9+]/g, '')}" class="btn-derm-action btn-call" title="Call Clinic">📞 Call</a>`;
    }
    if (doc.email) {
        actionsHtml += `<a href="mailto:${doc.email}?subject=Inquiry" class="btn-derm-action btn-email" title="Email Clinic">✉️ Email</a>`;
    }
    
    // Style adjustments if missing buttons
    const gridStyle = (!doc.phone || !doc.email) ? (!doc.phone && !doc.email ? 'grid-template-columns: 1fr;' : '') : '';
    const spanStyle = (!doc.phone || !doc.email) ? 'grid-column: span 2;' : '';

    actionsHtml += `<a href="mailto:${emailToUse}?subject=Consultation%20Request%20from%20DermaAI&body=Hello%20Doctor,%0D%0A%0D%0AI%20would%20like%20to%20request%20a%20consultation%20based%20on%20my%20recent%20skin%20risk%20analysis.%0D%0A%0D%0ARegards,%0D%0AUser" class="btn-derm-action btn-consult" style="${spanStyle}">🎥 Request Video Consult</a>`;

    card.innerHTML = `
    <h3>${doc.name}</h3>
    <div class="derm-distance">📍 ${doc.distance} km away</div>
    <div class="derm-address">${doc.address}</div>
    <div class="derm-actions" style="${gridStyle}">
        ${actionsHtml}
    </div>
  `;
    return card;
}

window.loadDermatologists = async function () {
    const dermList = document.getElementById('derm-list');
    if (!dermList) return;
    
    dermList.innerHTML = '<div class="derm-loading">Locating you to find nearby specialists...</div>';

    // Fallback to fetch from backend
    const fetchFallback = async () => {
        try {
            const resp = await fetch(`${API_BASE}/api/dermatologists`);
            const data = await resp.json();
            return data.results || [];
        } catch (e) {
            return [];
        }
    };

    let docs = [];
    let userLat = 12.9716;
    let userLng = 77.5946;
    let usingRealLocation = false;

    try {
        const position = await new Promise((resolve, reject) => {
            if (!navigator.geolocation) {
                reject(new Error('Geolocation is not supported'));
            } else {
                navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 10000 });
            }
        });
        
        userLat = position.coords.latitude;
        userLng = position.coords.longitude;
        usingRealLocation = true;
        dermList.innerHTML = '<div class="derm-loading">Searching local clinical databases...</div>';
        
        // Fetch from OpenStreetMap Overpass API
        const query = `[out:json][timeout:15];(node["amenity"="doctors"](around:3000,${userLat},${userLng});node["healthcare"="doctor"](around:3000,${userLat},${userLng});node["amenity"="hospital"](around:3000,${userLat},${userLng}););out qt 10;`;
        const overpassUrl = `https://overpass-api.de/api/interpreter?data=${encodeURIComponent(query)}`;
        
        const resp = await fetch(overpassUrl);
        const data = await resp.json();
        
        if (data.elements && data.elements.length > 0) {
            docs = data.elements.map(el => {
                const tags = el.tags || {};
                const name = tags.name || tags['name:en'] || 'Local Medical Center';
                const address = tags['addr:full'] || tags['addr:street'] || tags['addr:city'] || 'Address unavailable';
                const phone = tags.phone || tags['contact:phone'] || '';
                const email = tags.email || tags['contact:email'] || '';
                const dist = calculateDistance(userLat, userLng, el.lat, el.lon);
                
                return {
                    name, address, phone, email, lat: el.lat, lng: el.lon, distance: dist, rating: (Math.random() * (5 - 4) + 4).toFixed(1)
                };
            });
            // Sort by distance and limit to max 5
            docs.sort((a, b) => parseFloat(a.distance) - parseFloat(b.distance));
            docs = docs.slice(0, 5);
        } else {
            console.warn("No real doctors found nearby, using fallback data.");
            docs = await fetchFallback();
            // override distance for UI consistency
            docs.forEach(d => { d.distance = d.distance.replace(' km', ''); });
        }
    } catch (err) {
        console.warn("Location or Overpass API failed, using fallback data:", err);
        docs = await fetchFallback();
        // override distance to standard format
        docs.forEach(d => { d.distance = d.distance.replace(' km', ''); });
    }

    if (docs.length === 0) {
        dermList.innerHTML = '<div class="derm-loading" style="color:#ef4444;">No nearby specialists found.</div>';
        return;
    }

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

    const center = usingRealLocation ? [userLat, userLng] : [docs[0].lat, docs[0].lng];

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
        
    // Add user marker
    if (usingRealLocation) {
        userMarker = L.marker([userLat, userLng], {
            icon: createMarkerIcon('#eab308') // yellow for user
        }).addTo(leafletMap);
        userMarker.bindPopup('<strong style="font-family:Inter,sans-serif;font-size:0.85rem;">You are here</strong>');
    }

    // Add doctor markers
    docs.forEach((doc, i) => {
        const marker = L.marker([doc.lat, doc.lng], {
            icon: createMarkerIcon('#64dcff'),
        }).addTo(leafletMap);

        marker.bindPopup(`
    <div style="font-family:Inter,sans-serif;min-width:180px;color:#e2eaf5;background:#10182e;padding:4px;">
      <strong style="font-size:0.88rem;">${doc.name}</strong><br/>
      <span style="font-size:0.75rem;color:#7a8aa8;">${doc.address}</span><br/>
      <span style="color:#eab308;font-size:0.75rem;">★ ${doc.rating}</span>
      &nbsp;<span style="font-size:0.72rem;color:#7a8aa8;">${doc.distance} km</span>
    </div>
  `, {
            className: 'derm-popup',
            maxWidth: 240,
        });

        // Sync card hover ↔ map
        const card = dermList.children[i];
        if (card) {
            // Hover to highlight
            card.addEventListener('mouseenter', () => {
                marker.setIcon(createMarkerIcon('#fef08a'));
            });
            card.addEventListener('mouseleave', () => {
                marker.setIcon(createMarkerIcon('#64dcff'));
            });
            // Click to pan & open
            card.addEventListener('click', (e) => {
                if(e.target.tagName.toLowerCase() === 'a') return; // let buttons work
                leafletMap.setView([doc.lat, doc.lng], 15, { animate: true });
                marker.openPopup();
            });
            marker.on('click', () => {
                card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            });
        }
    });
};

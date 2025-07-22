const CACHE_NAME = 'face-checkin-v1.0.0';
const STATIC_CACHE = 'face-checkin-static-v1.0.0';
const DYNAMIC_CACHE = 'face-checkin-dynamic-v1.0.0';

// Files to cache immediately
const STATIC_ASSETS = [
  '/checkin',
  '/employee-images', 
  '/setup',
  '/offline',
  '/assets/face_checkin/manifest.json',
  '/assets/face_checkin/js/sw.js',
  '/assets/face_checkin/js/pwa-installer.js',
  '/assets/face_checkin/icon-192.png',
  '/assets/face_checkin/icon-512.png'
];

// API endpoints that should be cached
const API_CACHE_PATTERNS = [
  '/api/method/face_checkin.api.face_api.get_projects',
  '/api/method/face_checkin.api.face_api.get_checkin_status', 
  '/api/method/face_checkin.api.face_api.check_system_status',
  '/api/resource/Employee'
];

// Install event - cache static assets
self.addEventListener('install', event => {
  console.log('[SW] Installing service worker');
  
  event.waitUntil(
    caches.open(STATIC_CACHE).then(cache => {
      console.log('[SW] Caching static assets');
      return cache.addAll(STATIC_ASSETS.map(url => new Request(url, {
        cache: 'reload'
      })));
    }).catch(error => {
      console.error('[SW] Failed to cache static assets:', error);
    })
  );
  
  // Force activation of new service worker
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('[SW] Activating service worker');
  
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE) {
            console.log('[SW] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      // Take control of all pages immediately
      return self.clients.claim();
    })
  );
});

// Fetch event - serve from cache or network
self.addEventListener('fetch', event => {
  const request = event.request;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Skip Chrome extension requests
  if (url.protocol === 'chrome-extension:') {
    return;
  }
  
  event.respondWith(
    handleRequest(request)
  );
});

async function handleRequest(request) {
  const url = new URL(request.url);
  
  try {
    // Strategy 1: Cache First for static assets
    if (isStaticAsset(request)) {
      return await cacheFirst(request);
    }
    
    // Strategy 2: Network First for API calls with cache fallback
    if (isApiCall(request)) {
      return await networkFirst(request);
    }
    
    // Strategy 3: Stale While Revalidate for pages
    if (isPageRequest(request)) {
      return await staleWhileRevalidate(request);
    }
    
    // Default: Network only
    return await fetch(request);
    
  } catch (error) {
    console.error('[SW] Request failed:', error);
    
    // Return offline fallback for page requests
    if (isPageRequest(request)) {
      return await getOfflinePage();
    }
    
    throw error;
  }
}

// Cache First Strategy
async function cacheFirst(request) {
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    return cachedResponse;
  }
  
  const networkResponse = await fetch(request);
  if (networkResponse.ok) {
    const cache = await caches.open(STATIC_CACHE);
    cache.put(request, networkResponse.clone());
  }
  
  return networkResponse;
}

// Network First Strategy
async function networkFirst(request) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      console.log('[SW] Serving from cache:', request.url);
      return cachedResponse;
    }
    throw error;
  }
}

// Stale While Revalidate Strategy
async function staleWhileRevalidate(request) {
  const cache = await caches.open(DYNAMIC_CACHE);
  const cachedResponse = await cache.match(request);
  
  const fetchPromise = fetch(request).then(networkResponse => {
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  }).catch(() => {
    // Network failed, return cached version if available
    return cachedResponse;
  });
  
  return cachedResponse || await fetchPromise;
}

// Helper functions
function isStaticAsset(request) {
  const url = new URL(request.url);
  return url.pathname.includes('/assets/') || 
         url.pathname.includes('/files/') ||
         url.hostname !== self.location.hostname;
}

function isApiCall(request) {
  const url = new URL(request.url);
  return url.pathname.startsWith('/api/') ||
         API_CACHE_PATTERNS.some(pattern => url.pathname.includes(pattern));
}

function isPageRequest(request) {
  const url = new URL(request.url);
  return request.headers.get('accept')?.includes('text/html') ||
         url.pathname === '/' ||
         url.pathname.startsWith('/checkin') ||
         url.pathname.startsWith('/employee-images') ||
         url.pathname.startsWith('/setup');
}

async function getOfflinePage() {
  try {
    // Try to get the cached offline page first
    const offlinePage = await caches.match('/offline');
    if (offlinePage) {
      return offlinePage;
    }
  } catch (error) {
    console.error('[SW] Error getting offline page from cache:', error);
  }
  
  // Fallback to inline offline page
  return new Response(`
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Offline - Face Check-in System</title>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
      <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
          color: #f9fafb;
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          text-align: center;
          padding: 2rem;
        }
        .offline-container {
          max-width: 500px;
          background: rgba(31, 41, 55, 0.8);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(75, 85, 99, 0.3);
          border-radius: 16px;
          padding: 3rem;
        }
        .offline-icon {
          font-size: 4rem;
          margin-bottom: 2rem;
          opacity: 0.7;
        }
        h1 { font-size: 2rem; margin-bottom: 1rem; }
        p { font-size: 1.125rem; color: #d1d5db; margin-bottom: 2rem; line-height: 1.6; }
        .retry-btn {
          background: linear-gradient(135deg, #3b82f6, #1d4ed8);
          color: white;
          padding: 0.875rem 2rem;
          border: none;
          border-radius: 8px;
          font-weight: 600;
          cursor: pointer;
          font-size: 1rem;
          margin: 0.5rem;
        }
        .retry-btn:hover { transform: translateY(-2px); }
        .nav-btn {
          background: rgba(75, 85, 99, 0.5);
          color: white;
          padding: 0.75rem 1.5rem;
          border: none;
          border-radius: 8px;
          font-weight: 600;
          cursor: pointer;
          font-size: 0.875rem;
          margin: 0.25rem;
          text-decoration: none;
          display: inline-block;
        }
      </style>
    </head>
    <body>
      <div class="offline-container">
        <div class="offline-icon"><i class="fas fa-mobile-alt"></i></div>
        <h1>You're Offline</h1>
        <p>Face Check-in System is not available right now. Some cached features may still work.</p>
        <button class="retry-btn" onclick="window.location.reload()">
          <i class="fas fa-sync-alt"></i> Try Again
        </button>
        <br>
        <a href="/checkin" class="nav-btn"><i class="fas fa-clipboard-check"></i> Check-in</a>
        <a href="/employee-images" class="nav-btn"><i class="fas fa-users"></i> Employees</a>
      </div>
    </body>
    </html>
  `, {
    headers: { 'Content-Type': 'text/html' }
  });
}

// Background sync for offline checkins
self.addEventListener('sync', event => {
  console.log('[SW] Background sync:', event.tag);
  
  if (event.tag === 'background-checkin') {
    event.waitUntil(syncOfflineCheckins());
  }
});

async function syncOfflineCheckins() {
  try {
    // Get offline checkins from IndexedDB
    const offlineCheckins = await getOfflineCheckins();
    
    for (const checkin of offlineCheckins) {
      try {
        const response = await fetch('/api/method/face_checkin.api.face_api.recognize_and_checkin', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(checkin.data)
        });
        
        if (response.ok) {
          await removeOfflineCheckin(checkin.id);
          console.log('[SW] Synced offline checkin:', checkin.id);
        }
      } catch (error) {
        console.error('[SW] Failed to sync checkin:', error);
      }
    }
  } catch (error) {
    console.error('[SW] Background sync failed:', error);
  }
}

// Push notification handler
self.addEventListener('push', event => {
  console.log('[SW] Push received');
  
  const options = {
    body: event.data ? event.data.text() : 'Face recognition attendance system notification',
    icon: '/assets/face_checkin/icon-192.png',
    badge: '/assets/face_checkin/icon-96.png',
    vibrate: [200, 100, 200],
    data: {
      url: '/checkin'
    },
    actions: [
      {
        action: 'open',
        title: 'Open App',
        icon: '/assets/face_checkin/icon-96.png'
      }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification('Face Check-in System', options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', event => {
  console.log('[SW] Notification clicked');
  
  event.notification.close();
  
  if (event.action === 'open' || !event.action) {
    event.waitUntil(
      clients.openWindow(event.notification.data?.url || '/checkin')
    );
  }
});

// Message handler for communication with main thread
self.addEventListener('message', event => {
  console.log('[SW] Message received:', event.data);
  
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  
  if (event.data && event.data.type === 'GET_VERSION') {
    event.ports[0].postMessage({ version: CACHE_NAME });
  }
});

// Utility functions for IndexedDB operations
async function getOfflineCheckins() {
  try {
    // Basic IndexedDB implementation for offline checkins
    return new Promise((resolve) => {
      const request = indexedDB.open('FaceCheckinDB', 1);
      request.onsuccess = () => {
        const db = request.result;
        if (db.objectStoreNames.contains('offline_checkins')) {
          const transaction = db.transaction(['offline_checkins'], 'readonly');
          const store = transaction.objectStore('offline_checkins');
          const getAll = store.getAll();
          getAll.onsuccess = () => resolve(getAll.result || []);
        } else {
          resolve([]);
        }
      };
      request.onerror = () => resolve([]);
    });
  } catch (error) {
    console.error('[SW] Error getting offline checkins:', error);
    return [];
  }
}

async function removeOfflineCheckin(id) {
  try {
    return new Promise((resolve) => {
      const request = indexedDB.open('FaceCheckinDB', 1);
      request.onsuccess = () => {
        const db = request.result;
        if (db.objectStoreNames.contains('offline_checkins')) {
          const transaction = db.transaction(['offline_checkins'], 'readwrite');
          const store = transaction.objectStore('offline_checkins');
          store.delete(id);
          transaction.oncomplete = () => resolve();
        } else {
          resolve();
        }
      };
      request.onerror = () => resolve();
    });
  } catch (error) {
    console.error('[SW] Error removing offline checkin:', error);
  }
}

console.log('[SW] Service Worker loaded successfully');
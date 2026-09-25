// Per-viewer Google Maps key (browser-key mode). Stored only in this browser's localStorage.
const KEY_STORE = 'catforge.googleMapsKey'

export type GoogleMode = { kind: 'proxy' } | { kind: 'key'; key: string }

export function storedKey(): string | null {
  try { return localStorage.getItem(KEY_STORE) } catch { return null }
}
export function storeKey(key: string | null) {
  try { if (key) localStorage.setItem(KEY_STORE, key); else localStorage.removeItem(KEY_STORE) } catch { /* storage unavailable */ }
}

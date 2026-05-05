const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";
const API_ROOT = API_BASE_URL.replace(/\/api\/v1\/?$/, "");

/**
 * Fire-and-forget ping to wake the Render backend.
 * Called on the landing page so the server is warm by the time the user submits their invite code.
 */
export function prewarmBackend(): void {
  try {
    const ctrl = new AbortController();
    const timerId = setTimeout(() => ctrl.abort(), 90_000);

    fetch(`${API_ROOT}/health`, {
      method: "GET",
      signal: ctrl.signal,
      cache: "no-store",
    })
      .then(() => clearTimeout(timerId))
      .catch(() => clearTimeout(timerId));
  } catch {
    /* swallow — this is best-effort */
  }
}

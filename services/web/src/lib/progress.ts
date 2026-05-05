export function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "n/a";
  }
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  if (mins <= 0) {
    return `${secs}s`;
  }
  return `${mins}m ${secs}s`;
}

export function recordResponseTime(key: string, responseTimeMs: number): void {
  if (typeof window === "undefined") {
    return;
  }
  const current = window.localStorage.getItem(key);
  const stats = current ? JSON.parse(current) : { total: 0, count: 0 };
  stats.total += responseTimeMs;
  stats.count += 1;
  window.localStorage.setItem(key, JSON.stringify(stats));
}

export function getAverageResponseMs(key: string): number {
  if (typeof window === "undefined") {
    return 0;
  }
  const current = window.localStorage.getItem(key);
  if (!current) {
    return 0;
  }
  const stats = JSON.parse(current) as { total: number; count: number };
  if (!stats.count) {
    return 0;
  }
  return Math.round(stats.total / stats.count);
}

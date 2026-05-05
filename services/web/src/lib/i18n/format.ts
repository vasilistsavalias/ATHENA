export function formatI18n(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (_, key: string) => {
    if (!(key in values)) {
      return `{${key}}`;
    }
    return String(values[key]);
  });
}

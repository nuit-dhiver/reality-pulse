/** Join a site path to the configured Astro base URL. */
export function withBase(path = ''): string {
	const base = import.meta.env.BASE_URL;
	if (!path) return base;
	return `${base}${path.replace(/^\//, '')}`;
}

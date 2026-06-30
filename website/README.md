# Reality Pulse Website

Project site for [Reality Pulse](https://github.com/nuit-dhiver/reality-pulse), built with [Astro](https://astro.build) and deployed to GitHub Pages.

Live site: **https://nuit-dhiver.github.io/reality-pulse/**

## Development

```bash
cd website
npm install
npm run dev
```

Open http://localhost:4321/reality-pulse/ in your browser.

## Build

```bash
npm run build
npm run preview
```

## Adding documentation

Add Markdown files to `src/content/docs/`. Each file needs frontmatter:

```yaml
---
title: Page title
description: Short summary for listings and SEO.
order: 3
---
```

Pages are available at `/docs/<filename>/` after build.

## Deployment

Pushes to `main` that touch `website/` trigger the [Deploy website](../../.github/workflows/website.yml) workflow.

Before the first deploy, enable GitHub Pages in the repository settings:

1. Go to **Settings → Pages**
2. Set **Build and deployment → Source** to **GitHub Actions**

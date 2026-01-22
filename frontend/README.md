# AI-OCR Pro Frontend

This is the frontend application for AI-OCR Pro, built with [Next.js](https://nextjs.org) and [Tailwind CSS](https://tailwindcss.com).

## Architecture

The project follows a modern Next.js App Router architecture:

- **Framework**: Next.js 16.1 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **Data Fetching**: TanStack Query (React Query) v5
- **HTTP Client**: Axios
- **Icons**: Lucide React

### Project Structure

```
src/
├── app/
│   ├── (landing)/      # Public landing page route group
│   ├── (dashboard)/    # Protected dashboard route group (requires auth)
│   ├── components/     # Shared UI components (LayoutShell, etc.)
│   ├── layout.tsx      # Root layout
│   └── providers.tsx   # Global providers (QueryClient)
├── lib/
│   └── api.ts          # API client and type definitions
└── data/               # Static data (e.g., landing page content)
```

### Key Features

- **Route Groups**: Separation of concerns between the public landing page and the authenticated dashboard using Next.js Route Groups (`(landing)` vs `(dashboard)`).
- **Layouts**:
    - **Root Layout**: Clean layout for the landing page.
    - **Dashboard Layout**: Includes a persistent sidebar and top bar for the application interface.
- **Client Components**: Interactive UI elements using `"use client"`.

## Backend API Requirements

The frontend expects a backend API running at `/api/v1` (configured via Next.js proxy or CORS). The following endpoints are required:

### Sources
- `GET /sources`: Retrieve a list of configured data sources.
- `POST /sources`: Create a new data source.
    - Body: `{ name: string, path: string, config?: object }`
- `POST /sources/:id/scan`: Trigger a scan job for a specific source.

### Jobs
- `GET /jobs`: Retrieve the status of background processing jobs.

### Documents
- `GET /documents`: Retrieve a list of processed documents.
    - Query Params: `source_id` (optional)

### Search
- `POST /search`: Perform a semantic search across processed documents.
    - Body: `{ query: string, top_k: number }`

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   # or
   yarn
   # or
   pnpm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   # or
   pnpm dev
   ```

3. Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

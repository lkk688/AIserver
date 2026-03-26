"use client";

import { useState, useCallback } from "react";
import {
  onlineSearch,
  OnlineSearchRecord,
  OnlineSearchDomain,
} from "@/lib/api";
import {
  Search,
  Loader2,
  Globe,
  Newspaper,
  GraduationCap,
  Stethoscope,
  ExternalLink,
  X,
  Clock,
  Zap,
  BookOpen,
  ChevronRight,
  RefreshCw,
  Wifi,
  Database,
} from "lucide-react";
import clsx from "clsx";

// ── Domain config ─────────────────────────────────────────────────────────────

interface DomainConfig {
  id: OnlineSearchDomain;
  label: string;
  icon: React.ElementType;
  color: string;
  accent: string;
  defaultLanguage: string;
  placeholder: string;
}

const DOMAINS: DomainConfig[] = [
  {
    id: "web",
    label: "Web",
    icon: Globe,
    color: "text-blue-600",
    accent: "bg-blue-50 border-blue-200",
    defaultLanguage: "mixed",
    placeholder: "Search the web…",
  },
  {
    id: "news",
    label: "News",
    icon: Newspaper,
    color: "text-orange-600",
    accent: "bg-orange-50 border-orange-200",
    defaultLanguage: "mixed",
    placeholder: "Search news articles…",
  },
  {
    id: "academic",
    label: "Academic",
    icon: GraduationCap,
    color: "text-purple-600",
    accent: "bg-purple-50 border-purple-200",
    defaultLanguage: "en",
    placeholder: "Search papers on arXiv, Semantic Scholar…",
  },
  {
    id: "medical_academic",
    label: "Med Papers",
    icon: Stethoscope,
    color: "text-teal-600",
    accent: "bg-teal-50 border-teal-200",
    defaultLanguage: "en",
    placeholder: "Search PubMed, Europe PMC for research papers…",
  },
  {
    id: "medical",
    label: "Health Info",
    icon: Stethoscope,
    color: "text-green-600",
    accent: "bg-green-50 border-green-200",
    defaultLanguage: "en",
    placeholder: "Search MedlinePlus, PubMed for health information…",
  },
];

// ── Source badge ───────────────────────────────────────────────────────────────

const SOURCE_COLORS: Record<string, string> = {
  pubmed: "bg-green-100 text-green-800",
  "europe pmc": "bg-teal-100 text-teal-800",
  medlineplus: "bg-cyan-100 text-cyan-800",
  arxiv: "bg-purple-100 text-purple-800",
  "semantic scholar": "bg-yellow-100 text-yellow-800",
  wikipedia: "bg-gray-100 text-gray-700",
  gnews: "bg-blue-100 text-blue-800",
  newsdata: "bg-indigo-100 text-indigo-800",
  newsapi: "bg-sky-100 text-sky-800",
  serper: "bg-pink-100 text-pink-800",
  tavily: "bg-rose-100 text-rose-800",
  crawler: "bg-slate-100 text-slate-700",
};

function SourceBadge({ source }: { source: string }) {
  const key = Object.keys(SOURCE_COLORS).find((k) =>
    source.toLowerCase().includes(k)
  );
  const cls = key ? SOURCE_COLORS[key] : "bg-gray-100 text-gray-600";
  return (
    <span className={clsx("px-2 py-0.5 rounded-full text-xs font-medium", cls)}>
      {source}
    </span>
  );
}

// ── Result card ───────────────────────────────────────────────────────────────

function ResultCard({
  record,
  onClick,
}: {
  record: OnlineSearchRecord;
  onClick: () => void;
}) {
  const date = record.published_at || record.fetched_at;
  const isBreaking = record.is_breaking;

  return (
    <div
      onClick={onClick}
      className="group bg-white border border-gray-200 rounded-xl p-5 hover:shadow-md hover:border-blue-300 cursor-pointer transition-all duration-150"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          {/* Source + badges row */}
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <SourceBadge source={record.source} />
            {isBreaking && (
              <span className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700">
                <Zap className="w-3 h-3" /> Breaking
              </span>
            )}
            {record.from_cache === true ? (
              <span className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-600">
                <Database className="w-3 h-3" /> Cached
              </span>
            ) : record.from_cache === false ? (
              <span className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-700">
                <Wifi className="w-3 h-3" /> Online
              </span>
            ) : null}
          </div>

          {/* Title */}
          <h3 className="text-base font-semibold text-gray-900 group-hover:text-blue-700 line-clamp-2 mb-1">
            {record.title || "(no title)"}
          </h3>

          {/* Summary */}
          <p className="text-sm text-gray-600 line-clamp-3 leading-relaxed">
            {record.summary || record.content?.slice(0, 200) || "No preview available."}
          </p>
        </div>

        <ChevronRight className="w-5 h-5 text-gray-300 group-hover:text-blue-500 flex-shrink-0 mt-1 transition-colors" />
      </div>

      {/* Footer */}
      <div className="mt-3 pt-3 border-t border-gray-50 flex items-center gap-3 text-xs text-gray-400">
        {date && (
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {new Date(date).toLocaleDateString()}
          </span>
        )}
        <span className="truncate text-blue-500/70">{record.url}</span>
      </div>
    </div>
  );
}

// ── Detail panel ───────────────────────────────────────────────────────────────

function IframeWithFallback({ src, title, content, url }: { src: string; title?: string; content?: string | null; url: string }) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 px-6 text-center bg-gray-50">
        <Globe className="w-10 h-10 text-gray-300" />
        <p className="text-sm text-gray-500 font-medium">This page cannot be embedded.</p>
        <a
          href={url}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-xl text-sm hover:bg-blue-700 transition-colors"
        >
          <ExternalLink className="w-4 h-4" /> Open in New Tab
        </a>
        {content && (
          <div className="w-full mt-2 bg-white border border-gray-200 rounded-xl p-4 max-h-64 overflow-y-auto text-left">
            <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-2">Extracted Text</p>
            <p className="text-xs text-gray-700 whitespace-pre-wrap leading-relaxed">{content}</p>
          </div>
        )}
      </div>
    );
  }

  return (
    <iframe
      src={src}
      title={title}
      className="w-full h-full border-0"
      sandbox="allow-scripts allow-same-origin"
      onError={() => setFailed(true)}
      onLoad={(e) => {
        // Detect X-Frame-Options / CSP block: contentDocument is null or empty
        try {
          const doc = (e.target as HTMLIFrameElement).contentDocument;
          if (!doc || doc.body?.innerHTML === "") setFailed(true);
        } catch {
          setFailed(true);
        }
      }}
    />
  );
}

function DetailPanel({
  record,
  onClose,
}: {
  record: OnlineSearchRecord;
  onClose: () => void;
}) {
  const [leftTab, setLeftTab] = useState<"web" | "pdf">("web");
  const isPdf =
    record.url?.toLowerCase().includes(".pdf") ||
    record.url?.toLowerCase().includes("pdf");

  return (
    <div className="fixed inset-0 z-50 flex">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="relative ml-auto w-full max-w-6xl h-full bg-white shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-gray-50">
          <div className="flex items-center gap-3 min-w-0">
            <SourceBadge source={record.source} />
            <span className="text-sm font-medium text-gray-700 truncate">
              {record.title}
            </span>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <a
              href={record.url}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 text-sm text-blue-600 hover:text-blue-800 px-3 py-1.5 rounded-lg hover:bg-blue-50 transition-colors"
            >
              <ExternalLink className="w-4 h-4" /> Open
            </a>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-gray-500 hover:bg-gray-200 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Two-panel body */}
        <div className="flex flex-1 min-h-0 divide-x divide-gray-200">
          {/* Left: rendered source */}
          <div className="flex-1 flex flex-col min-w-0">
            {/* Sub-tabs */}
            <div className="flex gap-1 px-4 pt-3 pb-0 border-b border-gray-100">
              {(["web", "pdf"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setLeftTab(t)}
                  className={clsx(
                    "px-3 py-1.5 text-xs font-medium rounded-t-lg border-b-2 transition-colors",
                    leftTab === t
                      ? "border-blue-600 text-blue-700 bg-blue-50"
                      : "border-transparent text-gray-500 hover:text-gray-700"
                  )}
                >
                  {t === "web" ? "Web View" : "PDF View"}
                </button>
              ))}
            </div>

            <div className="flex-1 overflow-hidden bg-gray-50">
              {leftTab === "web" ? (
                <IframeWithFallback
                  src={record.url}
                  title={record.title}
                  content={record.content}
                  url={record.url}
                />
              ) : isPdf ? (
                <IframeWithFallback
                  src={`https://mozilla.github.io/pdf.js/web/viewer.html?file=${encodeURIComponent(record.url)}`}
                  title="PDF viewer"
                  content={record.content}
                  url={record.url}
                />
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-gray-400 gap-2">
                  <BookOpen className="w-8 h-8" />
                  <p className="text-sm">Not a PDF document</p>
                </div>
              )}
            </div>
          </div>

          {/* Right: extracted info */}
          <div className="w-96 flex flex-col overflow-y-auto bg-white">
            <div className="p-5 space-y-5">
              {/* Title */}
              <div>
                <h2 className="text-lg font-bold text-gray-900 leading-snug">
                  {record.title}
                </h2>
                <div className="flex flex-wrap items-center gap-2 mt-2">
                  <SourceBadge source={record.source} />
                  <span className="text-xs text-gray-400 capitalize">
                    {record.domain} · {record.category}
                  </span>
                </div>
              </div>

              {/* URL */}
              <div>
                <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-1">
                  URL
                </p>
                <a
                  href={record.url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs text-blue-600 break-all hover:underline"
                >
                  {record.url}
                </a>
              </div>

              {/* Dates */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-1">
                    Published
                  </p>
                  <p className="text-sm text-gray-700">
                    {record.published_at
                      ? new Date(record.published_at).toLocaleDateString()
                      : "—"}
                  </p>
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-1">
                    Fetched
                  </p>
                  <p className="text-sm text-gray-700">
                    {record.fetched_at
                      ? new Date(record.fetched_at).toLocaleDateString()
                      : "—"}
                  </p>
                </div>
              </div>

              {/* Summary */}
              <div>
                <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-2">
                  Summary
                </p>
                <p className="text-sm text-gray-700 leading-relaxed">
                  {record.summary || "No summary available."}
                </p>
              </div>

              {/* Full content */}
              {record.content && (
                <div>
                  <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-2">
                    Extracted Content
                  </p>
                  <div className="bg-gray-50 rounded-lg p-3 max-h-64 overflow-y-auto">
                    <p className="text-xs text-gray-700 whitespace-pre-wrap leading-relaxed">
                      {record.content}
                    </p>
                  </div>
                </div>
              )}

              {/* Meta */}
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-1">
                    Language
                  </p>
                  <p className="text-gray-700 uppercase">{record.language}</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-1">
                    Type
                  </p>
                  <p className="text-gray-700 capitalize">{record.record_type}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Quick category chips ───────────────────────────────────────────────────────

const QUICK_CATEGORIES: Record<OnlineSearchDomain, string[]> = {
  web: ["AI tools", "Python packaging", "NVIDIA Jetson", "vLLM updates"],
  news: ["OpenAI", "AI regulation", "breaking news", "robotics"],
  academic: [
    "large language models",
    "autonomous driving",
    "transformer architecture",
    "reinforcement learning",
  ],
  medical_academic: [
    "GLP-1 weight loss RCT",
    "CBT anxiety disorder efficacy",
    "HIIT metabolic syndrome",
    "mRNA vaccine immunogenicity",
  ],
  medical: [
    "depression treatment options",
    "dietary guidelines adults",
    "physical activity recommendations",
    "child nutrition guidelines",
  ],
};

// ── Per-domain state ───────────────────────────────────────────────────────────

interface DomainState {
  query: string;
  results: OnlineSearchRecord[];
  resultMeta: { query: string; count: number; sources: Record<string, number> } | null;
  error: string | null;
}

const emptyDomainState = (): DomainState => ({
  query: "",
  results: [],
  resultMeta: null,
  error: null,
});

// ── Main page ──────────────────────────────────────────────────────────────────

export default function OnlineSearchPage() {
  const [activeDomain, setActiveDomain] = useState<OnlineSearchDomain>("web");
  const [noCache, setNoCache] = useState(false);
  const [loading, setLoading] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<OnlineSearchRecord | null>(null);

  // Per-domain state — persists when switching tabs
  const [domainStates, setDomainStates] = useState<Record<OnlineSearchDomain, DomainState>>({
    web: emptyDomainState(),
    news: emptyDomainState(),
    academic: emptyDomainState(),
    medical_academic: emptyDomainState(),
    medical: emptyDomainState(),
  });

  const current = domainStates[activeDomain];
  const domain = DOMAINS.find((d) => d.id === activeDomain)!;

  const updateCurrent = useCallback(
    (patch: Partial<DomainState>) =>
      setDomainStates((prev) => ({
        ...prev,
        [activeDomain]: { ...prev[activeDomain], ...patch },
      })),
    [activeDomain]
  );

  const handleSearch = useCallback(
    async (q?: string) => {
      const finalQuery = (q ?? current.query).trim();
      if (!finalQuery) return;
      setLoading(true);
      updateCurrent({ error: null, results: [], resultMeta: null });
      try {
        const result = await onlineSearch(activeDomain, {
          query: finalQuery,
          limit: 12,
          language: domain.defaultLanguage,
          no_cache: noCache,
        });
        const sources: Record<string, number> = {};
        result.items.forEach((r) => {
          sources[r.source] = (sources[r.source] || 0) + 1;
        });
        updateCurrent({
          results: result.items,
          resultMeta: { query: finalQuery, count: result.count, sources },
        });
      } catch (e) {
        updateCurrent({ error: e instanceof Error ? e.message : "Search failed" });
      } finally {
        setLoading(false);
      }
    },
    [current.query, activeDomain, domain.defaultLanguage, noCache, updateCurrent]
  );

  return (
    <div className="min-h-full flex flex-col">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Online Search</h1>
        <p className="text-gray-500 mt-1 text-sm">
          Search news, academic papers, medical literature, and the web — with
          intelligent caching.
        </p>
      </div>

      {/* Domain tabs */}
      <div className="flex gap-2 mb-6 flex-wrap">
        {DOMAINS.map((d) => {
          const Icon = d.icon;
          const isActive = d.id === activeDomain;
          const hasResults = domainStates[d.id].results.length > 0;
          return (
            <button
              key={d.id}
              onClick={() => setActiveDomain(d.id)}
              className={clsx(
                "flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium border transition-all duration-150",
                isActive
                  ? `${d.accent} ${d.color} shadow-sm`
                  : "bg-white border-gray-200 text-gray-600 hover:border-gray-300 hover:bg-gray-50"
              )}
            >
              <Icon className="w-4 h-4" />
              {d.label}
              {hasResults && !isActive && (
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
              )}
            </button>
          );
        })}
      </div>

      {/* Search bar */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSearch();
        }}
        className="relative mb-4"
      >
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
        <input
          type="text"
          value={current.query}
          onChange={(e) => updateCurrent({ query: e.target.value })}
          placeholder={domain.placeholder}
          className="w-full pl-12 pr-36 py-4 text-base border-2 border-gray-200 rounded-2xl shadow-sm focus:border-blue-500 focus:ring-4 focus:ring-blue-50 outline-none transition-all bg-white"
          autoFocus
        />
        <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
          <label className="flex items-center gap-1.5 text-xs text-gray-500 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={noCache}
              onChange={(e) => setNoCache(e.target.checked)}
              className="w-3.5 h-3.5 rounded accent-blue-600"
            />
            Live
          </label>
          <button
            type="submit"
            disabled={loading || !current.query.trim()}
            className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-medium"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Search className="w-4 h-4" />
            )}
            Search
          </button>
        </div>
      </form>

      {/* Quick chips */}
      <div className="flex flex-wrap items-center gap-2 mb-6">
        <span className="text-xs font-medium text-gray-400 uppercase tracking-wide">
          Examples:
        </span>
        {QUICK_CATEGORIES[activeDomain].map((chip) => (
          <button
            key={chip}
            onClick={() => {
              updateCurrent({ query: chip });
              handleSearch(chip);
            }}
            className="px-3 py-1 bg-white border border-gray-200 rounded-full text-xs text-gray-600 hover:bg-gray-50 hover:border-gray-300 transition-colors"
          >
            {chip}
          </button>
        ))}
      </div>

      {/* Error */}
      {current.error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
          {current.error}
        </div>
      )}

      {/* Results header */}
      {current.resultMeta && (
        <div className="mb-4 flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-sm text-gray-600">
              <span className="font-semibold text-gray-900">{current.resultMeta.count}</span>{" "}
              results for &quot;{current.resultMeta.query}&quot;
            </span>
            {Object.entries(current.resultMeta.sources).map(([src, cnt]) => (
              <span
                key={src}
                className="flex items-center gap-1 text-xs text-gray-500 bg-white border border-gray-200 px-2 py-1 rounded-full"
              >
                <SourceBadge source={src} />
                <span className="text-gray-400">×{cnt}</span>
              </span>
            ))}
          </div>
          <button
            onClick={() => handleSearch()}
            className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Refresh
          </button>
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div className="space-y-4">
          {[...Array(4)].map((_, i) => (
            <div
              key={i}
              className="bg-white border border-gray-200 rounded-xl p-5 animate-pulse"
            >
              <div className="flex gap-2 mb-3">
                <div className="h-5 w-20 bg-gray-200 rounded-full" />
                <div className="h-5 w-14 bg-gray-100 rounded-full" />
              </div>
              <div className="h-5 w-3/4 bg-gray-200 rounded mb-2" />
              <div className="h-4 w-full bg-gray-100 rounded mb-1" />
              <div className="h-4 w-5/6 bg-gray-100 rounded" />
            </div>
          ))}
        </div>
      )}

      {/* Results */}
      {!loading && current.results.length > 0 && (
        <div className="space-y-3">
          {current.results.map((r) => (
            <ResultCard
              key={r.id}
              record={r}
              onClick={() => setSelectedRecord(r)}
            />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && current.resultMeta && current.results.length === 0 && (
        <div className="text-center py-16 text-gray-500">
          <Search className="w-10 h-10 mx-auto mb-3 text-gray-300" />
          <p className="font-medium">No results found for &quot;{current.resultMeta.query}&quot;</p>
          <p className="text-sm mt-1">Try enabling &quot;Live&quot; to bypass the cache.</p>
        </div>
      )}

      {/* Initial empty state */}
      {!loading && !current.resultMeta && (
        <div className="flex flex-col items-center justify-center py-20 text-gray-400">
          <div className={clsx("w-16 h-16 rounded-2xl flex items-center justify-center mb-4", domain.accent)}>
            <domain.icon className={clsx("w-8 h-8", domain.color)} />
          </div>
          <p className="font-medium text-gray-600">Search {domain.label}</p>
          <p className="text-sm mt-1">
            {activeDomain === "medical"
              ? "Sources: PubMed · MedlinePlus · Europe PMC"
              : activeDomain === "medical_academic"
              ? "Sources: PubMed · Europe PMC"
              : activeDomain === "academic"
              ? "Sources: arXiv · Semantic Scholar"
              : activeDomain === "news"
              ? "Sources: GNews · NewsData · NewsAPI · RSS"
              : "Sources: Serper · Tavily · Wikipedia · Crawler"}
          </p>
        </div>
      )}

      {/* Detail panel */}
      {selectedRecord && (
        <DetailPanel
          record={selectedRecord}
          onClose={() => setSelectedRecord(null)}
        />
      )}
    </div>
  );
}

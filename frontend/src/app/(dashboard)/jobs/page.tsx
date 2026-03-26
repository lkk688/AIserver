"use client";

import { useState, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  onlineSchedulerStatus,
  onlineSchedulerStart,
  onlineSchedulerStop,
  onlineSchedulerRunNow,
  onlineSchedulerUpdateConfig,
  onlineGetCache,
  onlineDeleteCacheRecord,
  onlineReadUrl,
  SchedulerConfig,
  OnlineSearchRecord,
} from "@/lib/api";
import {
  Play,
  Square,
  Zap,
  Settings,
  Database,
  Link2,
  Plus,
  Trash2,
  Loader2,
  CheckCircle,
  XCircle,
  Clock,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Globe,
  Newspaper,
  GraduationCap,
  Stethoscope,
  Save,
} from "lucide-react";
import clsx from "clsx";

// ── Helpers ────────────────────────────────────────────────────────────────────

function formatNextRun(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  const now = new Date();
  const diffMs = d.getTime() - now.getTime();
  const diffMin = Math.round(diffMs / 60000);
  if (diffMin < 0) return "overdue";
  if (diffMin < 60) return `in ${diffMin}m`;
  if (diffMin < 1440) return `in ${Math.round(diffMin / 60)}h`;
  return d.toLocaleDateString();
}

function JobIdLabel({ id }: { id: string }) {
  const labels: Record<string, { label: string; icon: React.ElementType; color: string }> = {
    breaking_news_job: { label: "Breaking News", icon: Zap, color: "text-red-600 bg-red-50" },
    news_refresh_job: { label: "News Refresh", icon: Newspaper, color: "text-orange-600 bg-orange-50" },
    academic_refresh_job: { label: "Academic Refresh", icon: GraduationCap, color: "text-purple-600 bg-purple-50" },
    web_refresh_job: { label: "Web Refresh", icon: Globe, color: "text-blue-600 bg-blue-50" },
    url_seed_job: { label: "URL Seed Crawl", icon: Link2, color: "text-green-600 bg-green-50" },
  };
  const meta = labels[id] ?? { label: id, icon: Clock, color: "text-gray-600 bg-gray-50" };
  const Icon = meta.icon;
  return (
    <span className={clsx("flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium", meta.color)}>
      <Icon className="w-3.5 h-3.5" />
      {meta.label}
    </span>
  );
}

// ── Tab definitions ────────────────────────────────────────────────────────────

type Tab = "schedule" | "config" | "urls" | "cache";

// ── URL editor ─────────────────────────────────────────────────────────────────

function UrlEditor({
  urls,
  onChange,
}: {
  urls: string[];
  onChange: (u: string[]) => void;
}) {
  const [newUrl, setNewUrl] = useState("");

  const add = () => {
    const u = newUrl.trim();
    if (u && !urls.includes(u)) {
      onChange([...urls, u]);
      setNewUrl("");
    }
  };

  return (
    <div className="space-y-3">
      {/* Add row */}
      <div className="flex gap-2">
        <input
          value={newUrl}
          onChange={(e) => setNewUrl(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && add()}
          placeholder="https://…"
          className="flex-1 px-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          onClick={add}
          className="flex items-center gap-1 px-3 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 transition-colors"
        >
          <Plus className="w-4 h-4" /> Add
        </button>
      </div>

      {/* List */}
      <div className="space-y-1.5 max-h-72 overflow-y-auto pr-1">
        {urls.map((url, i) => (
          <div
            key={i}
            className="flex items-center gap-2 px-3 py-2 bg-gray-50 rounded-lg group"
          >
            <Link2 className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
            <span className="flex-1 text-xs text-gray-700 truncate">{url}</span>
            <a
              href={url}
              target="_blank"
              rel="noreferrer"
              className="opacity-0 group-hover:opacity-100 text-blue-500 hover:text-blue-700 transition-opacity"
            >
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
            <button
              onClick={() => onChange(urls.filter((_, j) => j !== i))}
              className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-600 transition-opacity"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
        {urls.length === 0 && (
          <p className="text-center py-4 text-xs text-gray-400">No URLs configured.</p>
        )}
      </div>
    </div>
  );
}

// ── Query list editor ──────────────────────────────────────────────────────────

function QueryListEditor({
  label,
  queries,
  onChange,
}: {
  label: string;
  queries: string[];
  onChange: (q: string[]) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [newQ, setNewQ] = useState("");

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 transition-colors text-sm font-medium text-gray-700"
      >
        <span>{label} ({queries.length})</span>
        {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>
      {expanded && (
        <div className="px-4 py-3 space-y-2">
          <div className="flex gap-2">
            <input
              value={newQ}
              onChange={(e) => setNewQ(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && newQ.trim()) {
                  onChange([...queries, newQ.trim()]);
                  setNewQ("");
                }
              }}
              placeholder="Add query…"
              className="flex-1 px-3 py-1.5 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={() => {
                if (newQ.trim()) {
                  onChange([...queries, newQ.trim()]);
                  setNewQ("");
                }
              }}
              className="px-3 py-1.5 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {queries.map((q, i) => (
              <span
                key={i}
                className="flex items-center gap-1 px-2.5 py-1 bg-white border border-gray-200 rounded-full text-xs text-gray-700"
              >
                {q}
                <button
                  onClick={() => onChange(queries.filter((_, j) => j !== i))}
                  className="text-gray-400 hover:text-red-500 transition-colors"
                >
                  <XCircle className="w-3.5 h-3.5" />
                </button>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Cache record card ──────────────────────────────────────────────────────────

const DOMAIN_ICONS: Record<string, React.ElementType> = {
  news: Newspaper,
  academic: GraduationCap,
  research: GraduationCap,
  medical: Stethoscope,
  medical_academic: Stethoscope,
  general: Globe,
};

const DOMAIN_COLORS: Record<string, string> = {
  news: "text-orange-600 bg-orange-50",
  academic: "text-purple-600 bg-purple-50",
  research: "text-purple-600 bg-purple-50",
  medical: "text-green-600 bg-green-50",
  medical_academic: "text-teal-600 bg-teal-50",
  general: "text-blue-600 bg-blue-50",
};

function CacheCard({
  record,
  onDelete,
}: {
  record: OnlineSearchRecord;
  onDelete: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const Icon = DOMAIN_ICONS[record.domain] ?? Globe;
  const domainColor = DOMAIN_COLORS[record.domain] ?? "text-gray-600 bg-gray-50";
  const date = record.published_at || record.fetched_at;

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm(`Delete "${record.title || record.id}"?`)) return;
    setDeleting(true);
    try {
      await onlineDeleteCacheRecord(record.id);
      onDelete(record.id);
    } catch {
      alert("Failed to delete record.");
      setDeleting(false);
    }
  };

  return (
    <div className={clsx("bg-white border border-gray-200 rounded-xl overflow-hidden", deleting && "opacity-50 pointer-events-none")}>
      <div className="flex items-start gap-3 px-4 py-3">
        <button
          onClick={() => setExpanded((v) => !v)}
          className="flex-1 flex items-start gap-3 text-left min-w-0"
        >
          <Icon className="w-4 h-4 text-gray-400 mt-0.5 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-0.5 flex-wrap">
              <span className={clsx("px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide", domainColor)}>
                {record.domain.replace("_", " ")}
              </span>
              <span className="text-xs font-medium text-gray-500">{record.source}</span>
              {date && (
                <span className="text-xs text-gray-400">
                  {new Date(date).toLocaleDateString()}
                </span>
              )}
            </div>
            <p className="text-sm font-medium text-gray-900 line-clamp-1">
              {record.title || "(no title)"}
            </p>
            {!expanded && (
              <p className="text-xs text-gray-500 line-clamp-2 mt-0.5">
                {record.summary}
              </p>
            )}
          </div>
          {expanded ? (
            <ChevronUp className="w-4 h-4 text-gray-400 flex-shrink-0" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-400 flex-shrink-0" />
          )}
        </button>
        <button
          onClick={handleDelete}
          disabled={deleting}
          title="Delete from cache"
          className="p-1.5 rounded-lg text-gray-300 hover:text-red-500 hover:bg-red-50 transition-colors flex-shrink-0"
        >
          {deleting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
        </button>
      </div>

      {expanded && (
        <div className="px-4 pb-4 space-y-3 border-t border-gray-100">
          <a
            href={record.url}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1 text-xs text-blue-600 hover:underline mt-2"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            {record.url}
          </a>
          {record.summary && (
            <p className="text-sm text-gray-700 leading-relaxed">{record.summary}</p>
          )}
          {record.content && (
            <div className="bg-gray-50 rounded-lg p-3 max-h-40 overflow-y-auto">
              <p className="text-xs text-gray-600 whitespace-pre-wrap">{record.content}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Cache tab ──────────────────────────────────────────────────────────────────

function CacheTab() {
  const queryClient = useQueryClient();
  const [domainFilter, setDomainFilter] = useState<string>("all");
  const [urlInput, setUrlInput] = useState("");
  const [crawling, setCrawling] = useState(false);
  const [crawlMsg, setCrawlMsg] = useState<string | null>(null);

  const { data, isLoading, refetch } = useQuery({
    queryKey: ["online-cache", domainFilter],
    queryFn: () =>
      onlineGetCache({
        domain: domainFilter === "all" ? undefined : domainFilter,
        limit: 200,
      }),
    refetchInterval: 0,
  });

  const handleCrawlUrl = async () => {
    const url = urlInput.trim();
    if (!url) return;
    setCrawling(true);
    setCrawlMsg(null);
    try {
      const rec = await onlineReadUrl(url, "general", "general", false);
      setCrawlMsg(`Saved: ${rec.title || url}`);
      setUrlInput("");
      refetch();
    } catch (e) {
      setCrawlMsg(`Error: ${e instanceof Error ? e.message : "failed"}`);
    } finally {
      setCrawling(false);
    }
  };

  const domains = ["all", "news", "academic", "medical_academic", "medical", "general"];

  const handleDelete = (id: string) => {
    queryClient.setQueryData(
      ["online-cache", domainFilter],
      (old: typeof data) =>
        old ? { ...old, items: old.items.filter((i) => i.id !== id), count: old.count - 1 } : old
    );
  };

  return (
    <div className="space-y-5">
      {/* Crawl URL */}
      <div className="bg-white border border-gray-200 rounded-xl p-4">
        <p className="text-sm font-medium text-gray-700 mb-3">Read & Cache a URL</p>
        <div className="flex gap-2">
          <input
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCrawlUrl()}
            placeholder="https://…"
            className="flex-1 px-3 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleCrawlUrl}
            disabled={crawling || !urlInput.trim()}
            className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {crawling ? <Loader2 className="w-4 h-4 animate-spin" /> : <Globe className="w-4 h-4" />}
            Crawl
          </button>
        </div>
        {crawlMsg && (
          <p className="mt-2 text-xs text-gray-600">{crawlMsg}</p>
        )}
      </div>

      {/* Domain filter */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-xs text-gray-500 font-medium">Domain:</span>
        {domains.map((d) => (
          <button
            key={d}
            onClick={() => setDomainFilter(d)}
            className={clsx(
              "px-3 py-1 rounded-full text-xs font-medium border transition-colors capitalize",
              domainFilter === d
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-white text-gray-600 border-gray-200 hover:border-gray-300"
            )}
          >
            {d}
          </button>
        ))}
        <button
          onClick={() => refetch()}
          className="ml-auto flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Refresh
        </button>
      </div>

      {/* Record count */}
      {data && (
        <p className="text-xs text-gray-500">
          {data.count} cached record{data.count !== 1 ? "s" : ""}
          {domainFilter !== "all" ? ` in [${domainFilter}]` : ""}
        </p>
      )}

      {/* Loading */}
      {isLoading && (
        <div className="flex justify-center py-10">
          <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
        </div>
      )}

      {/* Records */}
      <div className="space-y-2">
        {data?.items.map((r) => (
          <CacheCard key={r.id} record={r} onDelete={handleDelete} />
        ))}
        {!isLoading && data?.items.length === 0 && (
          <div className="text-center py-10 text-gray-400 text-sm">
            No cached records found.
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────

export default function SchedulerPage() {
  const qc = useQueryClient();
  const [activeTab, setActiveTab] = useState<Tab>("schedule");
  const [localConfig, setLocalConfig] = useState<Partial<SchedulerConfig>>({});
  const [configDirty, setConfigDirty] = useState(false);
  const [runNowMsg, setRunNowMsg] = useState<string | null>(null);

  const { data: status, isLoading } = useQuery({
    queryKey: ["scheduler-status"],
    queryFn: onlineSchedulerStatus,
    refetchInterval: 5000,
  });

  // Sync localConfig from server on first load
  const cfg: SchedulerConfig = {
    ...(status?.config ?? {
      breaking_interval_minutes: 30,
      daily_refresh_hour_1: 7,
      daily_refresh_hour_2: 19,
      seed_crawl_hour: 3,
      seed_urls: [],
      news_refresh_queries: [],
      academic_refresh_queries: [],
      web_refresh_queries: [],
    }),
    ...localConfig,
  };

  const startMut = useMutation({
    mutationFn: () => onlineSchedulerStart(localConfig),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["scheduler-status"] }),
  });

  const stopMut = useMutation({
    mutationFn: onlineSchedulerStop,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["scheduler-status"] }),
  });

  const saveMut = useMutation({
    mutationFn: () => onlineSchedulerUpdateConfig(cfg),
    onSuccess: () => {
      setConfigDirty(false);
      qc.invalidateQueries({ queryKey: ["scheduler-status"] });
    },
  });

  const runNow = useCallback(async () => {
    setRunNowMsg(null);
    try {
      await onlineSchedulerRunNow();
      setRunNowMsg("All jobs triggered — check cache in a few minutes.");
    } catch (e) {
      setRunNowMsg(`Error: ${e instanceof Error ? e.message : "failed"}`);
    }
  }, []);

  const updateCfg = (patch: Partial<SchedulerConfig>) => {
    setLocalConfig((prev) => ({ ...prev, ...patch }));
    setConfigDirty(true);
  };

  const isRunning = status?.running ?? false;

  const tabs: { id: Tab; label: string; icon: React.ElementType }[] = [
    { id: "schedule", label: "Schedule", icon: Clock },
    { id: "config", label: "Config", icon: Settings },
    { id: "urls", label: "Seed URLs", icon: Link2 },
    { id: "cache", label: "Cache", icon: Database },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Search Scheduler</h1>
          <p className="text-gray-500 text-sm mt-1">
            Automate news, academic, and medical data collection
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3 flex-wrap">
          <div
            className={clsx(
              "flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium",
              isRunning
                ? "bg-green-100 text-green-700"
                : "bg-gray-100 text-gray-500"
            )}
          >
            <span
              className={clsx(
                "w-2 h-2 rounded-full",
                isRunning ? "bg-green-500 animate-pulse" : "bg-gray-400"
              )}
            />
            {isRunning ? "Running" : "Stopped"}
          </div>

          {!isRunning ? (
            <button
              onClick={() => startMut.mutate()}
              disabled={startMut.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-xl text-sm font-medium hover:bg-green-700 disabled:opacity-50 transition-colors"
            >
              {startMut.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              Start
            </button>
          ) : (
            <button
              onClick={() => stopMut.mutate()}
              disabled={stopMut.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-xl text-sm font-medium hover:bg-red-700 disabled:opacity-50 transition-colors"
            >
              {stopMut.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Square className="w-4 h-4" />
              )}
              Stop
            </button>
          )}

          <button
            onClick={runNow}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-xl text-sm font-medium hover:bg-blue-700 transition-colors"
          >
            <Zap className="w-4 h-4" /> Run Now
          </button>
        </div>
      </div>

      {runNowMsg && (
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-xl text-sm text-blue-700">
          {runNowMsg}
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <div className="flex gap-1 -mb-px">
          {tabs.map((t) => {
            const Icon = t.icon;
            return (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={clsx(
                  "flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors",
                  activeTab === t.id
                    ? "border-blue-600 text-blue-700"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                )}
              >
                <Icon className="w-4 h-4" />
                {t.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Loading */}
      {isLoading && (
        <div className="flex justify-center py-12">
          <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
        </div>
      )}

      {/* Schedule tab */}
      {!isLoading && activeTab === "schedule" && (
        <div className="space-y-4">
          {status?.jobs.length === 0 ? (
            <div className="text-center py-12 text-gray-400">
              <Clock className="w-10 h-10 mx-auto mb-3 text-gray-300" />
              <p className="text-sm">Scheduler is not running.</p>
              <p className="text-xs mt-1">Click &quot;Start&quot; to begin automatic data collection.</p>
            </div>
          ) : (
            <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
              <table className="w-full">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Job
                    </th>
                    <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Trigger
                    </th>
                    <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Next Run
                    </th>
                    <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {status?.jobs.map((job) => (
                    <tr key={job.id} className="hover:bg-gray-50 transition-colors">
                      <td className="px-5 py-4">
                        <JobIdLabel id={job.id} />
                      </td>
                      <td className="px-5 py-4 text-xs text-gray-500 font-mono">
                        {job.trigger}
                      </td>
                      <td className="px-5 py-4">
                        <span className="flex items-center gap-1.5 text-sm text-gray-700">
                          <Clock className="w-3.5 h-3.5 text-gray-400" />
                          {formatNextRun(job.next_run)}
                        </span>
                      </td>
                      <td className="px-5 py-4">
                        <span className="flex items-center gap-1.5 text-xs text-green-700">
                          <CheckCircle className="w-4 h-4 text-green-500" />
                          Scheduled
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Quick summary cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {[
              { label: "Breaking", value: `${cfg.breaking_interval_minutes}m`, sub: "interval", color: "text-red-600 bg-red-50" },
              { label: "News Refresh", value: `${cfg.daily_refresh_hour_1}:00`, sub: "daily", color: "text-orange-600 bg-orange-50" },
              { label: "Web Refresh", value: `${cfg.daily_refresh_hour_2}:00`, sub: "daily", color: "text-blue-600 bg-blue-50" },
              { label: "URL Seed", value: `${cfg.seed_crawl_hour}:30`, sub: "daily", color: "text-green-600 bg-green-50" },
            ].map((item) => (
              <div
                key={item.label}
                className="bg-white border border-gray-200 rounded-xl p-4"
              >
                <p className="text-xs text-gray-500 mb-1">{item.label}</p>
                <p className={clsx("text-xl font-bold", item.color.split(" ")[0])}>
                  {item.value}
                </p>
                <p className="text-xs text-gray-400 mt-0.5">{item.sub}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Config tab */}
      {!isLoading && activeTab === "config" && (
        <div className="space-y-5 max-w-2xl">
          {configDirty && (
            <div className="flex items-center justify-between p-3 bg-amber-50 border border-amber-200 rounded-xl">
              <span className="text-sm text-amber-700">Unsaved changes</span>
              <button
                onClick={() => saveMut.mutate()}
                disabled={saveMut.isPending}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-600 text-white rounded-lg text-sm hover:bg-amber-700 disabled:opacity-50 transition-colors"
              >
                {saveMut.isPending ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Save className="w-3.5 h-3.5" />
                )}
                Save & Apply
              </button>
            </div>
          )}

          {/* Intervals */}
          <div className="bg-white border border-gray-200 rounded-xl p-5 space-y-4">
            <h3 className="text-sm font-semibold text-gray-800">Timing</h3>

            <div className="grid grid-cols-2 gap-4">
              <label className="block">
                <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                  Breaking interval (min)
                </span>
                <input
                  type="number"
                  min={5}
                  value={cfg.breaking_interval_minutes}
                  onChange={(e) =>
                    updateCfg({ breaking_interval_minutes: +e.target.value })
                  }
                  className="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </label>

              <label className="block">
                <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                  News refresh hour (0-23)
                </span>
                <input
                  type="number"
                  min={0}
                  max={23}
                  value={cfg.daily_refresh_hour_1}
                  onChange={(e) =>
                    updateCfg({ daily_refresh_hour_1: +e.target.value })
                  }
                  className="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </label>

              <label className="block">
                <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                  Web refresh hour (0-23)
                </span>
                <input
                  type="number"
                  min={0}
                  max={23}
                  value={cfg.daily_refresh_hour_2}
                  onChange={(e) =>
                    updateCfg({ daily_refresh_hour_2: +e.target.value })
                  }
                  className="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </label>

              <label className="block">
                <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                  URL seed hour (0-23)
                </span>
                <input
                  type="number"
                  min={0}
                  max={23}
                  value={cfg.seed_crawl_hour}
                  onChange={(e) =>
                    updateCfg({ seed_crawl_hour: +e.target.value })
                  }
                  className="mt-1 w-full px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </label>
            </div>
          </div>

          {/* Refresh queries */}
          <div className="bg-white border border-gray-200 rounded-xl p-5 space-y-3">
            <h3 className="text-sm font-semibold text-gray-800">Refresh Queries</h3>
            <QueryListEditor
              label="News queries"
              queries={cfg.news_refresh_queries}
              onChange={(q) => updateCfg({ news_refresh_queries: q })}
            />
            <QueryListEditor
              label="Academic queries"
              queries={cfg.academic_refresh_queries}
              onChange={(q) => updateCfg({ academic_refresh_queries: q })}
            />
            <QueryListEditor
              label="Web queries"
              queries={cfg.web_refresh_queries}
              onChange={(q) => updateCfg({ web_refresh_queries: q })}
            />
          </div>

          {!configDirty && (
            <button
              onClick={() => saveMut.mutate()}
              disabled={saveMut.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-xl text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {saveMut.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Save className="w-4 h-4" />
              )}
              Save & Apply
            </button>
          )}
        </div>
      )}

      {/* Seed URLs tab */}
      {!isLoading && activeTab === "urls" && (
        <div className="max-w-2xl space-y-4">
          <div className="bg-white border border-gray-200 rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-sm font-semibold text-gray-800">Seed URLs</h3>
                <p className="text-xs text-gray-500 mt-0.5">
                  Crawled daily at {cfg.seed_crawl_hour}:30 → saved to medical cache
                </p>
              </div>
              {configDirty && (
                <button
                  onClick={() => saveMut.mutate()}
                  disabled={saveMut.isPending}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50 transition-colors"
                >
                  {saveMut.isPending ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Save className="w-3.5 h-3.5" />
                  )}
                  Save
                </button>
              )}
            </div>
            <UrlEditor
              urls={cfg.seed_urls}
              onChange={(u) => updateCfg({ seed_urls: u })}
            />
          </div>
        </div>
      )}

      {/* Cache tab */}
      {!isLoading && activeTab === "cache" && <CacheTab />}
    </div>
  );
}

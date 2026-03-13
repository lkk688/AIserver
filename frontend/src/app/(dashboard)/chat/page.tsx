"use client";

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Send, Paperclip, FileText, Code, Download, Save, Sparkles,
  Search, Bot, User, Wrench, X, CheckCircle2, AlertCircle,
  Zap, Clock, BarChart2, ExternalLink, Plus, Trash2, Settings,
  ChevronRight, ArrowLeft, ChevronDown, ChevronUp, Globe,
  Copy, Check, Eye, History,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import clsx from 'clsx';

// ─── Types ────────────────────────────────────────────────────────────────────

type MessageRole = 'user' | 'assistant' | 'system';

interface TurnHeaderBlock  { type: 'turn_header'; turn: number; maxTurns: number }
interface ThinkBlock        { type: 'think'; content: string }
interface ToolBlock         { type: 'tool'; name: string }
interface TextBlock         { type: 'text'; content: string }
interface FileBlock         { type: 'file'; name: string; url: string }
interface CompletionBlock   {
  type: 'completion';
  success: boolean;
  stats?: { total_prompt_tokens: number; total_completion_tokens: number; total_elapsed_s: number };
}

type Block = TurnHeaderBlock | ThinkBlock | ToolBlock | TextBlock | FileBlock | CompletionBlock;

interface Message {
  id: string;
  role: MessageRole;
  content: string;
  blocks?: Block[];
}

interface DocumentState {
  title: string;
  content: string;
  isCustomizing: boolean;
}

interface AgentSettings {
  tool_strategy: 'native_all' | 'hybrid' | 'text_only';
  domain: string;
  max_turns: number;
  parallel_thinking: boolean;
}

interface SessionFile {
  name: string;
  size: number;
  ext: string;
}

interface Session {
  task_id: string;
  goal: string;
  status: string;
  success?: boolean;
  started_at?: string;
  finished_at?: string;
  files?: SessionFile[];
}

interface TurnApiData {
  turn: number;
  thinking?: string;
  text?: string;
  actions?: Array<{
    type: string;
    name?: string;
    path?: string;
    args?: Record<string, unknown>;
  }>;
}

interface TurnsApiResponse {
  task_id: string;
  goal: string;
  status: string;
  success?: boolean;
  result?: string;
  total_turns: number;
  turns: TurnApiData[];
  files?: SessionFile[];
}

// ─── Block renderers ─────────────────────────────────────────────────────────

function TurnHeader({ turn, maxTurns }: { turn: number; maxTurns: number }) {
  return (
    <div className="flex items-center gap-2 my-3 first:mt-0">
      <div className="h-px flex-1 bg-gradient-to-r from-transparent via-indigo-200 to-transparent" />
      <span className="text-[10px] font-semibold text-indigo-400 tracking-widest uppercase px-2 whitespace-nowrap">
        Turn {turn} / {maxTurns}
      </span>
      <div className="h-px flex-1 bg-gradient-to-r from-transparent via-indigo-200 to-transparent" />
    </div>
  );
}

function ThinkBox({ content }: { content: string }) {
  const [collapsed, setCollapsed] = useState(true);
  return (
    <div className="my-2 rounded-xl border border-indigo-900/20 bg-gradient-to-br from-slate-900/90 to-indigo-950/70 backdrop-blur-sm overflow-hidden shadow-inner">
      <button
        onClick={() => setCollapsed(c => !c)}
        className="w-full flex items-center gap-2 px-3 py-1.5 text-left hover:bg-white/5 transition-colors"
      >
        <span className="text-indigo-300 text-xs">💭</span>
        <span className="text-xs font-semibold text-indigo-300 tracking-wide">Thinking</span>
        <span className="ml-auto text-indigo-500 text-xs flex items-center gap-0.5">
          {collapsed ? <><ChevronRight className="w-3 h-3"/> show</> : <><ChevronDown className="w-3 h-3"/> hide</>}
        </span>
      </button>
      {!collapsed && (
        <pre
          className="px-4 pb-3 text-xs text-slate-300 whitespace-pre-wrap leading-relaxed overflow-x-auto max-h-72 overflow-y-auto"
          style={{ fontFamily: 'ui-monospace, SFMono-Regular, monospace' }}
        >
          {content}
        </pre>
      )}
    </div>
  );
}

function ToolBadge({ name }: { name: string }) {
  return (
    <div className="inline-flex items-center gap-1.5 my-0.5 px-2.5 py-1 rounded-lg bg-amber-50 border border-amber-200 text-amber-700 text-xs font-mono">
      <span className="text-amber-500">⚙</span>
      <span>{name}</span>
    </div>
  );
}

function FileBadge({ name, url }: { name: string; url: string }) {
  const ext = name.split('.').pop()?.toLowerCase() ?? '';
  return (
    <div className="flex items-center gap-3 my-2 p-3 rounded-xl border border-indigo-200 bg-gradient-to-r from-indigo-50 to-blue-50 shadow-sm">
      <div className="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center shrink-0">
        <FileText className="w-4 h-4 text-white" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-indigo-900 truncate">{name}</p>
        <p className="text-xs text-indigo-500 capitalize">{ext || 'file'} · saved to workspace</p>
      </div>
      {url && (
        <a
          href={url} target="_blank" rel="noopener noreferrer"
          className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-indigo-600 text-white text-xs font-medium hover:bg-indigo-700 transition-colors shrink-0"
        >
          <ExternalLink className="w-3 h-3" /> Open
        </a>
      )}
    </div>
  );
}

function CompletionBadge({ success, stats }: CompletionBlock) {
  const totalTokens = (stats?.total_prompt_tokens ?? 0) + (stats?.total_completion_tokens ?? 0);
  const elapsed = stats?.total_elapsed_s ?? 0;
  const speed = elapsed > 0 ? Math.round((stats?.total_completion_tokens ?? 0) / elapsed) : 0;

  return (
    <div className={clsx(
      "mt-4 rounded-2xl p-4 border flex flex-col gap-3",
      success ? "bg-gradient-to-br from-emerald-50 to-teal-50 border-emerald-200"
              : "bg-gradient-to-br from-amber-50 to-orange-50 border-amber-200"
    )}>
      <div className="flex items-center gap-2">
        {success
          ? <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
          : <AlertCircle className="w-5 h-5 text-amber-600 shrink-0" />
        }
        <span className={clsx("font-semibold text-sm", success ? "text-emerald-700" : "text-amber-700")}>
          {success ? "Task completed successfully" : "Task ended (max turns reached)"}
        </span>
      </div>
      {stats && totalTokens > 0 && (
        <div className="grid grid-cols-3 gap-2 pt-1 border-t border-black/5">
          <div className="flex flex-col items-center gap-0.5">
            <BarChart2 className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-xs font-bold text-slate-700">{totalTokens.toLocaleString()}</span>
            <span className="text-[10px] text-slate-400">total tokens</span>
          </div>
          <div className="flex flex-col items-center gap-0.5">
            <Zap className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-xs font-bold text-slate-700">{speed}<span className="font-normal text-slate-400"> t/s</span></span>
            <span className="text-[10px] text-slate-400">output speed</span>
          </div>
          <div className="flex flex-col items-center gap-0.5">
            <Clock className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-xs font-bold text-slate-700">{elapsed.toFixed(1)}s</span>
            <span className="text-[10px] text-slate-400">LLM time</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Code block with copy button + macOS chrome ──────────────────────────────

function CodeBlock({ lang, children }: { lang: string; children: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(children).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div className="my-3 rounded-xl overflow-hidden border border-gray-700 bg-gray-900 shadow-lg text-left">
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
          <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/80" />
          <div className="w-2.5 h-2.5 rounded-full bg-green-500/80" />
        </div>
        <span className="text-[11px] text-gray-400 font-mono tracking-wide">{lang || 'code'}</span>
        <button onClick={copy} className="flex items-center gap-1 text-[11px] text-gray-400 hover:text-white transition-colors px-2 py-0.5 rounded hover:bg-gray-700">
          {copied ? <><Check className="w-3 h-3 text-green-400" /> Copied</> : <><Copy className="w-3 h-3" /> Copy</>}
        </button>
      </div>
      <pre className="p-4 overflow-x-auto">
        <code className="text-[13px] font-mono text-green-200 leading-relaxed whitespace-pre">{children}</code>
      </pre>
    </div>
  );
}

// Proper markdown renderer with styled typography
function MarkdownContent({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h1: ({ children }) => <h1 className="text-2xl font-bold text-gray-900 mt-6 mb-3 pb-2 border-b border-gray-200">{children}</h1>,
        h2: ({ children }) => <h2 className="text-xl font-bold text-gray-800 mt-5 mb-2">{children}</h2>,
        h3: ({ children }) => <h3 className="text-lg font-semibold text-gray-800 mt-4 mb-2">{children}</h3>,
        h4: ({ children }) => <h4 className="text-base font-semibold text-gray-700 mt-3 mb-1">{children}</h4>,
        p: ({ children }) => <p className="text-gray-700 leading-relaxed mb-3">{children}</p>,
        ul: ({ children }) => <ul className="list-disc list-inside space-y-1 mb-3 text-gray-700">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal list-inside space-y-1 mb-3 text-gray-700">{children}</ol>,
        li: ({ children }) => <li className="text-gray-700">{children}</li>,
        blockquote: ({ children }) => <blockquote className="border-l-4 border-indigo-300 pl-4 py-1 my-3 bg-indigo-50 rounded-r-lg text-gray-600 italic">{children}</blockquote>,
        code: ({ className, children }) => {
          const lang = className?.replace('language-', '') ?? '';
          if (className?.includes('language-')) {
            return <CodeBlock lang={lang}>{String(children).replace(/\n$/, '')}</CodeBlock>;
          }
          return <code className="bg-gray-100 text-indigo-700 px-1.5 py-0.5 rounded text-xs font-mono">{children}</code>;
        },
        pre: ({ children }) => <>{children}</>,
        table: ({ children }) => <div className="overflow-x-auto mb-3"><table className="w-full text-sm border-collapse border border-gray-200 rounded-lg overflow-hidden">{children}</table></div>,
        thead: ({ children }) => <thead className="bg-gray-50">{children}</thead>,
        th: ({ children }) => <th className="border border-gray-200 px-3 py-2 text-left font-semibold text-gray-700">{children}</th>,
        td: ({ children }) => <td className="border border-gray-200 px-3 py-2 text-gray-700">{children}</td>,
        a: ({ href, children }) => <a href={href} target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:text-indigo-800 underline">{children}</a>,
        strong: ({ children }) => <strong className="font-semibold text-gray-900">{children}</strong>,
        hr: () => <hr className="my-4 border-gray-200" />,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function AssistantMessage({ blocks }: { blocks: Block[] }) {
  return (
    <div className="space-y-0.5">
      {blocks.map((block, i) => {
        switch (block.type) {
          case 'turn_header': return <TurnHeader key={i} turn={block.turn} maxTurns={block.maxTurns} />;
          case 'think':       return <ThinkBox key={i} content={block.content} />;
          case 'tool':        return <ToolBadge key={i} name={block.name} />;
          case 'text':        return (
            <div key={i} className="text-sm text-gray-800">
              <MarkdownContent content={block.content} />
            </div>
          );
          case 'file':        return <FileBadge key={i} name={block.name} url={block.url} />;
          case 'completion':  return <CompletionBadge key={i} {...block} />;
          default:            return null;
        }
      })}
    </div>
  );
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function appendToLastBlock(blocks: Block[], type: 'think' | 'text', data: string): Block[] {
  const last = blocks[blocks.length - 1];
  if (last && last.type === type) {
    return [...blocks.slice(0, -1), { ...last, content: last.content + data }];
  }
  return [...blocks, { type, content: data } as Block];
}

function fileIcon(ext: string): string {
  const map: Record<string, string> = {
    md: '📄', markdown: '📄', txt: '📝', py: '🐍', js: '📦', ts: '📦',
    json: '🗂️', csv: '📊', html: '🌐', css: '🎨', sh: '⚙️',
  };
  return map[ext] ?? '📁';
}

function formatDate(iso?: string): string {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  } catch { return iso; }
}

// ─── History reconstruction ───────────────────────────────────────────────────

function reconstructBlocksFromHistory(data: TurnsApiResponse): Block[] {
  const blocks: Block[] = [];
  const total = data.total_turns;

  for (const turn of data.turns) {
    blocks.push({ type: 'turn_header', turn: turn.turn + 1, maxTurns: total } as TurnHeaderBlock);

    if (turn.thinking?.trim()) {
      blocks.push({ type: 'think', content: turn.thinking } as ThinkBlock);
    }

    // Show tool badges before the text (matches live streaming order)
    for (const action of turn.actions ?? []) {
      if (action.name === 'finish_task') {
        // finish_task summary goes in as text after the loop
      } else if (action.type === 'ActionWriteFile') {
        const name = (action.path as string)?.split('/').pop() ?? String(action.path ?? 'file');
        blocks.push({ type: 'file', name, url: '' } as FileBlock);
      } else if (action.name && action.name !== 'json_parse_error') {
        blocks.push({ type: 'tool', name: action.name } as ToolBlock);
      }
    }

    if (turn.text?.trim()) {
      blocks.push({ type: 'text', content: turn.text } as TextBlock);
    }

    // Append finish_task summary as final text
    for (const action of turn.actions ?? []) {
      if (action.name === 'finish_task' && typeof action.args?.summary === 'string') {
        blocks.push({ type: 'text', content: action.args.summary as string } as TextBlock);
      }
    }
  }

  if (data.status === 'done' || data.status === 'failed') {
    blocks.push({ type: 'completion', success: data.success ?? false } as CompletionBlock);
  }

  return blocks;
}

// ─── Settings Panel ───────────────────────────────────────────────────────────

const DOMAINS = ['general', 'software_eng', 'science', 'finance', 'medical', 'legal'];

function SettingsPanel({ settings, onChange }: { settings: AgentSettings; onChange: (s: AgentSettings) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-t border-gray-100">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-4 py-2.5 text-xs font-medium text-gray-500 hover:bg-gray-50 transition-colors"
      >
        <Settings className="w-3.5 h-3.5" />
        Agent Settings
        {open ? <ChevronUp className="w-3 h-3 ml-auto" /> : <ChevronDown className="w-3 h-3 ml-auto" />}
      </button>
      {open && (
        <div className="px-4 pb-4 space-y-3 bg-gray-50/80 border-t border-gray-100">
          {/* Tool strategy */}
          <div className="pt-3">
            <label className="block text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1">Tool Strategy</label>
            <div className="flex gap-1">
              {(['hybrid', 'native_all', 'text_only'] as const).map(s => (
                <button key={s} onClick={() => onChange({ ...settings, tool_strategy: s })}
                  className={clsx("flex-1 py-1 text-xs rounded-md border transition-colors font-medium",
                    settings.tool_strategy === s ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-gray-600 border-gray-200 hover:border-indigo-300"
                  )}>
                  {s === 'hybrid' ? 'Hybrid' : s === 'native_all' ? 'Native' : 'Text'}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-gray-400 mt-1">
              {settings.tool_strategy === 'hybrid' ? 'Best for small models — JSON tools + text write_file' :
               settings.tool_strategy === 'native_all' ? 'All native JSON calls — may fail for long writes' :
               'Pure text XML tool calls'}
            </p>
          </div>

          {/* Domain */}
          <div>
            <label className="block text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1">Domain</label>
            <div className="relative">
              <Globe className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-gray-400" />
              <select value={settings.domain} onChange={e => onChange({ ...settings, domain: e.target.value })}
                className="w-full pl-7 pr-2 py-1.5 text-xs border border-gray-200 rounded-md bg-white focus:ring-1 focus:ring-indigo-400 focus:border-indigo-400 appearance-none">
                {DOMAINS.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>
          </div>

          {/* Max turns */}
          <div>
            <label className="block text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1">
              Max Turns <span className="font-bold text-indigo-600">{settings.max_turns}</span>
            </label>
            <input type="range" min={3} max={30} step={1}
              value={settings.max_turns}
              onChange={e => onChange({ ...settings, max_turns: +e.target.value })}
              className="w-full accent-indigo-600"
            />
            <div className="flex justify-between text-[10px] text-gray-400 mt-0.5">
              <span>3</span><span>30</span>
            </div>
          </div>

          {/* Parallel thinking */}
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-gray-700">Parallel Thinking</p>
              <p className="text-[10px] text-gray-400">Batch brainstorming mode</p>
            </div>
            <button onClick={() => onChange({ ...settings, parallel_thinking: !settings.parallel_thinking })}
              className={clsx("relative inline-flex h-5 w-9 items-center rounded-full transition-colors",
                settings.parallel_thinking ? "bg-indigo-600" : "bg-gray-200"
              )}>
              <span className={clsx("inline-block h-3.5 w-3.5 rounded-full bg-white transition-transform shadow-sm",
                settings.parallel_thinking ? "translate-x-4.5" : "translate-x-0.5"
              )} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Session History List ─────────────────────────────────────────────────────

function SessionList({
  onNew,
  onSelect,
}: {
  onNew: () => void;
  onSelect: (session: Session) => void;
}) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    fetch('/api/v1/agent/sessions?limit=50')
      .then(r => r.json())
      .then(d => setSessions(d.sessions ?? []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  const deleteSession = async (e: React.MouseEvent, taskId: string) => {
    e.stopPropagation();
    await fetch(`/api/v1/agent/sessions/${taskId}`, { method: 'DELETE' });
    setSessions(prev => prev.filter(s => s.task_id !== taskId));
  };

  return (
    <div className="-m-8 h-[calc(100vh-5rem)] flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-8 py-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center shadow-md">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-900">AI Agent</h1>
            <p className="text-xs text-gray-500">ReAct Agent with tool use</p>
          </div>
        </div>
        <button
          onClick={onNew}
          className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white text-sm font-semibold rounded-xl hover:bg-indigo-700 transition-colors shadow-sm"
        >
          <Plus className="w-4 h-4" /> New Task
        </button>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto px-8 py-6">
        {loading && (
          <div className="flex justify-center py-16">
            <div className="w-6 h-6 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
          </div>
        )}
        {!loading && sessions.length === 0 && (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <div className="w-16 h-16 rounded-2xl bg-indigo-100 flex items-center justify-center mb-4">
              <Bot className="w-8 h-8 text-indigo-400" />
            </div>
            <h3 className="text-base font-semibold text-gray-700 mb-1">No tasks yet</h3>
            <p className="text-sm text-gray-500 mb-4">Start a new task to see it here</p>
            <button onClick={onNew} className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-xl hover:bg-indigo-700 transition-colors">
              <Plus className="w-4 h-4" /> Start First Task
            </button>
          </div>
        )}
        {!loading && sessions.length > 0 && (
          <div className="grid gap-3 max-w-4xl mx-auto">
            {sessions.map(session => (
              <div
                key={session.task_id}
                onClick={() => onSelect(session)}
                className="group bg-white rounded-2xl border border-gray-200 hover:border-indigo-300 hover:shadow-md transition-all cursor-pointer p-4 flex gap-4"
              >
                {/* Status indicator */}
                <div className={clsx(
                  "w-10 h-10 rounded-xl flex items-center justify-center shrink-0 mt-0.5",
                  session.status === 'done' && session.success ? "bg-emerald-100" :
                  session.status === 'done' ? "bg-amber-100" :
                  session.status === 'running' ? "bg-blue-100" : "bg-gray-100"
                )}>
                  {session.status === 'done' && session.success ? <CheckCircle2 className="w-5 h-5 text-emerald-600" /> :
                   session.status === 'done' ? <AlertCircle className="w-5 h-5 text-amber-600" /> :
                   session.status === 'running' ? <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" /> :
                   <Bot className="w-5 h-5 text-gray-400" />}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-gray-900 line-clamp-2 mb-1">{session.goal}</p>
                  <p className="text-xs text-gray-400">{formatDate(session.finished_at || session.started_at)}</p>

                  {/* Files */}
                  {session.files && session.files.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 mt-2">
                      {session.files.slice(0, 6).map((f, i) => (
                        <span key={i} className="inline-flex items-center gap-1 px-2 py-0.5 bg-gray-100 rounded-md text-xs text-gray-600">
                          <span>{fileIcon(f.ext)}</span>
                          <span className="truncate max-w-[120px]">{f.name}</span>
                        </span>
                      ))}
                      {session.files.length > 6 && (
                        <span className="px-2 py-0.5 bg-gray-100 rounded-md text-xs text-gray-500">+{session.files.length - 6} more</span>
                      )}
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div className="flex items-start gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                  <button
                    onClick={e => deleteSession(e, session.task_id)}
                    className="p-1.5 rounded-lg hover:bg-red-50 text-gray-300 hover:text-red-500 transition-colors"
                    title="Delete"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                  <ChevronRight className="w-4 h-4 text-gray-400 mt-1.5" />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Chat View ────────────────────────────────────────────────────────────────

function ChatView({
  onBack,
  taskId,
  initialGoal,
}: {
  onBack: () => void;
  taskId?: string;       // set when viewing an existing session (read-only history)
  initialGoal?: string;
}) {
  const isHistory = !!taskId;

  const [messages, setMessages] = useState<Message[]>(
    isHistory ? [] : [{
      id: '1', role: 'assistant', content: '',
      blocks: [{ type: 'text', content: "Hello! I'm your AI Agent. I can search the web, write code, analyze data, and generate documents. What would you like me to do?" }],
    }]
  );
  const [historyLoading, setHistoryLoading] = useState(isHistory);
  const [input, setInput] = useState(isHistory ? '' : (initialGoal ?? ''));
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamStatus, setStreamStatus] = useState('');

  // Resizable split
  const containerRef = useRef<HTMLDivElement>(null);
  const [splitPct, setSplitPct] = useState(55);
  const dragging = useRef(false);

  const onMouseDown = () => { dragging.current = true; };
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const pct = ((e.clientX - rect.left) / rect.width) * 100;
      setSplitPct(Math.max(30, Math.min(75, pct)));
    };
    const onUp = () => { dragging.current = false; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
  }, []);

  // Right panel state
  const [activeTab, setActiveTab] = useState<'editor' | 'sources'>('editor');
  const [documentState, setDocumentState] = useState<DocumentState>({
    title: 'Untitled Document.md',
    content: '# Welcome\n\nGenerated content will appear here.',
    isCustomizing: false,
  });
  const [attachedFiles, setAttachedFiles] = useState<{ name: string; type: string }[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [activeTools, setActiveTools] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Agent settings
  const [settings, setSettings] = useState<AgentSettings>({
    tool_strategy: 'hybrid',
    domain: 'general',
    max_turns: 15,
    parallel_thinking: false,
  });

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    fetch('/api/v1/agent/tools?strategy=native_all')
      .then(r => r.json())
      .then(data => { if (Array.isArray(data.tools)) setActiveTools(data.tools.map((t: { name?: string; function?: { name?: string } }) => t.name || t.function?.name)); })
      .catch(() => {});
  }, []);

  // Load history when viewing an existing session
  useEffect(() => {
    if (!taskId) return;
    setHistoryLoading(true);
    fetch(`/api/v1/agent/sessions/${taskId}/turns`)
      .then(r => r.json())
      .then((data: TurnsApiResponse) => {
        const userMsg: Message = { id: 'user-0', role: 'user', content: data.goal };
        const assistantBlocks = reconstructBlocksFromHistory(data);
        const assistantMsg: Message = {
          id: 'assistant-0', role: 'assistant', content: '',
          blocks: assistantBlocks,
        };
        setMessages([userMsg, assistantMsg]);

        // Populate document panel with the first .md file found in workspace files
        const mdFile = data.files?.find(f => f.ext === 'md' || f.ext === 'markdown');
        if (mdFile) {
          setDocumentState(prev => ({ ...prev, title: mdFile.name }));
        }
      })
      .catch(console.error)
      .finally(() => setHistoryLoading(false));
  }, [taskId]);

  const downloadDocument = () => {
    const blob = new Blob([documentState.content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = documentState.title || 'document.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Detect if the document is code (not markdown) based on extension
  const docExt = documentState.title.split('.').pop()?.toLowerCase() ?? '';
  const isCodeDoc = !['md', 'markdown', 'txt', ''].includes(docExt);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setAttachedFiles(prev => [...prev, ...Array.from(e.target.files!).map(f => ({ name: f.name, type: f.name.split('.').pop() || 'file' }))]);
    e.target.value = '';
  };

  const updateBlocks = useCallback((updater: (prev: Block[]) => Block[]) => {
    setMessages(prev => {
      const last = prev[prev.length - 1];
      if (last?.role !== 'assistant') return prev;
      return [...prev.slice(0, -1), { ...last, blocks: updater(last.blocks ?? []) }];
    });
  }, []);

  const handleSend = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isStreaming) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsStreaming(true);
    setMessages(prev => [...prev, { id: (Date.now() + 1).toString(), role: 'assistant', content: '', blocks: [] }]);

    try {
      const response = await fetch('/api/v1/agent/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          goal: userMsg.content,
          backend: 'vllm',
          tool_strategy: settings.tool_strategy,
          domain: settings.domain,
          max_turns: settings.max_turns,
          enable_turn_limits: true,
        }),
      });

      if (!response.body) throw new Error('No readable stream');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw || raw === '[DONE]') continue;
          try {
            const event = JSON.parse(raw);
            if (event.type === 'turn_start') {
              setStreamStatus(`Turn ${event.turn}/${event.max_turns}`);
              updateBlocks(b => [...b, { type: 'turn_header', turn: event.turn, maxTurns: event.max_turns } as TurnHeaderBlock]);
            } else if (event.type === 'think') {
              updateBlocks(b => appendToLastBlock(b, 'think', event.data ?? ''));
            } else if (event.type === 'message' || event.type === 'token') {
              updateBlocks(b => appendToLastBlock(b, 'text', event.data ?? ''));
            } else if (event.type === 'tool' && event.status === 'started' && event.name) {
              setStreamStatus(`${event.name}…`);
              setActiveTools(prev => prev.includes(event.name) ? prev : [...prev, event.name]);
              updateBlocks(b => [...b, { type: 'tool', name: event.name } as ToolBlock]);
            } else if (event.type === 'file_written') {
              updateBlocks(b => [...b, { type: 'file', name: event.name, url: event.url } as FileBlock]);
              if (event.name?.endsWith('.md') || event.name?.endsWith('.markdown')) {
                const md = event.content || (event.url ? await fetch(event.url).then(r => r.text()).catch(() => '') : '');
                if (md) { setDocumentState(prev => ({ ...prev, title: event.name, content: md })); setActiveTab('editor'); }
              }
            } else if (event.type === 'done') {
              setStreamStatus('');
              updateBlocks(b => [...b, { type: 'completion', success: event.success, stats: event.stats } as CompletionBlock]);
            } else if (event.type === 'error') {
              setStreamStatus('');
              updateBlocks(b => [...b, { type: 'text', content: `\n\n❌ **Error:** ${event.detail ?? 'Unknown error'}` } as TextBlock]);
            }
          } catch { /* incomplete chunk */ }
        }
      }
    } catch (err) {
      console.error('Chat stream error:', err);
    } finally {
      setIsStreaming(false);
      setStreamStatus('');
    }
  };

  return (
    <div className="-m-8 h-[calc(100vh-5rem)] flex flex-col bg-white overflow-hidden">
      {/* Top bar */}
      <div className="h-12 border-b border-gray-200 bg-white flex items-center px-4 gap-3 shrink-0 z-10">
        <button onClick={onBack} className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-800 transition-colors px-2 py-1 rounded-lg hover:bg-gray-100">
          <ArrowLeft className="w-3.5 h-3.5" /> All tasks
        </button>
        <div className="w-px h-4 bg-gray-200" />
        {isHistory
          ? <><History className="w-4 h-4 text-amber-500" /><span className="font-semibold text-gray-800 text-sm">Session History</span></>
          : <><Sparkles className="w-4 h-4 text-indigo-500" /><span className="font-semibold text-gray-800 text-sm">Agent Chat</span></>
        }
        {!isHistory && (
          <div className="flex gap-1.5 ml-1">
            <span className="px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-700 text-xs font-medium">qwen3.5-9b</span>
            <span className="px-2 py-0.5 rounded-full bg-green-100 text-green-700 text-xs font-medium capitalize">{settings.tool_strategy}</span>
          </div>
        )}
      </div>

      {/* Resizable body — min-h-0 lets flex children shrink properly */}
      <div ref={containerRef} className="flex-1 flex overflow-hidden select-none min-h-0">
        {/* Chat pane */}
        <div className="flex flex-col min-h-0 overflow-hidden" style={{ width: `${splitPct}%` }}>
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-5 space-y-5 bg-gray-50/40 min-h-0">
            {historyLoading && (
              <div className="flex justify-center py-16">
                <div className="flex flex-col items-center gap-3">
                  <div className="w-6 h-6 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin" />
                  <span className="text-xs text-gray-400">Loading session history…</span>
                </div>
              </div>
            )}
            {messages.map(msg => (
              <div key={msg.id} className={clsx("flex max-w-[90%]", msg.role === 'user' ? "ml-auto" : "mr-auto")}>
                {msg.role !== 'user' && (
                  <div className="w-7 h-7 rounded-full bg-indigo-100 flex items-center justify-center mr-2.5 shrink-0 mt-1">
                    <Bot className="w-4 h-4 text-indigo-600" />
                  </div>
                )}
                <div className={clsx(
                  "rounded-2xl px-4 py-3 min-w-0 flex-1 text-sm",
                  msg.role === 'user'
                    ? "bg-indigo-600 text-white rounded-tr-sm shadow-sm"
                    : "bg-white border border-gray-200 text-gray-800 rounded-tl-sm shadow-sm"
                )}>
                  {msg.role === 'user'
                    ? <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                    : <AssistantMessage blocks={msg.blocks ?? [{ type: 'text', content: msg.content }]} />
                  }
                </div>
                {msg.role === 'user' && (
                  <div className="w-7 h-7 rounded-full bg-gray-200 flex items-center justify-center ml-2.5 shrink-0 mt-1">
                    <User className="w-4 h-4 text-gray-600" />
                  </div>
                )}
              </div>
            ))}
            {isStreaming && (
              <div className="flex max-w-[90%] mr-auto">
                <div className="w-7 h-7 rounded-full bg-indigo-100 flex items-center justify-center mr-2.5 shrink-0 mt-1">
                  <Bot className="w-4 h-4 text-indigo-600" />
                </div>
                <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-2 shadow-sm">
                  <div className="flex gap-1">
                    {['-0.3s', '-0.15s', '0s'].map(d => (
                      <div key={d} className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: d }} />
                    ))}
                  </div>
                  {streamStatus && <span className="text-xs text-indigo-500 font-medium">{streamStatus}</span>}
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input area + settings */}
          <div className="bg-white border-t border-gray-200 shrink-0">
            <div className="p-3">
              <form onSubmit={handleSend}
                className="flex items-end gap-2 bg-gray-50 border border-gray-300 rounded-2xl p-2 focus-within:ring-2 focus-within:ring-indigo-500 focus-within:border-transparent transition-all"
              >
                <button type="button" onClick={() => fileInputRef.current?.click()}
                  className="p-1.5 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-200 transition-colors shrink-0">
                  <Paperclip className="w-4 h-4" />
                </button>
                <input type="file" ref={fileInputRef} className="hidden" onChange={handleFileChange} multiple />
                <textarea value={input} onChange={e => setInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                  placeholder="Describe your task…"
                  className="flex-1 max-h-36 min-h-[36px] bg-transparent border-none focus:ring-0 resize-none py-2 px-1 text-sm text-gray-800"
                  rows={1}
                />
                <button type="submit" disabled={!input.trim() || isStreaming}
                  className="p-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 disabled:cursor-not-allowed text-white rounded-xl transition-colors shrink-0 mr-0.5 mb-0.5">
                  <Send className="w-3.5 h-3.5" />
                </button>
              </form>
            </div>
            <SettingsPanel settings={settings} onChange={setSettings} />
          </div>
        </div>

        {/* Drag handle */}
        <div
          onMouseDown={onMouseDown}
          className="w-1.5 shrink-0 cursor-col-resize bg-gray-200 hover:bg-indigo-400 active:bg-indigo-500 transition-colors flex items-center justify-center group z-10"
        >
          <div className="w-0.5 h-8 bg-gray-400 group-hover:bg-indigo-300 rounded-full transition-colors" />
        </div>

        {/* Document / Sources pane */}
        <div className="flex flex-col bg-white overflow-hidden flex-1">
          {/* Tabs */}
          <div className="flex border-b border-gray-200 text-sm shrink-0">
            {(['editor', 'sources'] as const).map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)}
                className={clsx(
                  "flex-1 py-3 flex items-center justify-center gap-1.5 border-b-2 transition-colors text-xs font-medium",
                  activeTab === tab ? "border-indigo-600 text-indigo-700 bg-indigo-50/30" : "border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50"
                )}>
                {tab === 'editor' ? <><FileText className="w-3.5 h-3.5" /> Output Document</> : <><Search className="w-3.5 h-3.5" /> Sources & Tools</>}
              </button>
            ))}
          </div>

          {activeTab === 'editor' ? (
            <div className="flex-1 flex flex-col overflow-hidden">
              {/* Toolbar */}
              <div className="px-4 py-2 border-b border-gray-100 flex items-center gap-2 bg-white shrink-0">
                <input value={documentState.title}
                  onChange={e => setDocumentState(p => ({ ...p, title: e.target.value }))}
                  className="font-mono text-xs font-semibold text-gray-700 bg-transparent border-none focus:ring-1 focus:ring-indigo-300 rounded px-1 min-w-0 flex-1 truncate"
                />
                <div className="flex items-center gap-1 shrink-0">
                  <div className="flex bg-gray-100 rounded-md p-0.5">
                    <button onClick={() => setDocumentState(p => ({ ...p, isCustomizing: false }))}
                      className={clsx("px-2 py-0.5 text-xs rounded transition-all", !documentState.isCustomizing ? "bg-white text-gray-800 shadow-sm" : "text-gray-500")}>
                      Preview
                    </button>
                    <button onClick={() => setDocumentState(p => ({ ...p, isCustomizing: true }))}
                      className={clsx("px-2 py-0.5 text-xs rounded transition-all flex items-center gap-1", documentState.isCustomizing ? "bg-white text-gray-800 shadow-sm" : "text-gray-500")}>
                      <Code className="w-2.5 h-2.5" /> Raw
                    </button>
                  </div>
                  <button className="p-1 text-gray-400 hover:text-indigo-600 rounded transition-colors"><Save className="w-3.5 h-3.5" /></button>
                  <button className="p-1 text-gray-400 hover:text-indigo-600 rounded transition-colors"><Download className="w-3.5 h-3.5" /></button>
                </div>
              </div>
              <div className="flex-1 overflow-auto bg-white">
                {documentState.isCustomizing ? (
                  <textarea value={documentState.content}
                    onChange={e => setDocumentState(p => ({ ...p, content: e.target.value }))}
                    className="w-full h-full p-5 text-xs font-mono text-gray-800 bg-transparent border-none focus:ring-0 resize-none outline-none leading-relaxed"
                    spellCheck={false}
                  />
                ) : (
                  <div className="px-8 py-6 pb-20 max-w-none min-h-full">
                    <MarkdownContent content={documentState.content} />
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto p-5 space-y-6 bg-gray-50/50">
              <section>
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                  <Paperclip className="w-3.5 h-3.5" /> Attached Context
                </h3>
                {attachedFiles.length > 0 ? (
                  <div className="space-y-1.5">
                    {attachedFiles.map((file, idx) => (
                      <div key={idx} className="flex items-center gap-2 p-2.5 bg-white border border-gray-200 rounded-xl group hover:border-indigo-200 transition-colors">
                        <FileText className="w-4 h-4 text-indigo-500 shrink-0" />
                        <span className="text-xs text-gray-700 flex-1 truncate">{file.name}</span>
                        <button onClick={e => { e.stopPropagation(); setAttachedFiles(prev => prev.filter((_, i) => i !== idx)); }}
                          className="p-1 hover:bg-red-50 text-gray-300 hover:text-red-500 rounded transition-colors opacity-0 group-hover:opacity-100">
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <button onClick={() => fileInputRef.current?.click()}
                    className="w-full py-4 border-2 border-dashed border-gray-200 rounded-xl text-xs text-gray-400 hover:border-indigo-300 hover:text-indigo-500 transition-colors">
                    + Attach files
                  </button>
                )}
              </section>
              <section>
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                  <Wrench className="w-3.5 h-3.5" /> Active Tools
                </h3>
                <div className="flex flex-wrap gap-1.5">
                  {activeTools.map((tool, idx) => (
                    <span key={idx} className="inline-flex items-center gap-1 px-2 py-1 bg-white border border-gray-200 rounded-lg text-xs text-gray-600 font-mono shadow-sm">
                      <div className="w-1.5 h-1.5 bg-green-500 rounded-full" />
                      {tool}
                    </span>
                  ))}
                </div>
              </section>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Root page ────────────────────────────────────────────────────────────────

export default function ChatAgentPage() {
  const [view, setView] = useState<'list' | 'chat'>('list');
  const [pendingGoal, setPendingGoal] = useState<string | undefined>();

  const startNew = () => { setPendingGoal(undefined); setView('chat'); };
  const openSession = (session: Session) => { setPendingGoal(session.goal); setView('chat'); };
  const goBack = () => { setPendingGoal(undefined); setView('list'); };

  if (view === 'chat') return <ChatView onBack={goBack} initialGoal={pendingGoal} />;
  return <SessionList onNew={startNew} onSelect={openSession} />;
}

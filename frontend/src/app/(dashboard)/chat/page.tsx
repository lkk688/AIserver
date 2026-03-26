"use client";

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Paperclip, FileText, Code, Download, Save, Sparkles,
  Search, Bot, User, Wrench, X, CheckCircle2, AlertCircle,
  Zap, Clock, BarChart2, ExternalLink, Plus, Trash2, Settings,
  ChevronRight, ArrowLeft, ChevronDown, ChevronUp, Globe, Type, CirclePlay, Link2,
  Copy, Check, History, Mic, Loader2, Pause, Play, Volume2, VolumeX,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import clsx from 'clsx';
import { useVoiceInput } from '../audio/useVoiceInput';
import { useTts } from '../audio/useTts';
import ChatComposerShell from './ChatComposerShell';

// ─── Types ────────────────────────────────────────────────────────────────────

type MessageRole = 'user' | 'assistant' | 'system';

interface TurnHeaderBlock { type: 'turn_header'; turn: number; maxTurns: number }
interface ThinkBlock { type: 'think'; content: string }
interface ToolBlock { type: 'tool'; name: string }
interface TextBlock { type: 'text'; content: string }
interface FileBlock { type: 'file'; name: string; url: string; fileState?: 'saved' | 'updated' }
interface CompletionBlock {
  type: 'completion';
  success: boolean;
  stats?: { total_prompt_tokens?: number; total_completion_tokens?: number; total_elapsed_s?: number };
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
  asr_language: 'auto' | 'en' | 'zh';
  voice_send_mode: 'input' | 'direct';
  tts_enabled: boolean;
}

interface SessionFile {
  name: string;
  size: number;
  ext: string;
  url?: string;
  file_api_path?: string;
  download_api_path?: string;
  content?: string;
  local_path?: string;
  source_url?: string;
  resource_type?: string;
  resource_title?: string;
}

interface Session {
  task_id: string;
  goal: string;
  status: string;
  success?: boolean;
  started_at?: string;
  finished_at?: string;
  files?: SessionFile[];
  token_stats?: { total_prompt_tokens?: number; total_completion_tokens?: number; total_elapsed_s?: number };
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
  chat_history?: Array<{
    type: string;
    data?: string;
    turn?: number;
    max_turns?: number;
    name?: string;
    status?: string;
    url?: string;
    file_api_path?: string;
    download_api_path?: string;
    content?: string;
    detail?: string;
    success?: boolean;
    stats?: { total_prompt_tokens?: number; total_completion_tokens?: number; total_elapsed_s?: number };
    isFinalSummary?: boolean;
  }>;
  token_stats?: { total_prompt_tokens?: number; total_completion_tokens?: number; total_elapsed_s?: number };
  files?: SessionFile[];
}

type SiteKey = 'default' | 'site_health' | 'site_jwl';

interface SiteAgentUiConfig {
  title: string;
  subtitle: string;
  welcomeMessage: string;
  inputPlaceholder: string;
}

interface AiAgentWorkspaceProps {
  siteKey?: SiteKey;
  apiBaseUrl?: string;
  agentApiUrl?: string;
  embedded?: boolean;
}

interface AttachedFileItem {
  id: string;
  name: string;
  ext: string;
  url?: string;
  snapshot?: string;
  size?: number;
  sourceUrl?: string;
  resourceType?: string;
  displayTitle?: string;
  status: 'processing' | 'ready' | 'error';
}

interface AgentContextResource {
  name: string;
  title: string;
  url: string;
  source_url: string;
  resource_type: string;
  description: string;
  transcript: string;
  content: string;
  include_all: boolean;
}

const SITE_AGENT_CONFIG: Record<SiteKey, SiteAgentUiConfig> = {
  default: {
    title: 'AI Agent',
    subtitle: 'ReAct Agent with tool use',
    welcomeMessage: "Hello! I'm your AI Agent. I can search the web, write code, analyze data, and generate documents. What would you like me to do?",
    inputPlaceholder: 'Describe your task…',
  },
  site_health: {
    title: 'Health AI Agent',
    subtitle: 'Clinical workflow assistant',
    welcomeMessage: "Hello! I'm your Health AI Agent. I can generate care notes, summarize context, and create clinical documents. What should we work on?",
    inputPlaceholder: 'Describe your clinical task…',
  },
  site_jwl: {
    title: 'AI Agent',
    subtitle: 'Task automation assistant',
    welcomeMessage: "Hello! I'm your AI Agent. Tell me the task and I will plan and execute it for you.",
    inputPlaceholder: 'Describe your task…',
  },
};

function buildAuthHeaders(extra: Record<string, string> = {}) {
  if (typeof window === 'undefined') return extra;
  const token = localStorage.getItem('token') || localStorage.getItem('admin_token') || '';
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}

function compactText(value: string, limit: number) {
  const normalized = value.replace(/\s+/g, ' ').trim();
  if (!normalized) return '';
  if (normalized.length <= limit) return normalized;
  return `${normalized.slice(0, limit).trimEnd()}…`;
}

function cleanUnavailable(value: string) {
  const trimmed = value.trim();
  if (!trimmed) return '';
  if (/^\[?unavailable\]?$/i.test(trimmed)) return '';
  return trimmed;
}

function parseResourceText(raw: string) {
  const sourceUrl = cleanUnavailable((raw.match(/^\s*Source:\s*(.+)$/mi)?.[1] || '').trim());
  const title = cleanUnavailable((raw.match(/^\s*Title:\s*(.+)$/mi)?.[1] || '').trim());
  const descriptionBlock = raw.match(/^\s*Description:\s*([\s\S]*?)^\s*Transcript\s*\([^)]+\):/mi)?.[1] || '';
  const transcriptBlock = raw.match(/^\s*Transcript\s*\([^)]+\):\s*([\s\S]*)$/mi)?.[1] || '';
  return {
    sourceUrl,
    title,
    description: cleanUnavailable(descriptionBlock.trim()),
    transcript: cleanUnavailable(transcriptBlock.trim()),
  };
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
          {collapsed ? <><ChevronRight className="w-3 h-3" /> show</> : <><ChevronDown className="w-3 h-3" /> hide</>}
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

function FileBadge({ name, url, fileState = 'saved' }: { name: string; url: string; fileState?: 'saved' | 'updated' }) {
  const ext = name.split('.').pop()?.toLowerCase() ?? '';
  return (
    <div className="flex items-center gap-3 my-2 p-3 rounded-xl border border-indigo-200 bg-gradient-to-r from-indigo-50 to-blue-50 shadow-sm">
      <div className="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center shrink-0">
        <FileText className="w-4 h-4 text-white" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-indigo-900 truncate">{name}</p>
        <p className="text-xs text-indigo-500 capitalize">{ext || 'file'} · {fileState === 'updated' ? 'updated in workspace' : 'saved to workspace'}</p>
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
    await navigator.clipboard.writeText(children).catch(() => { });
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
          case 'think': return <ThinkBox key={i} content={block.content} />;
          case 'tool': return <ToolBadge key={i} name={block.name} />;
          case 'text': return (
            <div key={i} className="text-sm text-gray-800">
              <MarkdownContent content={block.content} />
            </div>
          );
          case 'file': return <FileBadge key={i} name={block.name} url={block.url} fileState={block.fileState} />;
          case 'completion': return <CompletionBadge key={i} {...block} />;
          default: return null;
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

function extractAssistantText(blocks: Block[]): string {
  return blocks
    .filter((block): block is TextBlock => block.type === 'text')
    .map(block => block.content)
    .join('\n')
    .trim();
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
  const fileByName = new Map<string, SessionFile>();
  const fileWriteCounts = new Map<string, number>();
  const fileWriteOrder: string[] = [];
  for (const file of data.files ?? []) {
    fileByName.set(file.name, file);
  }

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
        if (!fileWriteCounts.has(name)) {
          fileWriteOrder.push(name);
          fileWriteCounts.set(name, 1);
        } else {
          fileWriteCounts.set(name, (fileWriteCounts.get(name) || 0) + 1);
        }
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

  const finalFileNames: string[] = [];
  for (const name of fileWriteOrder) {
    if (finalFileNames.includes(name)) continue;
    finalFileNames.push(name);
  }
  for (const file of data.files ?? []) {
    if (!finalFileNames.includes(file.name)) {
      finalFileNames.push(file.name);
    }
  }
  for (const name of finalFileNames) {
    const file = fileByName.get(name);
    blocks.push({
      type: 'file',
      name,
      url: file?.download_api_path || file?.url || file?.file_api_path || '',
      fileState: (fileWriteCounts.get(name) || 0) > 1 ? 'updated' : 'saved',
    } as FileBlock);
  }

  if (data.status === 'done' || data.status === 'failed') {
    blocks.push({ type: 'completion', success: data.success ?? false, stats: data.token_stats } as CompletionBlock);
  }

  return blocks;
}

function reconstructBlocksFromChatHistory(data: TurnsApiResponse): Block[] {
  const blocks: Block[] = [];
  const events = data.chat_history ?? [];
  const fileWriteCounts = new Map<string, number>();
  const fileWriteUrls = new Map<string, string>();
  const fileWriteOrder: string[] = [];
  for (const event of events) {
    if (event.type === 'turn_start') {
      blocks.push({ type: 'turn_header', turn: Number(event.turn || 1), maxTurns: Number(event.max_turns || data.total_turns || 1) } as TurnHeaderBlock);
    } else if (event.type === 'think') {
      blocks.splice(0, blocks.length, ...appendToLastBlock(blocks, 'think', String(event.data ?? '')));
    } else if (event.type === 'message' || event.type === 'token' || event.type === 'delta') {
      blocks.splice(0, blocks.length, ...appendToLastBlock(blocks, 'text', String(event.data ?? '')));
    } else if (event.type === 'tool' && event.name) {
      const toolName = String(event.name);
      if (toolName !== 'finish_task' && event.status === 'started') {
        const last = blocks[blocks.length - 1];
        if (!(last && last.type === 'tool' && last.name === toolName)) {
          blocks.push({ type: 'tool', name: toolName } as ToolBlock);
        }
      }
    } else if (event.type === 'file_written' && event.name) {
      const name = String(event.name);
      if (!fileWriteCounts.has(name)) {
        fileWriteOrder.push(name);
        fileWriteCounts.set(name, 1);
      } else {
        fileWriteCounts.set(name, (fileWriteCounts.get(name) || 0) + 1);
      }
      fileWriteUrls.set(name, String(event.download_api_path || event.url || event.file_api_path || ''));
    } else if (event.type === 'error') {
      blocks.push({ type: 'text', content: `\n\n❌ **Error:** ${String(event.detail ?? 'Unknown error')}` } as TextBlock);
    } else if (event.type === 'done') {
      blocks.push({ type: 'completion', success: Boolean(event.success), stats: event.stats } as CompletionBlock);
    }
  }
  for (const name of fileWriteOrder) {
    blocks.push({
      type: 'file',
      name,
      url: fileWriteUrls.get(name) || '',
      fileState: (fileWriteCounts.get(name) || 0) > 1 ? 'updated' : 'saved',
    } as FileBlock);
  }
  return blocks;
}

function upsertFileBlock(blocks: Block[], nextBlock: FileBlock): Block[] {
  let found = false;
  const updatedBlocks = blocks.map((block) => {
    if (block.type !== 'file' || block.name !== nextBlock.name) return block;
    found = true;
    return {
      ...block,
      url: nextBlock.url || block.url,
      fileState: 'updated' as const,
    } as FileBlock;
  });
  if (found) return updatedBlocks;
  return [...updatedBlocks, nextBlock];
}

// ─── Settings Panel ───────────────────────────────────────────────────────────

const DOMAINS = ['general', 'software_eng', 'science', 'finance', 'medical', 'legal'];

function SettingsPanel({
  settings,
  onChange,
  audio,
}: {
  settings: AgentSettings;
  onChange: (s: AgentSettings) => void;
  audio: {
    asrType: 'native' | 'webgpu' | 'none';
    setAsrType: (mode: 'native' | 'webgpu' | 'none') => void;
    availableASR: Array<'native' | 'webgpu' | 'none'>;
    ttsType: 'native' | 'webgpu' | 'doubao' | 'none';
    setTtsType: (mode: 'native' | 'webgpu' | 'doubao' | 'none') => void;
    availableTTS: Array<'native' | 'webgpu' | 'doubao' | 'none'>;
    ttsVolume: number;
    setTtsVolume: (v: number) => void;
    ttsStatus: string;
    isSpeaking: boolean;
    setTtsEnabled: (enabled: boolean) => void;
    initAudio: () => void;
  };
}) {
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
        <div className="px-4 pb-4 space-y-3 bg-gray-50/80 border-t border-gray-100 max-h-72 overflow-y-auto">
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

          <div className="pt-2 border-t border-gray-200">
            <label className="block text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1">ASR Backend</label>
            <div className="flex gap-1">
              {audio.availableASR.includes('native') && (
                <button
                  onClick={() => audio.setAsrType('native')}
                  className={clsx(
                    "flex-1 py-1 text-xs rounded-md border transition-colors font-medium",
                    audio.asrType === 'native' ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-gray-600 border-gray-200 hover:border-indigo-300",
                  )}
                >
                  Native
                </button>
              )}
              {audio.availableASR.includes('webgpu') && (
                <button
                  onClick={() => audio.setAsrType('webgpu')}
                  className={clsx(
                    "flex-1 py-1 text-xs rounded-md border transition-colors font-medium",
                    audio.asrType === 'webgpu' ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-gray-600 border-gray-200 hover:border-indigo-300",
                  )}
                >
                  WebGPU
                </button>
              )}
            </div>
          </div>

          <div>
            <label className="block text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1">ASR Language</label>
            <select
              value={settings.asr_language}
              onChange={e => onChange({ ...settings, asr_language: e.target.value as AgentSettings['asr_language'] })}
              className="w-full px-2 py-1.5 text-xs border border-gray-200 rounded-md bg-white focus:ring-1 focus:ring-indigo-400 focus:border-indigo-400"
            >
              <option value="auto">Auto (Browser)</option>
              <option value="en">English</option>
              <option value="zh">Chinese (中文)</option>
            </select>
          </div>

          <div>
            <label className="block text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1">Voice Send Mode</label>
            <div className="flex gap-1">
              {(['input', 'direct'] as const).map(mode => (
                <button
                  key={mode}
                  onClick={() => onChange({ ...settings, voice_send_mode: mode })}
                  className={clsx(
                    "flex-1 py-1 text-xs rounded-md border transition-colors font-medium",
                    settings.voice_send_mode === mode ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-gray-600 border-gray-200 hover:border-indigo-300",
                  )}
                >
                  {mode === 'input' ? 'Input Only' : 'Direct Send'}
                </button>
              ))}
            </div>
          </div>

          <div className="pt-2 border-t border-gray-200">
            <div className="flex items-center justify-between mb-2">
              <div>
                <p className="text-xs font-medium text-gray-700">Text to Speech</p>
                <p className="text-[10px] text-gray-400">{settings.tts_enabled ? 'Enabled' : 'Disabled'}</p>
              </div>
              <button
                onClick={() => {
                  const next = !settings.tts_enabled;
                  onChange({ ...settings, tts_enabled: next });
                  audio.setTtsEnabled(next);
                  if (next) {
                    audio.initAudio();
                  } else {
                    audio.setTtsEnabled(false);
                  }
                }}
                className={clsx("relative inline-flex h-5 w-9 items-center rounded-full transition-colors", settings.tts_enabled ? "bg-indigo-600" : "bg-gray-200")}
              >
                <span className={clsx("inline-block h-3.5 w-3.5 rounded-full bg-white transition-transform shadow-sm", settings.tts_enabled ? "translate-x-4.5" : "translate-x-0.5")} />
              </button>
            </div>
            {settings.tts_enabled && (
              <div className="space-y-2">
                <label className="block text-[11px] font-semibold text-gray-500 uppercase tracking-wide">TTS Backend</label>
                <div className="flex gap-1">
                  {audio.availableTTS.includes('native') && (
                    <button
                      onClick={() => audio.setTtsType('native')}
                      className={clsx("flex-1 py-1 text-xs rounded-md border transition-colors font-medium",
                        audio.ttsType === 'native' ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-gray-600 border-gray-200 hover:border-indigo-300")}
                    >
                      Native
                    </button>
                  )}
                  {audio.availableTTS.includes('webgpu') && (
                    <button
                      onClick={() => audio.setTtsType('webgpu')}
                      className={clsx("flex-1 py-1 text-xs rounded-md border transition-colors font-medium",
                        audio.ttsType === 'webgpu' ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-gray-600 border-gray-200 hover:border-indigo-300")}
                    >
                      WebGPU
                    </button>
                  )}
                  {audio.availableTTS.includes('doubao') && (
                    <button
                      onClick={() => audio.setTtsType('doubao')}
                      className={clsx("flex-1 py-1 text-xs rounded-md border transition-colors font-medium",
                        audio.ttsType === 'doubao' ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-gray-600 border-gray-200 hover:border-indigo-300")}
                    >
                      Cloud
                    </button>
                  )}
                </div>
                <label className="block text-[11px] font-semibold text-gray-500 uppercase tracking-wide">
                  Volume <span className="font-bold text-indigo-600">{Math.round(audio.ttsVolume * 100)}%</span>
                </label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.1}
                  value={audio.ttsVolume}
                  onChange={e => audio.setTtsVolume(parseFloat(e.target.value))}
                  className="w-full accent-indigo-600"
                />
                {(audio.ttsStatus || audio.isSpeaking) && (
                  <p className="text-[10px] text-indigo-500">{audio.ttsStatus || 'Speaking...'}</p>
                )}
              </div>
            )}
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
  uiConfig,
  apiBaseUrl,
  agentApiUrl,
  embedded,
}: {
  onNew: () => void;
  onSelect: (session: Session) => void;
  uiConfig: SiteAgentUiConfig;
  apiBaseUrl: string;
  agentApiUrl: string;
  embedded?: boolean;
}) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    fetch(`${agentApiUrl}/agent/sessions?limit=50`, { headers: buildAuthHeaders() })
      .then(r => r.json())
      .then(d => setSessions(d.sessions ?? []))
      .catch(() => { })
      .finally(() => setLoading(false));
  }, [agentApiUrl]);

  useEffect(() => {
    const t = window.setTimeout(() => { load(); }, 0);
    return () => window.clearTimeout(t);
  }, [load]);

  const deleteSession = async (e: React.MouseEvent, taskId: string) => {
    e.stopPropagation();
    await fetch(`${agentApiUrl}/agent/sessions/${taskId}`, { method: 'DELETE', headers: buildAuthHeaders() });
    setSessions(prev => prev.filter(s => s.task_id !== taskId));
  };

  return (
    <div className={clsx(
      "flex flex-col bg-gray-50",
      embedded ? "h-full" : "-m-8 h-screen"
    )}>
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-8 py-5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center shadow-md">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-900">{uiConfig.title}</h1>
            <p className="text-xs text-gray-500">{uiConfig.subtitle}</p>
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
                  {(() => {
                    const promptTokens = session.token_stats?.total_prompt_tokens ?? 0;
                    const completionTokens = session.token_stats?.total_completion_tokens ?? 0;
                    const totalTokens = promptTokens + completionTokens;
                    const elapsed = session.token_stats?.total_elapsed_s ?? 0;
                    const speed = elapsed > 0 ? Math.round(completionTokens / elapsed) : 0;
                    if (totalTokens <= 0) return null;
                    return (
                      <div className="flex items-center gap-3 mt-2 text-[11px] text-gray-500">
                        <span className="inline-flex items-center gap-1"><BarChart2 className="w-3 h-3" /> {totalTokens.toLocaleString()} tokens</span>
                        <span className="inline-flex items-center gap-1"><Zap className="w-3 h-3" /> {speed} t/s</span>
                      </div>
                    );
                  })()}

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
  uiConfig,
  apiBaseUrl,
  agentApiUrl,
  embedded,
}: {
  onBack: () => void;
  taskId?: string;
  initialGoal?: string;
  uiConfig: SiteAgentUiConfig;
  apiBaseUrl: string;
  agentApiUrl: string;
  embedded?: boolean;
}) {
  const isHistory = !!taskId;
  const [currentTaskId, setCurrentTaskId] = useState(taskId ?? '');

  const [messages, setMessages] = useState<Message[]>(
    isHistory ? [] : [{
      id: '1', role: 'assistant', content: '',
      blocks: [{ type: 'text', content: uiConfig.welcomeMessage }],
    }]
  );
  const [historyLoading, setHistoryLoading] = useState(isHistory);
  const [input, setInput] = useState(isHistory ? '' : (initialGoal ?? ''));
  const [isVoiceMode, setIsVoiceMode] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamStatus, setStreamStatus] = useState('');
  const abortControllerRef = useRef<AbortController | null>(null);

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
  const [attachedFiles, setAttachedFiles] = useState<AttachedFileItem[]>([]);
  const [showUrlComposer, setShowUrlComposer] = useState(false);
  const [urlDraft, setUrlDraft] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [activeTools, setActiveTools] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Agent settings
  const [settings, setSettings] = useState<AgentSettings>({
    tool_strategy: 'hybrid',
    domain: 'general',
    max_turns: 15,
    parallel_thinking: true,
    asr_language: 'auto',
    voice_send_mode: 'input',
    tts_enabled: false,
  });
  const [browserLanguage, setBrowserLanguage] = useState<'en' | 'zh'>('en');

  useEffect(() => {
    if (typeof window !== 'undefined') {
      setBrowserLanguage(window.navigator.language.toLowerCase().startsWith('zh') ? 'zh' : 'en');
    }
  }, []);

  useEffect(() => {
    setCurrentTaskId(taskId ?? '');
  }, [taskId]);

  const effectiveAsrLanguage = settings.asr_language === 'auto' ? browserLanguage : settings.asr_language;
  const {
    isRecording,
    isProcessing,
    isReady,
    transcript,
    finalTranscript,
    error: voiceError,
    statusMessage,
    asrType,
    setAsrType,
    availableASR,
    startRecording,
    stopRecording,
    resetTranscript,
    requestPermission,
  } = useVoiceInput(effectiveAsrLanguage);
  const {
    setIsEnabled: setIsTtsEnabled,
    ttsType,
    setTtsType,
    availableTTS,
    speak,
    pause,
    resume,
    cancel,
    initAudio,
    isSpeaking,
    isPaused,
    statusMessage: ttsStatus,
    currentId,
    volume: ttsVolume,
    setVolume: setTtsVolume,
  } = useTts(effectiveAsrLanguage);

  useEffect(() => {
    setIsTtsEnabled(settings.tts_enabled);
  }, [settings.tts_enabled, setIsTtsEnabled]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    fetch(`${agentApiUrl}/agent/tools?strategy=native_all`)
      .then(r => r.json())
      .then(data => { if (Array.isArray(data.tools)) setActiveTools(data.tools.map((t: { name?: string; function?: { name?: string } }) => t.name || t.function?.name)); })
      .catch(() => { });
  }, [apiBaseUrl]);

  const resolveFileUrl = useCallback((raw?: string) => {
    if (!raw) return '';
    if (/^https?:\/\//i.test(raw)) return raw;
    if (raw.startsWith('/')) return `${apiBaseUrl}${raw}`;
    return `${apiBaseUrl}/${raw}`;
  }, [apiBaseUrl]);

  // Load history when viewing an existing session
  useEffect(() => {
    if (!taskId) return;
    setHistoryLoading(true);
    fetch(`${agentApiUrl}/agent/sessions/${taskId}/turns`, { headers: buildAuthHeaders() })
      .then(r => r.json())
      .then(async (data: TurnsApiResponse) => {
        setCurrentTaskId(data.task_id || taskId);
        const userMsg: Message = { id: 'user-0', role: 'user', content: data.goal };
        const turnBlocks = reconstructBlocksFromHistory(data);
        const hasRichTurnContent = turnBlocks.some((b) => b.type === 'turn_header' || b.type === 'think' || b.type === 'tool' || b.type === 'text');
        const assistantBlocksRaw = hasRichTurnContent ? turnBlocks : reconstructBlocksFromChatHistory(data);
        const assistantBlocks = assistantBlocksRaw.map((b) => (
          b.type === 'file' ? ({ ...b, url: resolveFileUrl(b.url) } as FileBlock) : b
        ));
        const assistantMsg: Message = {
          id: 'assistant-0', role: 'assistant', content: '',
          blocks: assistantBlocks,
        };
        setMessages([userMsg, assistantMsg]);
        const mappedFiles = (data.files ?? []).map((f, idx) => ({
          id: `${f.name}-${idx}`,
          name: f.name,
          ext: f.ext || f.name.split('.').pop() || 'file',
          url: resolveFileUrl(f.url || f.download_api_path || f.file_api_path),
          snapshot: (f.content || '').slice(0, 300),
          size: f.size,
          sourceUrl: f.source_url,
          resourceType: f.resource_type,
          displayTitle: f.resource_title || f.name,
          status: 'ready' as const,
        }));
        setAttachedFiles(mappedFiles);

        const mdFile = data.files?.find(f => f.ext === 'md' || f.ext === 'markdown' || f.name.toLowerCase().endsWith('.md') || f.name.toLowerCase().endsWith('.markdown'));
        if (mdFile) {
          if (mdFile.content) {
            setDocumentState(prev => ({ ...prev, title: mdFile.name, content: mdFile.content ?? '' }));
            setActiveTab('editor');
            return;
          }
          const resolved = resolveFileUrl(mdFile.url || mdFile.file_api_path);
          if (resolved) {
            try {
              const content = await fetch(resolved, {
                headers: ((resolved.startsWith(apiBaseUrl) || resolved.startsWith(agentApiUrl)) || resolved.startsWith(agentApiUrl)) ? buildAuthHeaders() : undefined,
              }).then(r => (r.ok ? r.text() : ''));
              if (content) {
                setDocumentState(prev => ({ ...prev, title: mdFile.name, content }));
                setActiveTab('editor');
                return;
              }
            } catch { }
          }
          try {
            const fileApi = `${agentApiUrl}/agent/sessions/${taskId}/files/content?name=${encodeURIComponent(mdFile.name)}`;
            const payload = await fetch(fileApi, { headers: buildAuthHeaders() }).then(r => r.ok ? r.json() : null);
            const text = typeof payload?.content === 'string' ? payload.content : '';
            if (text) {
              setDocumentState(prev => ({ ...prev, title: mdFile.name, content: text }));
              setActiveTab('editor');
              return;
            }
          } catch { }
          setDocumentState(prev => ({ ...prev, title: mdFile.name, content: prev.content }));
        }
      })
      .catch(console.error)
      .finally(() => setHistoryLoading(false));
  }, [agentApiUrl, apiBaseUrl, taskId, resolveFileUrl]);

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

  const saveDocument = useCallback(async () => {
    if (!currentTaskId) return;
    const name = (documentState.title || '').trim();
    if (!name) return;
    setStreamStatus('Saving document…');
    try {
      const res = await fetch(`${agentApiUrl}/agent/sessions/${currentTaskId}/files/content`, {
        method: 'PUT',
        headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ name, content: documentState.content }),
      });
      if (!res.ok) throw new Error('Failed to save document');
      const payload = await res.json().catch(() => null);
      if (payload?.name) {
        setDocumentState(prev => ({ ...prev, title: String(payload.name) }));
      }
    } catch (error) {
      console.error('Save document error:', error);
    } finally {
      setStreamStatus('');
    }
  }, [agentApiUrl, apiBaseUrl, currentTaskId, documentState.content, documentState.title]);

  const ensureTaskSession = useCallback(async () => {
    const existingTaskId = currentTaskId || taskId;
    if (existingTaskId) return existingTaskId;
    try {
      const response = await fetch(`${agentApiUrl}/agent/sessions/init`, {
        method: 'POST',
        headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ goal: (initialGoal || input || '').trim() }),
      });
      if (!response.ok) throw new Error('init failed');
      const payload = await response.json().catch(() => null);
      const createdTaskId = String(payload?.task_id || '').trim();
      if (!createdTaskId) throw new Error('missing task id');
      setCurrentTaskId(createdTaskId);
      return createdTaskId;
    } catch {
      setStreamStatus('Failed to initialize task session.');
      return '';
    }
  }, [agentApiUrl, apiBaseUrl, currentTaskId, initialGoal, input, taskId]);

  const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : [];
    e.target.value = '';
    if (files.length === 0) return;
    const effectiveTaskId = (currentTaskId || taskId) || await ensureTaskSession();
    if (!effectiveTaskId) {
      setStreamStatus('Unable to attach files right now.');
      return;
    }
    const pending = files.map((file, idx) => ({
      id: `${Date.now()}-${idx}-${file.name}`,
      name: file.name,
      ext: file.name.split('.').pop() || 'file',
      snapshot: 'Processing attachment...',
      size: file.size,
      status: 'processing' as const,
    }));
    setAttachedFiles(prev => [...prev, ...pending]);
    await Promise.all(pending.map(async (item, idx) => {
      const file = files[idx];
      try {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch(`${agentApiUrl}/agent/sessions/${effectiveTaskId}/files/upload`, {
          method: 'POST',
          headers: buildAuthHeaders(),
          body: formData,
        });
        if (!response.ok) {
          throw new Error('upload failed');
        }
        const payload = await response.json();
        if (!payload?.saved || !payload?.name) {
          throw new Error('upload failed');
        }
        const ext = String(payload.name).split('.').pop() || file.name.split('.').pop() || 'file';
        setAttachedFiles(prev => prev.map(existing => {
          if (existing.id !== item.id) return existing;
          return {
            ...existing,
            name: String(payload.name),
            ext,
            url: resolveFileUrl(String(payload.url || payload.download_api_path || payload.file_api_path || '')),
            snapshot: String(payload.snapshot || '').slice(0, 300),
            size: Number(payload.size || file.size || 0),
            sourceUrl: String(payload.source_url || ''),
            resourceType: String(payload.resource_type || ''),
            displayTitle: String(payload.resource_title || payload.name || ''),
            status: 'ready' as const,
          };
        }));
      } catch {
        setAttachedFiles(prev => prev.map(existing => {
          if (existing.id !== item.id) return existing;
          return { ...existing, snapshot: 'Failed to process attachment.', status: 'error' as const };
        }));
      }
    }));
  }, [agentApiUrl, apiBaseUrl, currentTaskId, ensureTaskSession, resolveFileUrl, taskId]);

  const submitAttachResources = useCallback(async () => {
    const urls = Array.from(new Set(
      urlDraft
        .split('\n')
        .flatMap(line => line.split(/[,\s]+/))
        .map(v => v.trim())
        .filter(v => /^https?:\/\//i.test(v))
    ));
    if (urls.length === 0) {
      setStreamStatus('Please enter at least one valid URL.');
      return;
    }
    const effectiveTaskId = (currentTaskId || taskId) || await ensureTaskSession();
    if (!effectiveTaskId) {
      setStreamStatus('Unable to attach links right now.');
      return;
    }
    setStreamStatus('Attaching resource…');
    const pending = urls.map((sourceUrl, idx) => ({
      id: `${Date.now()}-url-${idx}`,
      name: `resource_${idx + 1}.md`,
      ext: 'md',
      snapshot: 'Processing resource...',
      sourceUrl,
      resourceType: /(youtube\.com|youtu\.be)/i.test(sourceUrl) ? 'youtube' : 'web',
      displayTitle: sourceUrl,
      status: 'processing' as const,
    }));
    setAttachedFiles(prev => [...prev, ...pending]);
    setUrlDraft('');
    setShowUrlComposer(false);
    await Promise.all(pending.map(async (item) => {
      try {
        const response = await fetch(`${agentApiUrl}/agent/sessions/${effectiveTaskId}/resources/attach`, {
          method: 'POST',
          headers: buildAuthHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ url: item.sourceUrl }),
        });
        if (!response.ok) {
          throw new Error('attach failed');
        }
        const payload = await response.json();
        if (!payload?.saved || !payload?.name) {
          throw new Error('attach failed');
        }
        const ext = String(payload.name).split('.').pop() || 'md';
        setAttachedFiles(prev => prev.map(existing => {
          if (existing.id !== item.id) return existing;
          return {
            ...existing,
            name: String(payload.name),
            ext,
            url: resolveFileUrl(String(payload.url || payload.download_api_path || payload.file_api_path || '')),
            snapshot: String(payload.snapshot || '').slice(0, 300),
            size: Number(payload.size || 0),
            sourceUrl: String(payload.source_url || item.sourceUrl || ''),
            resourceType: String(payload.resource_type || item.resourceType || 'web'),
            displayTitle: String(payload.resource_title || payload.name || item.sourceUrl || ''),
            status: 'ready' as const,
          };
        }));
      } catch {
        setAttachedFiles(prev => prev.map(existing => {
          if (existing.id !== item.id) return existing;
          return { ...existing, snapshot: 'Failed to process URL resource.', status: 'error' as const };
        }));
      }
    }));
    setStreamStatus('');
  }, [agentApiUrl, apiBaseUrl, currentTaskId, ensureTaskSession, resolveFileUrl, taskId, urlDraft]);

  const handleAttachResource = useCallback(() => {
    setShowUrlComposer(prev => !prev);
  }, []);

  const removeAttachedFile = useCallback(async (item: AttachedFileItem) => {
    const effectiveTaskId = currentTaskId || taskId;
    if (!effectiveTaskId || !item.name) {
      setAttachedFiles(prev => prev.filter(existing => existing.id !== item.id));
      return;
    }
    try {
      const response = await fetch(
        `${agentApiUrl}/agent/sessions/${effectiveTaskId}/files?name=${encodeURIComponent(item.name)}`,
        { method: 'DELETE', headers: buildAuthHeaders() }
      );
      if (!response.ok) throw new Error('delete failed');
      setAttachedFiles(prev => prev.filter(existing => existing.id !== item.id));
    } catch {
      setStreamStatus('Failed to delete attached resource.');
    }
  }, [agentApiUrl, apiBaseUrl, currentTaskId, taskId]);

  const buildContextResources = useCallback(async (effectiveTaskId?: string): Promise<AgentContextResource[]> => {
    const readyFiles = attachedFiles.filter(file => file.status === 'ready');
    const resources = await Promise.all(readyFiles.map(async (file) => {
      let content = '';
      if (effectiveTaskId && file.name) {
        try {
          const fileApi = `${agentApiUrl}/agent/sessions/${effectiveTaskId}/files/content?name=${encodeURIComponent(file.name)}`;
          const payload = await fetch(fileApi, { headers: buildAuthHeaders() }).then(r => r.ok ? r.json() : null);
          if (typeof payload?.content === 'string') {
            content = payload.content;
          }
        } catch { }
      }
      const parsed = parseResourceText(content);
      const title = parsed.title || file.displayTitle || file.name;
      const sourceUrl = parsed.sourceUrl || file.sourceUrl || '';
      const fallbackUrl = sourceUrl || file.url || '';
      const description = compactText(parsed.description || file.snapshot || '', 700);
      const transcript = compactText(parsed.transcript, 1200);
      const normalizedContent = compactText(content, 2000);
      const inlineText = `${description}\n${transcript}\n${normalizedContent}`.trim();
      const includeAll = inlineText.length > 0 && inlineText.length <= 1800;
      return {
        name: file.name,
        title,
        url: fallbackUrl,
        source_url: sourceUrl || fallbackUrl,
        resource_type: file.resourceType || 'file',
        description,
        transcript: includeAll ? transcript : compactText(transcript, 260),
        content: includeAll ? normalizedContent : compactText(normalizedContent, 260),
        include_all: includeAll,
      };
    }));
    return resources.filter(resource => Boolean(resource.url || resource.description || resource.content || resource.transcript));
  }, [agentApiUrl, apiBaseUrl, attachedFiles]);

  const updateBlocks = useCallback((updater: (prev: Block[]) => Block[]) => {
    setMessages(prev => {
      const last = prev[prev.length - 1];
      if (last?.role !== 'assistant') return prev;
      return [...prev.slice(0, -1), { ...last, blocks: updater(last.blocks ?? []) }];
    });
  }, []);

  const handleStop = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsStreaming(false);
    setStreamStatus('');
  }, []);

  const handleSend = useCallback(async (e?: React.FormEvent, forcedInput?: string) => {
    e?.preventDefault();
    const outgoingInput = (forcedInput ?? input).trim();
    if (!outgoingInput || isStreaming) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: outgoingInput };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsStreaming(true);
    const assistantId = (Date.now() + 1).toString();
    setMessages(prev => [...prev, { id: assistantId, role: 'assistant', content: '', blocks: [] }]);
    let assistantTextBuffer = '';

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const effectiveTaskId = currentTaskId || taskId || undefined;
      const contextResources = await buildContextResources(effectiveTaskId);
      const response = await fetch(`${agentApiUrl}/agent/stream`, {
        method: 'POST',
        signal: controller.signal,
        headers: buildAuthHeaders({
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
          'Cache-Control': 'no-cache',
        }),
        body: JSON.stringify({
          task_id: effectiveTaskId,
          goal: userMsg.content,
          context_resources: contextResources,
          backend: 'vllm',
          tool_strategy: settings.tool_strategy,
          domain: settings.domain,
          max_turns: settings.max_turns,
          enable_turn_limits: true,
          parallel_thinking: settings.parallel_thinking,
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
        const events = buffer.split('\n\n');
        buffer = events.pop() ?? '';

        for (const eventChunk of events) {
          const dataLines = eventChunk
            .split('\n')
            .map(line => line.trim())
            .filter(line => line.startsWith('data: '))
            .map(line => line.slice(6));
          if (dataLines.length === 0) continue;
          const raw = dataLines.join('\n').trim();
          if (!raw || raw === '[DONE]') continue;
          try {
            const event = JSON.parse(raw);
            if (event.type === 'start') {
              if (event.task_id) setCurrentTaskId(String(event.task_id));
            } else if (event.type === 'turn_start') {
              setStreamStatus(`Turn ${event.turn}/${event.max_turns}`);
              updateBlocks(b => [...b, { type: 'turn_header', turn: event.turn, maxTurns: event.max_turns } as TurnHeaderBlock]);
            } else if (event.type === 'think') {
              updateBlocks(b => appendToLastBlock(b, 'think', event.data ?? ''));
            } else if (event.type === 'message' || event.type === 'token') {
              const chunk = event.data ?? '';
              assistantTextBuffer += chunk;
              updateBlocks(b => appendToLastBlock(b, 'text', chunk));
            } else if (event.type === 'tool' && event.status === 'started' && event.name) {
              setStreamStatus(`${event.name}…`);
              setActiveTools(prev => prev.includes(event.name) ? prev : [...prev, event.name]);
              updateBlocks(b => [...b, { type: 'tool', name: event.name } as ToolBlock]);
            } else if (event.type === 'file_written') {
              const resolvedUrl = resolveFileUrl(event.url || event.download_api_path || event.file_api_path);
              updateBlocks(b => upsertFileBlock(b, { type: 'file', name: event.name, url: resolvedUrl, fileState: 'saved' } as FileBlock));
              if (event.name) {
                const writtenName = String(event.name);
                const writtenExt = writtenName.split('.').pop() || 'file';
                const writtenSnapshot = typeof event.content === 'string' ? event.content.slice(0, 300) : '';
                setAttachedFiles(prev => {
                  const next = prev.filter(item => item.name !== writtenName);
                  return [...next, {
                    id: `${Date.now()}-stream-${writtenName}`,
                    name: writtenName,
                    ext: writtenExt,
                    url: resolvedUrl,
                    snapshot: writtenSnapshot,
                    sourceUrl: String(event.source_url || ''),
                    resourceType: String(event.resource_type || ''),
                    displayTitle: String(event.resource_title || writtenName),
                    status: 'ready' as const,
                  }];
                });
              }
              const fileName = String(event.name || '');
              const lowerName = fileName.toLowerCase();
              const isTextDoc = ['.md', '.markdown', '.txt', '.rst', '.log', '.csv'].some(ext => lowerName.endsWith(ext));
              if (isTextDoc) {
                const inline = event.content;
                if (inline) {
                  setDocumentState(prev => ({ ...prev, title: fileName || prev.title, content: inline }));
                  setActiveTab('editor');
                } else if (resolvedUrl) {
                  fetch(resolvedUrl, {
                    headers: ((resolvedUrl.startsWith(apiBaseUrl) || resolvedUrl.startsWith(agentApiUrl)) || resolvedUrl.startsWith(agentApiUrl)) ? buildAuthHeaders() : undefined,
                  })
                    .then(r => (r.ok ? r.text() : ''))
                    .then(md => {
                      if (md) {
                        setDocumentState(prev => ({ ...prev, title: fileName || prev.title, content: md }));
                        setActiveTab('editor');
                      } else if (currentTaskId) {
                        const fileApi = `${agentApiUrl}/agent/sessions/${currentTaskId}/files/content?name=${encodeURIComponent(fileName)}`;
                        fetch(fileApi, { headers: buildAuthHeaders() })
                          .then(r => r.ok ? r.json() : null)
                          .then(payload => {
                            const text = typeof payload?.content === 'string' ? payload.content : '';
                            if (text) {
                              setDocumentState(prev => ({ ...prev, title: fileName || prev.title, content: text }));
                              setActiveTab('editor');
                            }
                          })
                          .catch(() => { });
                      }
                    })
                    .catch(() => { });
                } else {
                  if (!currentTaskId) {
                    setDocumentState(prev => ({ ...prev, title: fileName || prev.title, content: prev.content }));
                    setActiveTab('editor');
                    continue;
                  }
                  const fileApi = `${agentApiUrl}/agent/sessions/${currentTaskId}/files/content?name=${encodeURIComponent(fileName)}`;
                  fetch(fileApi, { headers: buildAuthHeaders() })
                    .then(r => r.ok ? r.json() : null)
                    .then(payload => {
                      const text = typeof payload?.content === 'string' ? payload.content : '';
                      if (text) {
                        setDocumentState(prev => ({ ...prev, title: fileName || prev.title, content: text }));
                        setActiveTab('editor');
                      } else {
                        setDocumentState(prev => ({ ...prev, title: fileName || prev.title, content: prev.content }));
                        setActiveTab('editor');
                      }
                    })
                    .catch(() => {
                      setDocumentState(prev => ({ ...prev, title: fileName || prev.title, content: prev.content }));
                      setActiveTab('editor');
                    });
                }
              }
            } else if (event.type === 'done') {
              if (event.task_id) setCurrentTaskId(String(event.task_id));
              setStreamStatus('');
              updateBlocks(b => [...b, { type: 'completion', success: event.success, stats: event.stats } as CompletionBlock]);
              if (settings.tts_enabled && assistantTextBuffer.trim() && Boolean(event.isFinalSummary)) {
                initAudio();
                speak(assistantTextBuffer, false, assistantId);
              }
            } else if (event.type === 'error') {
              setStreamStatus('');
              updateBlocks(b => [...b, { type: 'text', content: `\n\n❌ **Error:** ${event.detail ?? 'Unknown error'}` } as TextBlock]);
            }
          } catch { /* incomplete chunk */ }
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        // user stopped the task — not an error
      } else {
        console.error('Chat stream error:', err);
      }
    } finally {
      abortControllerRef.current = null;
      setIsStreaming(false);
      setStreamStatus('');
    }
  }, [
    input,
    isStreaming,
    apiBaseUrl,
    settings.tool_strategy,
    settings.domain,
    settings.max_turns,
    settings.tts_enabled,
    currentTaskId,
    updateBlocks,
    initAudio,
    speak,
    resolveFileUrl,
    buildContextResources,
    taskId,
  ]);

  useEffect(() => {
    const text = finalTranscript.trim();
    if (!text) return;
    if (settings.voice_send_mode === 'direct' && !isStreaming) {
      void handleSend(undefined, text);
    } else {
      setInput(prev => (prev ? `${prev} ${text}` : text));
    }
    resetTranscript();
  }, [finalTranscript, settings.voice_send_mode, isStreaming, resetTranscript, handleSend]);

  return (
    <div className={clsx(
      "flex flex-col bg-white overflow-hidden",
      embedded ? "h-full" : "-m-8 h-screen"
    )}>
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
      <div ref={containerRef} className="flex-1 flex overflow-hidden min-h-0">
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
            {messages.map(msg => {
              const assistantText = msg.role === 'assistant' ? extractAssistantText(msg.blocks ?? []) : '';
              const speakingThis = currentId === msg.id && isSpeaking;
              return (
                <div key={msg.id} className={clsx("flex max-w-[90%]", msg.role === 'user' ? "ml-auto" : "mr-auto")}>
                  {msg.role !== 'user' && (
                    <div className="w-7 h-7 rounded-full bg-indigo-100 flex items-center justify-center mr-2.5 shrink-0 mt-1">
                      <Bot className="w-4 h-4 text-indigo-600" />
                    </div>
                  )}
                  <div className={clsx(
                    "rounded-2xl px-4 py-3 min-w-0 flex-1 text-sm select-text",
                    msg.role === 'user'
                      ? "bg-indigo-600 text-white rounded-tr-sm shadow-sm"
                      : "bg-white border border-gray-200 text-gray-800 rounded-tl-sm shadow-sm"
                  )}>
                    {msg.role === 'user'
                      ? <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                      : (
                        <>
                          <AssistantMessage blocks={msg.blocks ?? [{ type: 'text', content: msg.content }]} />
                          <div className="mt-2 flex items-center gap-1.5">
                            <button
                              type="button"
                              onClick={() => {
                                if (!assistantText) return;
                                initAudio();
                                if (speakingThis) {
                                  if (isPaused) {
                                    resume();
                                  } else {
                                    pause();
                                  }
                                  return;
                                }
                                speak(assistantText, true, msg.id);
                              }}
                              disabled={!assistantText}
                              className="inline-flex items-center gap-1 px-2 py-1 rounded-md border border-gray-200 text-[10px] text-gray-600 hover:bg-gray-50 disabled:opacity-40"
                            >
                              {speakingThis ? (
                                isPaused ? <Play className="w-3 h-3" /> : <Pause className="w-3 h-3" />
                              ) : (
                                <Volume2 className="w-3 h-3" />
                              )}
                              {speakingThis ? (isPaused ? 'Resume' : 'Pause') : 'Play'}
                            </button>
                            {speakingThis && (
                              <button
                                type="button"
                                onClick={() => cancel()}
                                className="inline-flex items-center gap-1 px-2 py-1 rounded-md border border-gray-200 text-[10px] text-gray-600 hover:bg-gray-50"
                              >
                                <VolumeX className="w-3 h-3" /> Stop
                              </button>
                            )}
                          </div>
                        </>
                      )
                    }
                  </div>
                  {msg.role === 'user' && (
                    <div className="w-7 h-7 rounded-full bg-gray-200 flex items-center justify-center ml-2.5 shrink-0 mt-1">
                      <User className="w-4 h-4 text-gray-600" />
                    </div>
                  )}
                </div>
              )
            })}
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
            <ChatComposerShell
              modes={[
                { key: 'text', icon: <Type size={15} />, label: 'Text', active: !isVoiceMode, onClick: () => setIsVoiceMode(false) },
                { key: 'voice', icon: <Mic size={15} />, label: 'Voice', active: isVoiceMode, onClick: () => { setIsVoiceMode(true); requestPermission(); } },
              ]}
              onSend={() => { if (!isVoiceMode) void handleSend(); }}
              onStop={handleStop}
              isStreaming={isStreaming}
              sendDisabled={isVoiceMode || !input.trim() || isStreaming}
              rightSlot={(
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="p-2 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-100 transition-colors"
                  title="Attach files"
                >
                  <Paperclip className="w-4 h-4" />
                </button>
              )}
            >
              <input type="file" ref={fileInputRef} className="hidden" onChange={handleFileChange} multiple />
              {isVoiceMode ? (
                <button
                  type="button"
                  className={clsx(
                    "w-full min-h-[84px] rounded-xl border text-sm font-medium transition-colors",
                    isRecording ? "bg-gray-200 border-gray-300 text-gray-800" :
                      isProcessing ? "bg-gray-100 border-gray-200 text-gray-400 cursor-wait" :
                        "bg-white border-gray-200 hover:bg-gray-50 text-gray-700"
                  )}
                  onMouseDown={e => { e.preventDefault(); if (settings.tts_enabled) initAudio(); if (!isProcessing) startRecording(); }}
                  onMouseUp={e => { e.preventDefault(); stopRecording(); }}
                  onMouseLeave={e => { e.preventDefault(); if (isRecording) stopRecording(); }}
                  onTouchStart={e => { e.preventDefault(); if (settings.tts_enabled) initAudio(); if (!isProcessing) startRecording(); }}
                  onTouchEnd={e => { e.preventDefault(); stopRecording(); }}
                  disabled={isStreaming}
                >
                  {isProcessing ? (
                    <span className="inline-flex items-center gap-2">
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      {statusMessage || (isReady ? 'Processing...' : 'Loading Model...')}
                    </span>
                  ) : isRecording ? 'Release to Send' : 'Hold to Talk'}
                </button>
              ) : (
                <textarea
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && e.shiftKey) { e.preventDefault(); void handleSend(); } }}
                  placeholder={uiConfig.inputPlaceholder}
                  className="w-full min-h-[84px] max-h-[220px] bg-white border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-200 focus:border-blue-300 resize-y p-3 text-sm text-gray-800"
                />
              )}
              {(voiceError || transcript || ttsStatus) && (
                <div className="mt-1 px-1 text-[10px] text-gray-500 truncate">
                  {voiceError || transcript || ttsStatus}
                </div>
              )}
            </ChatComposerShell>
            <SettingsPanel
              settings={settings}
              onChange={setSettings}
              audio={{
                asrType,
                setAsrType,
                availableASR,
                ttsType,
                setTtsType,
                availableTTS,
                ttsVolume,
                setTtsVolume,
                ttsStatus,
                isSpeaking,
                setTtsEnabled: setIsTtsEnabled,
                initAudio,
              }}
            />
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
                  {/* Audio read-aloud button — same TTS hook used by chat messages */}
                  {documentState.content && (() => {
                    const docId = 'doc-panel';
                    const speakingDoc = currentId === docId && isSpeaking;
                    return (
                      <>
                        <button
                          type="button"
                          onClick={() => {
                            initAudio();
                            if (speakingDoc) {
                              isPaused ? resume() : pause();
                            } else {
                              speak(documentState.content, true, docId);
                            }
                          }}
                          title={speakingDoc ? (isPaused ? 'Resume reading' : 'Pause reading') : 'Read aloud'}
                          className="p-1 text-gray-400 hover:text-indigo-600 rounded transition-colors"
                        >
                          {speakingDoc ? (
                            isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />
                          ) : (
                            <Volume2 className="w-3.5 h-3.5" />
                          )}
                        </button>
                        {speakingDoc && (
                          <button
                            type="button"
                            onClick={() => cancel()}
                            title="Stop reading"
                            className="p-1 text-gray-400 hover:text-red-500 rounded transition-colors"
                          >
                            <VolumeX className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </>
                    );
                  })()}
                  <button onClick={saveDocument} disabled={!currentTaskId} className={clsx("p-1 rounded transition-colors", currentTaskId ? "text-gray-400 hover:text-indigo-600" : "text-gray-300 cursor-not-allowed")}><Save className="w-3.5 h-3.5" /></button>
                  <button onClick={downloadDocument} className="p-1 text-gray-400 hover:text-indigo-600 rounded transition-colors"><Download className="w-3.5 h-3.5" /></button>
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
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="mb-3 w-full py-2 border border-dashed border-gray-300 rounded-xl text-xs text-gray-500 hover:border-indigo-300 hover:text-indigo-500 transition-colors bg-white"
                >
                  + Attach files
                </button>
                <button
                  onClick={handleAttachResource}
                  className="mb-3 w-full py-2 border border-dashed border-gray-300 rounded-xl text-xs text-gray-500 hover:border-indigo-300 hover:text-indigo-500 transition-colors bg-white"
                >
                  + Attach link/resource
                </button>
                {showUrlComposer && (
                  <div className="mb-3 rounded-xl border border-indigo-200 bg-indigo-50/50 p-3 space-y-2">
                    <div className="text-xs text-indigo-900 font-medium">Attach web pages or videos</div>
                    <div className="text-[11px] text-indigo-700">
                      Paste one or multiple URLs. Use one per line, or separate multiple URLs with spaces or commas.
                    </div>
                    <textarea
                      value={urlDraft}
                      onChange={(e) => setUrlDraft(e.target.value)}
                      placeholder="https://example.com/article
https://www.youtube.com/watch?v=..."
                      className="w-full min-h-[88px] rounded-lg border border-indigo-200 bg-white p-2 text-xs text-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-200"
                    />
                    <div className="flex items-center justify-end gap-2">
                      <button
                        onClick={() => { setShowUrlComposer(false); setUrlDraft(''); }}
                        className="px-2.5 py-1 rounded-lg border border-gray-300 text-[11px] text-gray-600 hover:bg-gray-50"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={submitAttachResources}
                        className="px-2.5 py-1 rounded-lg bg-indigo-600 text-[11px] text-white hover:bg-indigo-500"
                      >
                        Add URLs
                      </button>
                    </div>
                  </div>
                )}
                {attachedFiles.length > 0 ? (
                  <div className="space-y-1.5">
                    {attachedFiles.map((file, idx) => (
                      <div key={file.id || idx} className="p-2.5 bg-white border border-gray-200 rounded-xl group hover:border-indigo-200 transition-colors">
                        <div className="flex items-center gap-2">
                          {/youtube/i.test(file.resourceType || '') || /(youtube\.com|youtu\.be)/i.test(file.sourceUrl || '') ? (
                            <CirclePlay className="w-4 h-4 text-red-500 shrink-0" />
                          ) : file.sourceUrl ? (
                            <Globe className="w-4 h-4 text-sky-500 shrink-0" />
                          ) : (
                            <FileText className="w-4 h-4 text-indigo-500 shrink-0" />
                          )}
                          <div className="min-w-0 flex-1">
                            <div className="text-xs text-gray-700 truncate">{file.displayTitle || file.name}</div>
                            {file.sourceUrl && (
                              <a
                                href={file.sourceUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 text-[10px] text-indigo-600 hover:text-indigo-700 truncate max-w-full"
                              >
                                <Link2 className="w-2.5 h-2.5 shrink-0" />
                                <span className="truncate">{file.sourceUrl}</span>
                              </a>
                            )}
                            {file.snapshot && (
                              <div
                                className="mt-1 text-[10px] text-gray-500 whitespace-pre-wrap overflow-hidden"
                                style={{ display: '-webkit-box', WebkitLineClamp: 4, WebkitBoxOrient: 'vertical' }}
                              >
                                {file.snapshot}
                              </div>
                            )}
                          </div>
                          {file.status === 'processing' ? (
                            <div className="px-1.5 py-0.5 rounded border border-indigo-200 text-[10px] text-indigo-600 inline-flex items-center gap-1">
                              <Loader2 className="w-3 h-3 animate-spin" />
                              Processing
                            </div>
                          ) : file.url ? (
                            <a
                              href={file.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="px-1.5 py-0.5 rounded border border-indigo-200 text-[10px] text-indigo-600 hover:bg-indigo-50"
                            >
                              Open
                            </a>
                          ) : (
                            <span className="px-1.5 py-0.5 rounded border border-red-200 text-[10px] text-red-600">Failed</span>
                          )}
                          <button onClick={e => { e.stopPropagation(); void removeAttachedFile(file); }}
                            className="p-1 hover:bg-red-50 text-gray-300 hover:text-red-500 rounded transition-colors opacity-0 group-hover:opacity-100">
                            <X className="w-3 h-3" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="py-4 border-2 border-dashed border-gray-200 rounded-xl text-xs text-gray-400 text-center bg-white">
                    No files attached yet
                  </div>
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

export function AiAgentWorkspace({ siteKey = 'default', apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || '/api/v1', agentApiUrl = process.env.NEXT_PUBLIC_AGENT_API_URL || '', embedded = false }: AiAgentWorkspaceProps) {
  const uiConfig = SITE_AGENT_CONFIG[siteKey] ?? SITE_AGENT_CONFIG.default;
  const [view, setView] = useState<'list' | 'chat'>('list');
  const [pendingGoal, setPendingGoal] = useState<string | undefined>();
  const [selectedTaskId, setSelectedTaskId] = useState<string | undefined>();

  const startNew = () => {
    setPendingGoal(undefined);
    setSelectedTaskId(undefined);
    setView('chat');
  };
  const openSession = (session: Session) => {
    setPendingGoal(session.goal);
    setSelectedTaskId(session.task_id);
    setView('chat');
  };
  const goBack = () => {
    setPendingGoal(undefined);
    setSelectedTaskId(undefined);
    setView('list');
  };

  if (view === 'chat') {
    return (
      <ChatView
        onBack={goBack}
        taskId={selectedTaskId}
        initialGoal={pendingGoal}
        uiConfig={uiConfig}
        apiBaseUrl={apiBaseUrl}
        agentApiUrl={agentApiUrl}
        embedded={embedded}
      />
    );
  }
  return <SessionList onNew={startNew} onSelect={openSession} uiConfig={uiConfig} apiBaseUrl={apiBaseUrl}
        agentApiUrl={agentApiUrl} embedded={embedded} />;
}

export default function ChatAgentPage() {
  return <AiAgentWorkspace />;
}

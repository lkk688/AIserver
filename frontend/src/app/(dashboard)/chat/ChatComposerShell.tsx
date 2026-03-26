"use client";

import React from "react";
import { Send, Square } from "lucide-react";
import clsx from "clsx";

interface ComposerMode {
  key: string;
  label: string;
  icon: React.ReactNode;
  active: boolean;
  onClick: () => void;
}

interface ChatComposerShellProps {
  children: React.ReactNode;
  modes?: ComposerMode[];
  onSend: () => void;
  onStop?: () => void;
  isStreaming?: boolean;
  sendTitle?: string;
  sendDisabled?: boolean;
  rightSlot?: React.ReactNode;
  className?: string;
}

export default function ChatComposerShell({
  children,
  modes = [],
  onSend,
  onStop,
  isStreaming = false,
  sendTitle = "Send (Shift+Enter)",
  sendDisabled = false,
  rightSlot,
  className,
}: ChatComposerShellProps) {
  return (
    <div
      className={clsx("bg-white border-t border-gray-100 flex-shrink-0 z-20", className)}
      style={{ paddingBottom: "max(0rem, env(safe-area-inset-bottom))" }}
    >
      <div className="px-4 pt-3 pb-1">
        {children}
      </div>
      <div className="flex items-center justify-between px-4 py-2 pr-4">
        <div className="flex items-center gap-1">
          {modes.map(({ key, icon, label, active, onClick }) => (
            <button
              key={key}
              onClick={onClick}
              className={clsx(
                "flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors",
                active ? "bg-blue-600 text-white shadow-sm" : "text-gray-500 hover:bg-gray-100"
              )}
            >
              {icon}
              {label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          {rightSlot}
          {isStreaming ? (
            <button
              onClick={onStop}
              title="Stop task"
              className="p-2.5 bg-red-500 text-white rounded-full shadow-md hover:bg-red-600 transition-all"
            >
              <Square size={18} />
            </button>
          ) : (
            <button
              onClick={onSend}
              title={sendTitle}
              disabled={sendDisabled}
              className="p-2.5 bg-blue-600 text-white rounded-full shadow-md hover:bg-blue-700 hover:shadow-blue-500/30 transition-all disabled:bg-blue-300 disabled:cursor-not-allowed"
            >
              <Send size={18} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

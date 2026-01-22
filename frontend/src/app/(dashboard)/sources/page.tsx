"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchSources, createSource, scanSource, Source } from "@/lib/api";
import { useState } from "react";
import { FolderPlus, RefreshCw, Loader2 } from "lucide-react";
import clsx from "clsx";

export default function SourcesPage() {
  const queryClient = useQueryClient();
  const { data: sources, isLoading } = useQuery({ queryKey: ["sources"], queryFn: fetchSources });
  const [isCreating, setIsCreating] = useState(false);
  
  // Form State
  const [name, setName] = useState("");
  const [path, setPath] = useState("");

  const createMutation = useMutation({
    mutationFn: createSource,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      setIsCreating(false);
      setName("");
      setPath("");
    },
  });

  const scanMutation = useMutation({
    mutationFn: scanSource,
    onSuccess: () => {
      // Maybe show toast? For now just log
      console.log("Scan started");
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    createMutation.mutate({ name, path });
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Sources</h1>
        <button
          onClick={() => setIsCreating(!isCreating)}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg flex items-center gap-2 hover:bg-blue-700 transition"
        >
          <FolderPlus className="w-4 h-4" />
          Add Source
        </button>
      </div>

      {isCreating && (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                placeholder="My Documents"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Path</label>
              <input
                type="text"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                placeholder="/absolute/path/to/docs"
                required
              />
            </div>
            <div className="flex gap-2 justify-end">
              <button
                type="button"
                onClick={() => setIsCreating(false)}
                className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={createMutation.isPending}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg flex items-center gap-2 disabled:opacity-50"
              >
                {createMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
                Create
              </button>
            </div>
          </form>
        </div>
      )}

      {isLoading ? (
        <div className="flex justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
        </div>
      ) : (
        <div className="grid gap-4">
          {sources?.map((source) => (
            <div key={source.id} className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 flex justify-between items-center">
              <div>
                <h3 className="font-semibold text-lg">{source.name}</h3>
                <p className="text-gray-500 text-sm font-mono mt-1">{source.path}</p>
              </div>
              <button
                onClick={() => scanMutation.mutate(source.id)}
                disabled={scanMutation.isPending}
                className="text-gray-600 hover:text-blue-600 p-2 rounded-lg hover:bg-blue-50 transition"
                title="Trigger Scan"
              >
                <RefreshCw className={clsx("w-5 h-5", scanMutation.isPending && "animate-spin")} />
              </button>
            </div>
          ))}
          {sources?.length === 0 && (
            <div className="text-center py-12 text-gray-500">
              No sources found. Add one to get started.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

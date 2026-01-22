"use client";

import { useMutation } from "@tanstack/react-query";
import { search, SearchResult } from "@/lib/api";
import { useState } from "react";
import { Search as SearchIcon, Loader2, FileText, ExternalLink } from "lucide-react";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);

  const searchMutation = useMutation({
    mutationFn: (q: string) => search(q),
    onSuccess: (data) => {
      setResults(data);
    },
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      searchMutation.mutate(query);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-gray-900">Local Search</h1>
        <p className="text-gray-500 text-lg">Search through your indexed documents instantly</p>
      </div>

      <form onSubmit={handleSearch} className="relative">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="What are you looking for?"
          className="w-full px-6 py-4 text-lg border-2 border-gray-200 rounded-2xl shadow-sm focus:border-blue-500 focus:ring-4 focus:ring-blue-50 outline-none transition-all pl-14"
          autoFocus
        />
        <SearchIcon className="absolute left-5 top-1/2 -translate-y-1/2 text-gray-400 w-6 h-6" />
        <button
          type="submit"
          disabled={searchMutation.isPending || !query.trim()}
          className="absolute right-3 top-1/2 -translate-y-1/2 bg-blue-600 text-white p-2 rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 transition-colors"
        >
          {searchMutation.isPending ? <Loader2 className="w-5 h-5 animate-spin" /> : <SearchIcon className="w-5 h-5" />}
        </button>
      </form>

      <div className="space-y-6">
        {results.map((result) => (
          <div key={result.chunk_id} className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow group">
            <div className="flex justify-between items-start mb-2">
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <FileText className="w-4 h-4" />
                <span className="truncate max-w-md">{result.metadata.title || result.metadata.uri}</span>
              </div>
              <span className="text-xs font-mono bg-blue-50 text-blue-700 px-2 py-1 rounded-full">
                {Math.round(result.score * 100)}%
              </span>
            </div>
            
            <p className="text-gray-800 leading-relaxed">
              {result.text}
            </p>

            <div className="mt-4 pt-4 border-t border-gray-50 flex justify-end opacity-0 group-hover:opacity-100 transition-opacity">
              <a 
                href={result.metadata.uri} 
                target="_blank" 
                rel="noreferrer"
                className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1"
              >
                Open Document <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        ))}

        {searchMutation.isSuccess && results.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            No results found for "{query}"
          </div>
        )}
      </div>
    </div>
  );
}

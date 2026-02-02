"use client";

import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { fetchDocument, fetchDocumentChunks, fetchDocumentSections, Chunk, Section } from "@/lib/api";
import { Loader2 } from "lucide-react";

export default function DocumentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = typeof params?.id === "string" ? params.id : Array.isArray(params?.id) ? params.id[0] : "";

  const {
    data: document,
    isLoading: docLoading,
    isError: docError,
  } = useQuery({
    queryKey: ["document", id],
    queryFn: () => fetchDocument(id),
    enabled: !!id,
  });

  const {
    data: chunks,
    isLoading: chunksLoading,
    isError: chunksError,
  } = useQuery({
    queryKey: ["document_chunks", id],
    queryFn: () => fetchDocumentChunks(id),
    enabled: !!id,
  });

  const {
    data: sections,
    isLoading: sectionsLoading,
    isError: sectionsError,
  } = useQuery({
    queryKey: ["document_sections", id],
    queryFn: () => fetchDocumentSections(id),
    enabled: !!id,
  });

  const renderChunksBySection = (
    allChunks: Chunk[],
    allSections?: Section[]
  ) => {
    if (!allChunks || allChunks.length === 0) {
      return (
        <div className="text-sm text-gray-500">
          No text chunks found for this document.
        </div>
      );
    }

    if (!allSections || allSections.length === 0) {
      return (
        <div className="space-y-3 text-sm">
          {allChunks.map((chunk) => (
            <div key={chunk.id} className="border border-gray-200 rounded-lg p-3">
              <div className="text-xs text-gray-500 mb-1">
                Chunk #{chunk.chunk_index}
              </div>
              <p className="whitespace-pre-wrap text-gray-800">
                {chunk.text}
              </p>
            </div>
          ))}
        </div>
      );
    }

    const bySection: Record<string, Chunk[]> = {};
    const unsectioned: Chunk[] = [];

    for (const chunk of allChunks) {
      if (chunk.section_id) {
        const key = chunk.section_id;
        if (!bySection[key]) {
          bySection[key] = [];
        }
        bySection[key].push(chunk);
      } else {
        unsectioned.push(chunk);
      }
    }

    const sectionOrder = allSections;

    return (
      <div className="space-y-4 text-sm">
        <div className="border border-gray-200 rounded-lg overflow-hidden">
          <div className="bg-gray-50 px-3 py-2 border-b border-gray-200 text-xs font-semibold text-gray-700">
            Detected sections
          </div>
          <table className="min-w-full text-xs">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wide">
                  Title
                </th>
                <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wide">
                  Level
                </th>
                <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wide">
                  Chunks
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {sectionOrder.map((section) => (
                <tr key={section.id}>
                  <td className="px-3 py-2 truncate max-w-xs" title={section.title}>
                    {section.title}
                  </td>
                  <td className="px-3 py-2">{section.level}</td>
                  <td className="px-3 py-2">
                    {(bySection[section.id] || []).length}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {unsectioned.length > 0 && (
          <div className="border border-gray-200 rounded-lg p-3">
            <div className="text-xs font-semibold text-gray-600 mb-2">
              Document preface / other content
            </div>
            <div className="space-y-3">
              {unsectioned.map((chunk) => (
                <div key={chunk.id}>
                  <div className="text-xs text-gray-500 mb-1">
                    Chunk #{chunk.chunk_index}
                  </div>
                  <p className="whitespace-pre-wrap text-gray-800">
                    {chunk.text}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {sectionOrder.map((section) => {
          const sectionChunks = bySection[section.id] || [];
          if (sectionChunks.length === 0) {
            return null;
          }
          return (
            <div
              key={section.id}
              className="border border-gray-200 rounded-lg p-3"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="space-y-0.5">
                  <div className="text-xs uppercase tracking-wide text-gray-500">
                    Section
                  </div>
                  <div className="text-sm font-semibold text-gray-800">
                    {section.title}
                  </div>
                </div>
                <div className="text-[10px] text-gray-400">
                  Level {section.level}
                </div>
              </div>
              <div className="space-y-3">
                {sectionChunks.map((chunk) => (
                  <div key={chunk.id}>
                    <div className="text-xs text-gray-500 mb-1">
                      Chunk #{chunk.chunk_index}
                    </div>
                    <p className="whitespace-pre-wrap text-gray-800">
                      {chunk.text}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  if (!id) {
    return (
      <div className="space-y-4">
        <button
          className="text-sm text-blue-600 hover:underline"
          onClick={() => router.push("/documents")}
        >
          ← Back to documents
        </button>
        <div className="text-sm text-red-600">Invalid document id.</div>
      </div>
    );
  }

  if (docLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-500">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span>Loading document...</span>
      </div>
    );
  }

  if (docError || !document) {
    return (
      <div className="space-y-4">
        <button
          className="text-sm text-blue-600 hover:underline"
          onClick={() => router.push("/documents")}
        >
          ← Back to documents
        </button>
        <div className="text-sm text-red-600">Failed to load document.</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <Link href="/documents" className="text-sm text-blue-600 hover:underline">
            ← Back to documents
          </Link>
          <h1 className="text-2xl font-bold text-gray-900">
            {document.title || document.uri.split("/").pop()}
          </h1>
          <p className="text-xs text-gray-500 break-all">
            Source: {document.uri}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[70vh]">
        <div className="border rounded-xl overflow-hidden bg-gray-50">
          <iframe
            title="Document preview"
            src={`/api/v1/documents/${document.id}/file`}
            className="w-full h-full"
          />
        </div>
        <div className="border rounded-xl p-4 overflow-y-auto bg-white">
          <h2 className="text-sm font-semibold text-gray-800 mb-3">
            Extracted text
          </h2>
          {chunksLoading && (
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Loading chunks...</span>
            </div>
          )}
          {(chunksError || sectionsError) && (
            <div className="text-sm text-red-600 bg-red-50 border border-red-100 rounded-lg px-3 py-2">
              Failed to load text chunks or sections.
            </div>
          )}
          {(chunksLoading || sectionsLoading) && (
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Loading structure...</span>
            </div>
          )}
          {!chunksLoading &&
            !sectionsLoading &&
            !chunksError &&
            !sectionsError &&
            chunks && (
              <>{renderChunksBySection(chunks, sections)}</>
            )}
        </div>
      </div>
    </div>
  );
}

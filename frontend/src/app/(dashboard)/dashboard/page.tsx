"use client";

import Link from "next/link";
import { Upload, FileText, History, Settings, CloudUpload } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { fetchDocuments, Document } from "@/lib/api";

function QuickActionCard({ icon: Icon, title, description, href }: { icon: any, title: string, description: string, href: string }) {
  return (
    <Link href={href} className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow flex items-start gap-4">
      <div className="bg-indigo-50 p-3 rounded-lg">
        <Icon className="w-6 h-6 text-indigo-600" />
      </div>
      <div>
        <h3 className="font-semibold text-gray-900">{title}</h3>
        <p className="text-sm text-gray-500 mt-1">{description}</p>
      </div>
    </Link>
  );
}

function RecentDocumentCard({ doc }: { doc: Document }) {
  // Format date
  const date = new Date(doc.created_at).toLocaleDateString();
  const fileType = doc.mime_type?.split('/')[1]?.toUpperCase() || 'FILE';
  
  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex flex-col justify-between h-40">
      <div className="flex items-start gap-4">
        <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center shrink-0">
          <FileText className="w-5 h-5 text-blue-600" />
        </div>
        <div className="overflow-hidden">
          <h4 className="font-semibold text-gray-900 truncate" title={doc.uri}>{doc.title || doc.uri.split('/').pop()}</h4>
          <p className="text-xs text-gray-500 mt-1">{doc.size_bytes ? (doc.size_bytes / 1024 / 1024).toFixed(1) + ' MB' : 'Unknown Size'}</p>
        </div>
      </div>
      
      <div className="flex items-end justify-between mt-4">
        <div className="text-xs text-gray-400">
          Processed: {date}
        </div>
        <div className="flex items-center gap-2">
           <span className="text-xs font-medium text-green-500 bg-green-50 px-2 py-1 rounded-full">Completed</span>
        </div>
      </div>
      
      <div className="flex gap-2 mt-2 pt-2 border-t border-gray-50">
         <button className="text-xs font-medium text-indigo-600 hover:bg-indigo-50 px-3 py-1.5 rounded w-full">View</button>
         <button className="text-xs font-medium text-gray-600 hover:bg-gray-50 px-3 py-1.5 rounded w-full">Download</button>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const { data: documents } = useQuery({ queryKey: ["documents"], queryFn: () => fetchDocuments() });
  const recentDocs = documents?.slice(0, 3) || [];

  return (
    <div className="space-y-8">
      {/* Banner */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 text-white shadow-lg">
        <h1 className="text-3xl font-bold mb-2">Welcome back, John!</h1>
        <p className="text-blue-100 mb-6 max-w-xl">
          Upload your documents and let our AI-powered OCR system extract text with high accuracy.
        </p>
        <Link 
          href="/sources" 
          className="bg-white text-blue-600 px-6 py-2.5 rounded-lg font-semibold inline-flex items-center gap-2 hover:bg-blue-50 transition-colors"
        >
          <Upload className="w-4 h-4" />
          Upload Documents
        </Link>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <QuickActionCard 
          icon={Upload} 
          title="Upload Documents" 
          description="PDF, JPG, PNG formats" 
          href="/sources"
        />
        <QuickActionCard 
          icon={FileText} 
          title="Export Results" 
          description="TXT, DOCX, CSV formats" 
          href="#"
        />
        <QuickActionCard 
          icon={History} 
          title="Recent Files" 
          description="View processing history" 
          href="/jobs"
        />
        <QuickActionCard 
          icon={Settings} 
          title="Settings" 
          description="Customize your experience" 
          href="#"
        />
      </div>

      {/* Process New Documents (Drag & Drop Area) */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-gray-800">Process New Documents</h2>
        <div className="border-2 border-dashed border-gray-300 rounded-2xl bg-white p-12 flex flex-col items-center justify-center text-center hover:border-blue-400 transition-colors cursor-pointer">
          <CloudUpload className="w-12 h-12 text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Drag & drop files here</h3>
          <p className="text-gray-500 mb-6">or click to browse your files</p>
          <Link 
            href="/sources"
            className="bg-blue-600 text-white px-8 py-2.5 rounded-lg font-medium hover:bg-blue-700 transition-colors shadow-sm"
          >
            Select Files
          </Link>
          <p className="text-xs text-gray-400 mt-4">Supports PDF, JPG, PNG up to 10MB each</p>
        </div>
      </div>

      {/* Recent Documents */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-gray-800">Recent Documents</h2>
        {recentDocs.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {recentDocs.map(doc => (
              <RecentDocumentCard key={doc.id} doc={doc} />
            ))}
          </div>
        ) : (
          <div className="bg-white p-8 rounded-xl border border-gray-100 text-center text-gray-500">
            No recent documents found.
          </div>
        )}
      </div>
    </div>
  );
}

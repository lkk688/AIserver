"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchJobs, Job } from "@/lib/api";
import { Loader2, CheckCircle, XCircle, Clock, PlayCircle } from "lucide-react";
import clsx from "clsx";

const StatusIcon = ({ status }: { status: Job['status'] }) => {
  switch (status) {
    case 'done': return <CheckCircle className="w-5 h-5 text-green-500" />;
    case 'failed': return <XCircle className="w-5 h-5 text-red-500" />;
    case 'running': return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
    case 'pending': return <Clock className="w-5 h-5 text-gray-400" />;
    default: return null;
  }
};

const StatusBadge = ({ status }: { status: Job['status'] }) => {
  const styles = {
    done: "bg-green-100 text-green-700",
    failed: "bg-red-100 text-red-700",
    running: "bg-blue-100 text-blue-700",
    pending: "bg-gray-100 text-gray-700",
  };
  return (
    <span className={clsx("px-2 py-1 rounded-full text-xs font-medium capitalize", styles[status])}>
      {status}
    </span>
  );
};

export default function JobsPage() {
  const { data: jobs, isLoading } = useQuery({ 
    queryKey: ["jobs"], 
    queryFn: fetchJobs,
    refetchInterval: 2000 // Poll every 2s
  });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Jobs</h1>

      {isLoading ? (
        <div className="flex justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
        </div>
      ) : (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Details</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Progress</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {jobs?.map((job) => (
                <tr key={job.id}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center gap-2">
                      <StatusIcon status={job.status} />
                      <StatusBadge status={job.status} />
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 capitalize">
                    {job.type.replace('_', ' ')}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    <div className="max-w-xs truncate">
                      {job.error ? (
                        <span className="text-red-600">{job.error}</span>
                      ) : (
                        JSON.stringify(job.payload)
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="w-full bg-gray-200 rounded-full h-2.5 max-w-[100px]">
                      <div 
                        className="bg-blue-600 h-2.5 rounded-full transition-all duration-500" 
                        style={{ width: `${(job.progress || 0) * 100}%` }}
                      ></div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {new Date(job.created_at).toLocaleString()}
                  </td>
                </tr>
              ))}
              {jobs?.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-6 py-12 text-center text-gray-500">
                    No jobs found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

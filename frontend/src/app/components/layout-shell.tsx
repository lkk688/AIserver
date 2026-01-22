"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, 
  Upload, 
  FileText, 
  History, 
  Sparkles, 
  Search,
  Bell,
  User,
  Settings
} from "lucide-react";
import clsx from "clsx";

function Sidebar() {
  const pathname = usePathname();

  const menuItems = [
    { icon: LayoutDashboard, label: "Dashboard", href: "/dashboard" },
    { icon: Upload, label: "Upload Documents", href: "/sources" },
    { icon: FileText, label: "My Documents", href: "/documents" },
    { icon: History, label: "History", href: "/jobs" },
    { icon: Sparkles, label: "AI Enhancements", href: "#" },
    { icon: Search, label: "Smart Search", href: "/search" },
  ];

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 bg-[#2b2d7e] text-white flex flex-col">
      {/* Brand */}
      <div className="p-6 flex items-center gap-3">
        <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
          <div className="w-4 h-4 bg-white rounded-full opacity-50"></div>
        </div>
        <div>
          <h1 className="font-bold text-lg leading-none">AI-OCR</h1>
          <p className="text-xs text-gray-300">User Dashboard</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 space-y-2 mt-4">
        {menuItems.map((item) => {
          const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
          return (
            <Link
              key={item.label}
              href={item.href}
              className={clsx(
                "flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors",
                isActive 
                  ? "bg-[#3e41a6] text-white shadow-sm" 
                  : "text-gray-300 hover:bg-[#3e41a6]/50 hover:text-white"
              )}
            >
              <item.icon className="w-5 h-5" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* User Profile */}
      <div className="p-4 border-t border-indigo-800">
        <div className="flex items-center gap-3 px-4 py-2">
          <div className="w-10 h-10 bg-indigo-500 rounded-full flex items-center justify-center">
            <User className="w-6 h-6 text-indigo-200" />
          </div>
          <div className="overflow-hidden">
            <p className="text-sm font-medium truncate">John Doe</p>
            <p className="text-xs text-gray-400 truncate">user@example.com</p>
          </div>
        </div>
      </div>
    </aside>
  );
}

function TopBar() {
  return (
    <header className="h-20 bg-white border-b border-gray-100 flex items-center justify-between px-8 sticky top-0 z-10">
      <h2 className="text-xl font-semibold text-gray-800">Document Processing Center</h2>
      <div className="flex items-center gap-6">
        <div className="relative">
          <div className="w-2 h-2 bg-red-500 rounded-full absolute top-0 right-0"></div>
          <Bell className="w-5 h-5 text-gray-500" />
        </div>
        <div className="relative">
          <input 
            type="text" 
            placeholder="Search documents..." 
            className="w-64 pl-4 pr-10 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <Search className="w-4 h-4 text-gray-400 absolute right-3 top-1/2 -translate-y-1/2" />
        </div>
      </div>
    </header>
  );
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gray-50 flex font-sans">
      <Sidebar />
      <div className="flex-1 ml-64 flex flex-col">
        <TopBar />
        <main className="flex-1 p-8 overflow-y-auto">
          {children}
        </main>
        <footer className="px-8 py-4 text-xs text-gray-400 border-t flex justify-between">
          <span>© 2025 AI-OCR Pro. All rights reserved.</span>
          <div className="flex items-center gap-2">
            <span>User</span>
            <div className="w-8 h-4 bg-gray-300 rounded-full relative cursor-pointer">
              <div className="w-4 h-4 bg-white rounded-full shadow-sm absolute left-0"></div>
            </div>
            <span>Admin</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

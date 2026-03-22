import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8080/api/:path*",
      },
      {
        source: "/agent/:path*",
        destination: "http://127.0.0.1:8000/agent/:path*",
      },
    ];
  },
};

export default nextConfig;

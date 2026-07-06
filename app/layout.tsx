import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SmartVision AI - Real-time Spatial Perception & 3D Object Detection",
  description:
    "Real-time spatial perception and multi-object tracking platform featuring 2D-to-3D metric spatial fusion, Kalman filtering, and interactive Three.js 3D visualizers.",
  manifest: "/manifest.json",
  icons: {
    icon: "/icons/icon-192x192.png",
    apple: "/icons/icon-192x192.png",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="min-h-full flex flex-col bg-[#080c14] text-slate-100">
        {children}
      </body>
    </html>
  );
}

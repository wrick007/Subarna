import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, Inter, Space_Grotesk } from "next/font/google";
import "./globals.css";

// Three type roles (see globals.css's token comment for the reasoning
// behind each): Space Grotesk for the wordmark only -- a single glance
// per screen, not a running display face -- Inter for everything else
// UI/prose, and IBM Plex Mono specifically for monetary figures so
// amounts line up like a printed statement.
const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
  weight: ["500", "600"],
});

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "FinMate AI",
  description:
    "A multi-agent personal finance assistant. Every answer traces back to a retrieved transaction or a deterministic calculation, and a critic agent checks it before you see it.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#faf8f3",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${spaceGrotesk.variable} ${inter.variable} ${plexMono.variable} h-full`}
    >
      <body className="h-full bg-paper text-ink antialiased">{children}</body>
    </html>
  );
}

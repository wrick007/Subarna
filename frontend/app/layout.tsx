import type { Metadata, Viewport } from "next";
import { Fraunces, IBM_Plex_Mono, Inter } from "next/font/google";
import "./globals.css";

// Three type roles, deliberately not the same pairing you'd reach for
// on any other project (see globals.css's token comment): Fraunces for
// the wordmark and section headers (characterful, used sparingly), Inter
// for everything else UI/prose, and IBM Plex Mono specifically for
// monetary figures so amounts line up like a printed statement.
const fraunces = Fraunces({
  variable: "--font-fraunces",
  subsets: ["latin"],
  weight: ["500", "600"],
  style: ["normal", "italic"],
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
  themeColor: "#f3f4ee",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${fraunces.variable} ${inter.variable} ${plexMono.variable} h-full`}
    >
      <body className="h-full bg-paper text-ink antialiased">{children}</body>
    </html>
  );
}

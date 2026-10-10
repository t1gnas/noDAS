import type { Metadata } from "next";
import { ResultsView } from "@/components/ResultsView";

export const metadata: Metadata = { title: "noDAS · Results" };

export default function ResultsPage() {
  return <ResultsView />;
}

import type { Metadata } from "next";
import { MapView } from "@/components/MapView";

export const metadata: Metadata = { title: "noDAS · Map" };

export default function MapPage() {
  return <MapView />;
}

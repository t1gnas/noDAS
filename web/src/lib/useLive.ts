"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertEvent, Frame, LiveMessage, Meta, Summary, wsUrl } from "./api";

export type Status = "idle" | "connecting" | "playing" | "paused" | "ended" | "error";

export type AlertItem = AlertEvent & { open: boolean; opened_t: number; upgraded: boolean };

type Handlers = { onMeta: (m: Meta) => void; onFrame: (f: Frame, m: Meta) => void };

function applyEvent(list: AlertItem[], e: AlertEvent): AlertItem[] {
  if (e.kind === "open") return [{ ...e, open: true, opened_t: e.t, upgraded: false }, ...list];
  return list.map((a) =>
    a.id !== e.id
      ? a
      : e.kind === "update"
        ? { ...a, ...e, open: true, opened_t: a.opened_t, upgraded: true }
        : { ...a, ...e, verified: a.verified || e.verified, open: false, opened_t: a.opened_t },
  );
}

export function useLive(handlers: Handlers) {
  const wsRef = useRef<WebSocket | null>(null);
  const metaRef = useRef<Meta | null>(null);
  const handlersRef = useRef(handlers);
  const [status, setStatus] = useState<Status>("idle");
  const [meta, setMeta] = useState<Meta | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [speed, setSpeedState] = useState(5);

  useEffect(() => {
    handlersRef.current = handlers;
  });

  const stop = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  const onMessage = useCallback((msg: LiveMessage) => {
    if (msg.type === "meta") {
      metaRef.current = msg;
      setMeta(msg);
      handlersRef.current.onMeta(msg);
      setStatus("playing");
    } else if (msg.type === "frame" && metaRef.current) {
      handlersRef.current.onFrame(msg, metaRef.current);
      setElapsed(msg.end * metaRef.current.row_seconds);
      if (msg.events.length) setAlerts((prev) => msg.events.reduce(applyEvent, prev));
    } else if (msg.type === "end") {
      setSummary(msg);
      setStatus("ended");
    } else if (msg.type === "error") {
      setError(msg.message);
      setStatus("error");
    }
  }, []);

  const start = useCallback(
    (scenarioId: string) => {
      stop();
      setAlerts([]);
      setSummary(null);
      setError(null);
      setElapsed(0);
      setStatus("connecting");
      const ws = new WebSocket(wsUrl(`/api/live/${scenarioId}?speed=${speed}`));
      wsRef.current = ws;
      ws.onmessage = (ev) => onMessage(JSON.parse(ev.data) as LiveMessage);
      ws.onerror = () => {
        setError("Cannot reach the detection backend. Is the API running on port 8000?");
        setStatus("error");
      };
    },
    [onMessage, speed, stop],
  );

  const send = useCallback((payload: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(JSON.stringify(payload));
  }, []);

  const pause = useCallback(() => {
    send({ action: "pause" });
    setStatus("paused");
  }, [send]);

  const resume = useCallback(() => {
    send({ action: "resume" });
    setStatus("playing");
  }, [send]);

  const setSpeed = useCallback(
    (s: number) => {
      setSpeedState(s);
      send({ speed: s });
    },
    [send],
  );

  useEffect(() => stop, [stop]);

  return { status, meta, elapsed, alerts, summary, error, speed, start, pause, resume, setSpeed, stop };
}

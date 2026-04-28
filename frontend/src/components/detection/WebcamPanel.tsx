import { Camera, Power, PowerOff, Radio } from "lucide-react";

import { Button } from "@/components/ui/button";
import { DetectionShell, EmptyPreview } from "./DetectionShell";

type WebcamPanelProps = {
  apiBaseUrl: string;
  isActive: boolean;
  isBusy: boolean;
  statusText: string;
  onStart: () => void;
  onStop: () => void;
};

export function WebcamPanel({
  apiBaseUrl,
  isActive,
  isBusy,
  statusText,
  onStart,
  onStop,
}: WebcamPanelProps) {
  return (
    <DetectionShell
      title="Live webcam detection"
      description="Start the backend camera stream and view YOLO annotations as multipart frames from /stream/webcam."
      icon={<Camera className="size-5" />}
      aside={
        <div className="space-y-5">
          <div>
            <p className="text-sm font-semibold text-muted-foreground">Stream state</p>
            <p className="mt-2 text-3xl font-bold text-foreground">
              {isActive ? "Active" : "Inactive"}
            </p>
            <p className="mt-2 text-sm text-muted-foreground">{statusText}</p>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Button variant="hero" onClick={onStart} disabled={isBusy || isActive}>
              <Power />
              Start
            </Button>
            <Button variant="console" onClick={onStop} disabled={isBusy || !isActive}>
              <PowerOff />
              Stop
            </Button>
          </div>
          <div className="rounded-xl bg-surface p-4 text-sm text-muted-foreground">
            <Radio className="mb-3 size-5 text-primary" />
            Uses GET /stream/start, /stream/stop, /stream/status and MJPEG /stream/webcam.
          </div>
        </div>
      }
    >
      {isActive ? (
        <div className="overflow-hidden rounded-2xl border border-border bg-surface shadow-glow">
          <img
            src={`${apiBaseUrl.replace(/\/$/, "")}/stream/webcam?ts=${Date.now()}`}
            alt="YOLO webcam detection stream"
            className="w-full h-auto object-contain"
          />
        </div>
      ) : (
        <EmptyPreview label="Webcam stream is waiting" />
      )}
    </DetectionShell>
  );
}

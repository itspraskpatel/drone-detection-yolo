import { Film, Loader2, PlaySquare, UploadCloud } from "lucide-react";

import { Button } from "@/components/ui/button";
import { DetectionShell, EmptyPreview } from "./DetectionShell";

type VideoPanelProps = {
  previewUrl: string | null;
  resultUrl: string | null;
  isProcessing: boolean;
  onFileChange: (file: File | null) => void;
  onSubmit: () => void;
};

export function VideoPanel({
  previewUrl,
  resultUrl,
  isProcessing,
  onFileChange,
  onSubmit,
}: VideoPanelProps) {
  return (
    <DetectionShell
      title="Video upload detection"
      description="Upload an MP4 or other browser-supported video, process it through /predict/video, and play back the annotated output."
      icon={<Film className="size-5" />}
      aside={
        <div className="space-y-5">
          <label className="flex min-h-36 cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-border bg-surface px-4 text-center transition hover:bg-surface-strong">
            <UploadCloud className="mb-3 size-7 text-primary" />
            <span className="text-sm font-bold text-foreground">Choose video</span>
            <span className="mt-1 text-xs text-muted-foreground">MP4, MOV or WEBM</span>
            <input
              type="file"
              accept="video/*"
              className="sr-only"
              onChange={(event) => onFileChange(event.target.files?.[0] ?? null)}
            />
          </label>
          <Button
            variant="hero"
            className="w-full"
            onClick={onSubmit}
            disabled={!previewUrl || isProcessing}
          >
            {isProcessing ? <Loader2 className="animate-spin" /> : <PlaySquare />}
            Process video
          </Button>
          <div className="rounded-xl bg-surface p-4 text-sm text-muted-foreground">
            The API returns a processed MP4 file directly, so playback appears here after processing
            completes.
          </div>
        </div>
      }
    >
      {previewUrl || resultUrl ? (
        <div className="grid gap-4 md:grid-cols-2">
          {previewUrl && (
            <figure className="overflow-hidden rounded-2xl border border-border bg-surface">
              <video src={previewUrl} controls className="aspect-video w-full object-contain" />
              <figcaption className="border-t border-border px-4 py-3 text-sm font-semibold text-muted-foreground">
                Original video
              </figcaption>
            </figure>
          )}
          {resultUrl && (
            <figure className="overflow-hidden rounded-2xl border border-border bg-surface shadow-glow">
              <video
                key={resultUrl}
                src={resultUrl}
                controls
                preload="metadata"
                playsInline
                className="aspect-video w-full object-contain"
              />
              <figcaption className="border-t border-border px-4 py-3 text-sm font-semibold text-muted-foreground">
                Processed output
              </figcaption>
            </figure>
          )}
        </div>
      ) : (
        <EmptyPreview label="Upload a video to process" />
      )}
    </DetectionShell>
  );
}

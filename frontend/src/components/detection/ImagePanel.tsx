import { Crosshair, ImageUp, Loader2, UploadCloud } from "lucide-react";

import { Button } from "@/components/ui/button";
import { DetectionShell, EmptyPreview } from "./DetectionShell";

export type Midpoint = {
  mid_x: number;
  mid_y: number;
};

type ImagePanelProps = {
  previewUrl: string | null;
  resultUrl: string | null;
  midpoints: Midpoint[];
  isProcessing: boolean;
  onFileChange: (file: File | null) => void;
  onSubmit: () => void;
};

export function ImagePanel({
  previewUrl,
  resultUrl,
  midpoints,
  isProcessing,
  onFileChange,
  onSubmit,
}: ImagePanelProps) {
  return (
    <DetectionShell
      title="Image upload detection"
      description="Send an image to /predict/image, then inspect the annotated result and detection midpoints."
      icon={<ImageUp className="size-5" />}
      aside={
        <div className="space-y-5">
          <label className="flex min-h-36 cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-border bg-surface px-4 text-center transition hover:bg-surface-strong">
            <UploadCloud className="mb-3 size-7 text-primary" />
            <span className="text-sm font-bold text-foreground">Choose image</span>
            <span className="mt-1 text-xs text-muted-foreground">PNG, JPG or WEBP</span>
            <input
              type="file"
              accept="image/*"
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
            {isProcessing ? <Loader2 className="animate-spin" /> : <Crosshair />}
            Detect objects
          </Button>
          <div>
            <p className="mb-3 text-sm font-semibold text-muted-foreground">Midpoints</p>
            <div className="max-h-56 space-y-2 overflow-auto pr-1">
              {midpoints.length ? (
                midpoints.map((point, index) => (
                  <div
                    key={`${point.mid_x}-${point.mid_y}-${index}`}
                    className="flex items-center justify-between rounded-xl bg-surface px-3 py-2 text-sm"
                  >
                    <span className="font-semibold text-foreground">Target {index + 1}</span>
                    <span className="text-muted-foreground">
                      {point.mid_x}, {point.mid_y}
                    </span>
                  </div>
                ))
              ) : (
                <p className="rounded-xl bg-surface p-4 text-sm text-muted-foreground">
                  No coordinates yet.
                </p>
              )}
            </div>
          </div>
        </div>
      }
    >
      {resultUrl || previewUrl ? (
        <div className="grid gap-4 md:grid-cols-2">
          {previewUrl && (
            <figure className="overflow-hidden rounded-2xl border border-border bg-surface">
              <img
                src={previewUrl}
                alt="Selected upload preview"
                className="aspect-video w-full object-contain"
              />
              <figcaption className="border-t border-border px-4 py-3 text-sm font-semibold text-muted-foreground">
                Original
              </figcaption>
            </figure>
          )}
          {resultUrl && (
            <figure className="overflow-hidden rounded-2xl border border-border bg-surface shadow-glow">
              <img
                src={resultUrl}
                alt="Annotated YOLO detection result"
                className="aspect-video w-full object-contain"
              />
              <figcaption className="border-t border-border px-4 py-3 text-sm font-semibold text-muted-foreground">
                Detected result
              </figcaption>
            </figure>
          )}
        </div>
      ) : (
        <EmptyPreview label="Drop in an image to analyze" />
      )}
    </DetectionShell>
  );
}

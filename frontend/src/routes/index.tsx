import { useCallback, useEffect, useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Camera, Film, ImageUp, Radar, ScanSearch, ShieldCheck, Sparkles } from "lucide-react";

import { ApiSettings } from "@/components/detection/ApiSettings";
import { ImagePanel, type Midpoint } from "@/components/detection/ImagePanel";
import { VideoPanel } from "@/components/detection/VideoPanel";
import { WebcamPanel } from "@/components/detection/WebcamPanel";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "YOLO Detection Dashboard" },
      {
        name: "description",
        content: "Modern React UI for YOLO webcam, image, and video object detection endpoints.",
      },
      { property: "og:title", content: "YOLO Detection Dashboard" },
      {
        property: "og:description",
        content:
          "Run webcam streams, upload images, and process videos through FastAPI YOLO endpoints.",
      },
    ],
  }),
  component: Index,
});

type HealthState = "idle" | "checking" | "healthy" | "offline";

function Index() {
  const [apiBaseUrl, setApiBaseUrl] = useState("http://localhost:8000");
  const [health, setHealth] = useState<HealthState>("idle");
  const [streamActive, setStreamActive] = useState(false);
  const [streamBusy, setStreamBusy] = useState(false);
  const [streamStatus, setStreamStatus] = useState("Ready to connect to the camera stream.");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageResult, setImageResult] = useState<string | null>(null);
  const [midpoints, setMidpoints] = useState<Midpoint[]>([]);
  const [imageProcessing, setImageProcessing] = useState(false);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [videoResult, setVideoResult] = useState<string | null>(null);
  const [videoProcessing, setVideoProcessing] = useState(false);

  const cleanBaseUrl = useMemo(() => apiBaseUrl.replace(/\/$/, ""), [apiBaseUrl]);

  const checkHealth = useCallback(async () => {
    setHealth("checking");
    try {
      const response = await fetch(`${cleanBaseUrl}/health`);
      setHealth(response.ok ? "healthy" : "offline");
    } catch {
      setHealth("offline");
    }
  }, [cleanBaseUrl]);

  const refreshStreamStatus = useCallback(async () => {
    try {
      const response = await fetch(`${cleanBaseUrl}/stream/status`);
      const data = (await response.json()) as { active: boolean };
      setStreamActive(Boolean(data.active));
      setStreamStatus(
        data.active ? "Backend webcam stream is running." : "Backend webcam stream is stopped.",
      );
    } catch {
      setStreamStatus("Unable to read stream status from the API.");
    }
  }, [cleanBaseUrl]);

  useEffect(() => {
    void refreshStreamStatus();
  }, [refreshStreamStatus]);

  const startStream = async () => {
    setStreamBusy(true);
    try {
      await fetch(`${cleanBaseUrl}/stream/start`);
      setStreamActive(true);
      setStreamStatus("Stream started. Live frames are loading.");
    } catch {
      setStreamStatus("Could not start stream. Check API and CORS settings.");
    } finally {
      setStreamBusy(false);
    }
  };

  const stopStream = async () => {
    setStreamBusy(true);
    try {
      await fetch(`${cleanBaseUrl}/stream/stop`);
      setStreamActive(false);
      setStreamStatus("Stream stopped safely.");
    } catch {
      setStreamStatus("Could not stop stream. Check API availability.");
    } finally {
      setStreamBusy(false);
    }
  };

  const setImageUpload = (file: File | null) => {
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImageFile(file);
    setImageResult(null);
    setMidpoints([]);
    setImagePreview(file ? URL.createObjectURL(file) : null);
  };

  const submitImage = async () => {
    if (!imageFile) return;
    setImageProcessing(true);
    const formData = new FormData();
    formData.append("file", imageFile);
    try {
      const response = await fetch(`${cleanBaseUrl}/predict/image`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Image detection failed");
      const data = (await response.json()) as { result_image_url: string; midpoints: Midpoint[] };
      setImageResult(`${cleanBaseUrl}${data.result_image_url}`);
      setMidpoints(data.midpoints ?? []);
    } finally {
      setImageProcessing(false);
    }
  };

  const setVideoUpload = (file: File | null) => {
    if (videoPreview) URL.revokeObjectURL(videoPreview);
    if (videoResult) URL.revokeObjectURL(videoResult);
    setVideoFile(file);
    setVideoResult(null);
    setVideoPreview(file ? URL.createObjectURL(file) : null);
  };

  const submitVideo = async () => {
    if (!videoFile) return;
    setVideoProcessing(true);
    const formData = new FormData();
    formData.append("file", videoFile);
    try {
      const response = await fetch(`${cleanBaseUrl}/predict/video`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Video processing failed");
      const blob = await response.blob();
      setVideoResult(URL.createObjectURL(blob));
    } finally {
      setVideoProcessing(false);
    }
  };

  return (
    <main className="min-h-screen bg-hero-gradient px-4 py-6 text-foreground md:px-8 md:py-10">
      <div className="mx-auto max-w-7xl space-y-6">
        <header className="grid gap-6 lg:grid-cols-[1fr_360px] lg:items-end">
          <div>

            <h1 className="max-w-4xl text-4xl font-black leading-tight tracking-normal text-foreground md:text-6xl">
             Drone Detection console for Secure AirSpace.
            </h1>
            <p className="mt-5 max-w-2xl text-base leading-7 text-muted-foreground md:text-lg">
              A React frontend tailored to your FastAPI routes with live stream controls, upload
              workflows, result previews, and midpoint inspection.
            </p>
          </div>
        </header>

        <ApiSettings
          apiBaseUrl={apiBaseUrl}
          health={health}
          onApiBaseUrlChange={setApiBaseUrl}
          onHealthCheck={checkHealth}
        />

        <Tabs defaultValue="webcam" className="space-y-4">
          <TabsList className="grid h-auto w-full grid-cols-1 md:w-fit md:grid-cols-3">
            <TabsTrigger value="webcam">
              <Camera />
              Webcam
            </TabsTrigger>
            <TabsTrigger value="image">
              <ImageUp />
              Upload image
            </TabsTrigger>
            <TabsTrigger value="video">
              <Film />
              Upload video
            </TabsTrigger>
          </TabsList>
          <TabsContent value="webcam">
            <WebcamPanel
              apiBaseUrl={cleanBaseUrl}
              isActive={streamActive}
              isBusy={streamBusy}
              statusText={streamStatus}
              onStart={startStream}
              onStop={stopStream}
            />
          </TabsContent>
          <TabsContent value="image">
            <ImagePanel
              previewUrl={imagePreview}
              resultUrl={imageResult}
              midpoints={midpoints}
              isProcessing={imageProcessing}
              onFileChange={setImageUpload}
              onSubmit={submitImage}
            />
          </TabsContent>
          <TabsContent value="video">
            <VideoPanel
              previewUrl={videoPreview}
              resultUrl={videoResult}
              isProcessing={videoProcessing}
              onFileChange={setVideoUpload}
              onSubmit={submitVideo}
            />
          </TabsContent>
        </Tabs>

        <footer className="flex flex-col gap-3 rounded-2xl border border-border bg-surface p-4 text-sm text-muted-foreground backdrop-blur md:flex-row md:items-center md:justify-between">
          <span className="inline-flex items-center gap-2">
            <ShieldCheck className="size-4 text-primary" />
            Designed for your Instant Drone Detection.
          </span>
          <span className="inline-flex items-center gap-2">
            <ScanSearch className="size-4 text-accent" />
            CONF 0.40 · IMG 512 · YOLO annotations
          </span>
        </footer>
      </div>
    </main>
  );
}

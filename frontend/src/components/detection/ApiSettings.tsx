import { Activity, Link2 } from "lucide-react";

import { Button } from "@/components/ui/button";

type ApiSettingsProps = {
  apiBaseUrl: string;
  health: "idle" | "checking" | "healthy" | "offline";
  onApiBaseUrlChange: (value: string) => void;
  onHealthCheck: () => void;
};

export function ApiSettings({
  apiBaseUrl,
  health,
  onApiBaseUrlChange,
  onHealthCheck,
}: ApiSettingsProps) {
  const statusLabel =
    health === "checking"
      ? "Checking"
      : health === "healthy"
        ? "Healthy"
        : health === "offline"
          ? "Offline"
          : "Ready";

  return (
    <section className="rounded-2xl border border-border bg-panel-gradient p-4 shadow-lab backdrop-blur md:p-5">
      <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div className="min-w-0 flex-1">
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-muted-foreground">
            <Link2 className="size-4 text-primary" />
            FastAPI endpoint
          </div>
          <label className="sr-only" htmlFor="api-base-url">
            API base URL
          </label>
          <input
            id="api-base-url"
            value={apiBaseUrl}
            onChange={(event) => onApiBaseUrlChange(event.target.value)}
            className="h-12 w-full rounded-xl border border-input bg-background/70 px-4 text-sm font-medium text-foreground outline-none transition focus:border-ring focus:ring-2 focus:ring-ring/25"
            placeholder="http://localhost:8000"
          />
        </div>
        <div className="flex items-center gap-3">
          <div className="rounded-xl border border-border bg-surface px-4 py-3 text-sm font-semibold text-foreground">
            <span className="mr-2 inline-flex size-2 rounded-full bg-primary" />
            {statusLabel}
          </div>
          <Button variant="console" onClick={onHealthCheck} disabled={health === "checking"}>
            <Activity />
            Health
          </Button>
        </div>
      </div>
    </section>
  );
}

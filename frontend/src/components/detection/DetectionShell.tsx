import { ReactNode } from "react";

type DetectionShellProps = {
  title: string;
  description: string;
  icon: ReactNode;
  children: ReactNode;
  aside?: ReactNode;
};

export function DetectionShell({ title, description, icon, children, aside }: DetectionShellProps) {
  return (
    <section className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
      <div className="rounded-2xl border border-border bg-panel-gradient p-5 shadow-lab backdrop-blur md:p-6">
        <div className="mb-5 flex items-start gap-3">
          <div className="flex size-11 shrink-0 items-center justify-center rounded-xl bg-primary-gradient text-primary-foreground shadow-glow">
            {icon}
          </div>
          <div>
            <h2 className="text-xl font-bold text-foreground md:text-2xl">{title}</h2>
            <p className="mt-1 max-w-2xl text-sm leading-6 text-muted-foreground">{description}</p>
          </div>
        </div>
        {children}
      </div>
      <aside className="rounded-2xl border border-border bg-card p-5 shadow-lab backdrop-blur">
        {aside}
      </aside>
    </section>
  );
}

export function EmptyPreview({ label }: { label: string }) {
  return (
    <div className="relative flex min-h-[340px] items-center justify-center overflow-hidden rounded-2xl border border-dashed border-border bg-surface">
      <div className="absolute inset-x-0 top-0 h-16 bg-primary/10 scanner-sweep" />
      <div className="text-center">
        <div className="mx-auto mb-3 size-14 rounded-2xl border border-border bg-card shadow-sm" />
        <p className="text-sm font-semibold text-muted-foreground">{label}</p>
      </div>
    </div>
  );
}

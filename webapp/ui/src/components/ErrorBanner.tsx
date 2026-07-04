import { Icon } from "./Icon";

interface ErrorBannerProps {
  message: string;
  hint?: string;
  title?: string;
}

/** Shows the backend's message verbatim, plus a friendly "what to try" hint. */
export function ErrorBanner({ message, hint, title = "Something went wrong" }: ErrorBannerProps) {
  return (
    <div className="error-banner" role="alert">
      <Icon name="warning" />
      <div className="error-banner-body">
        <div className="error-banner-title">{title}</div>
        <div>{message}</div>
        {hint ? <div className="error-banner-hint">{hint}</div> : null}
      </div>
    </div>
  );
}

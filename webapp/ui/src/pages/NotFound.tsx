import { Link } from "react-router-dom";
import { EmptyState } from "../components/EmptyState";

export default function NotFound() {
  return (
    <div className="page">
      <EmptyState
        title="Page not found"
        body="The page you're looking for doesn't exist."
        action={
          <Link className="btn btn-primary mt-3" to="/">
            Back to overview
          </Link>
        }
      />
    </div>
  );
}

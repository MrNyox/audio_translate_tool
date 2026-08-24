export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(url, options = {}) {
  const init = {
    cache: "no-store",
    ...options,
  };

  const isFormData =
    typeof FormData !== "undefined" && options.body instanceof FormData;

  if (isFormData) {
    init.headers = {
      ...(options.headers || {}),
    };
  } else {
    init.headers = {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    };
  }

  let response;

  try {
    response = await fetch(url, init);
  } catch {
    throw new ApiError("Network error – cannot reach the backend.", 0);
  }

  if (response.status === 204) {
    return null;
  }

  let body;

  try {
    body = await response.json();
  } catch {
    throw new ApiError(
      `Server returned non-JSON response (HTTP ${response.status}).`,
      response.status
    );
  }

  if (!response.ok || body.ok === false) {
    throw new ApiError(
      body.error || `Request failed with HTTP ${response.status}.`,
      response.status
    );
  }

  return body.data ?? body;
}

export function getHealth() {
  return request("/api/health");
}

export function createJob(file) {
  const formData = new FormData();
  formData.append("file", file);

  return request("/api/jobs", {
    method: "POST",
    body: formData,
  });
}

export function getJob(jobId) {
  return request(`/api/jobs/${encodeURIComponent(jobId)}`);
}

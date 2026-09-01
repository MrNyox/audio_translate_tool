import * as api from "../api.js";
import * as ui from "../ui.js";
import state, { updateState } from "../state.js";
import { handleStageOneCompletion } from "./translation.js";

const POLL_MS = 1200;

let selectedFile = null;
let pollTimer = null;
let pollInFlight = false;
let pollFailureNotified = false;

export function initPipeline() {
  if (ui.els.fileInput) {
    ui.els.fileInput.addEventListener("change", onFileChange);
  }

  if (ui.els.btnProcess) {
    ui.els.btnProcess.addEventListener("click", onProcess);
  }

  checkHealth();
}

export function handleVisibilityChange() {
  if (document.hidden) {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
    return;
  }

  if (state.polling && state.jobId && !pollInFlight) {
    schedulePoll(250);
  }
}

async function checkHealth() {
  try {
    const health = await api.getHealth();

    updateState("connected", true);
    ui.setConnectionStatus(true);

    if (!health.ffmpeg_available) {
      ui.toast(
        "FFmpeg was not found. Please install FFmpeg and restart the server.",
        "danger"
      );
    }

    if (!health.aligner_loaded) {
      ui.toast(
        "Word-level timestamps aren't available this session, so subtitles " +
          "won't be generated (transcript/translation still work normally). " +
          "Check the server logs for details.",
        "info"
      );
    }
  } catch (error) {
    updateState("connected", false);
    ui.setConnectionStatus(false);
    ui.toast(error.message || "Backend unreachable.", "danger");
  }
}

function onFileChange(event) {
  if (state.busy) {
    return;
  }

  const input = event.target;
  selectedFile = input.files && input.files[0] ? input.files[0] : null;

  updateState(
    "selectedFileName",
    selectedFile ? selectedFile.name : null
  );

  ui.setFileMeta(
    selectedFile ? selectedFile.name : "No file selected."
  );
}

async function onProcess() {
  if (state.busy) {
    return;
  }

  if (!selectedFile) {
    ui.showError("No file selected.");
    ui.toast("Select a media file first.", "danger");
    return;
  }

  updateState("busy", true);
  ui.setBusy(true);
  ui.resetJobView();
  ui.setStep("Uploading");
  ui.setStatusMode("UPLOAD");

  pollFailureNotified = false;

  try {
    const created = await api.createJob(selectedFile);

    updateState("jobId", created.job_id);
    updateState("lastStatus", created.status || "queued");

    ui.setJobId(created.job_id);
    ui.setStep(created.step || "Queued");
    ui.setProgress(created.progress || 0);

    startPolling();
  } catch (error) {
    updateState("busy", false);
    updateState("jobId", null);

    ui.setBusy(false);
    ui.setStatusMode("FAILED");
    ui.showError(error.message || "Upload failed.");
    ui.toast(error.message || "Upload failed.", "danger");
  }
}

function startPolling() {
  if (state.polling || !state.jobId) {
    return;
  }

  updateState("polling", true);
  pollInFlight = false;
  schedulePoll(300);
}

function stopPolling() {
  updateState("polling", false);

  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }

  pollInFlight = false;
}

function schedulePoll(delay = POLL_MS) {
  if (pollTimer) {
    clearTimeout(pollTimer);
  }

  pollTimer = setTimeout(pollJob, delay);
}

async function pollJob() {
  if (!state.polling || pollInFlight || document.hidden || !state.jobId) {
    return;
  }

  pollInFlight = true;

  try {
    const job = await api.getJob(state.jobId);
    pollFailureNotified = false;

    updateState("lastStatus", job.status);
    renderJob(job);

    if (job.status === "completed") {
      const outputs = job.outputs || {};

      if (!outputs.audio_url || !outputs.transcript_url) {
        stopPolling();
        failJob("Output files are missing after completion.");
        return;
      }

      stopPolling();
      completeJob(outputs);
      return;
    }

    if (job.status === "failed") {
      stopPolling();
      failJob(job.error || "Job failed.");
      return;
    }

    schedulePoll();
  } catch (error) {
    if (error.status === 404) {
      stopPolling();
      failJob(error.message || "Job not found.");
      return;
    }

    if (!pollFailureNotified) {
      ui.toast(error.message || "Status poll failed.", "danger");
      pollFailureNotified = true;
    }

    schedulePoll();
  } finally {
    pollInFlight = false;
  }
}

function renderJob(job) {
  ui.setProgress(job.progress ?? 0);
  ui.setStep(job.step || job.status || "Working");
  ui.setStatusMode((job.status || "WORKING").toUpperCase());

  if (job.error) {
    ui.showError(job.error);
  } else {
    ui.clearError();
  }
}

function completeJob(outputs) {
  ui.setProgress(100);
  ui.setStep("Completed");
  ui.setStatusMode("COMPLETE");
  ui.clearError();
  ui.showOutputs(outputs);

  updateState("busy", false);
  ui.setBusy(false);

  // Reveal the Stage Two (translation) panel now that a transcript exists.
  handleStageOneCompletion(state.jobId);

  ui.toast("Job completed.", "success");
}

function failJob(message) {
  const safeMessage = message || "Job failed.";

  ui.setStatusMode("FAILED");
  ui.showError(safeMessage);

  updateState("busy", false);
  ui.setBusy(false);

  ui.toast(safeMessage, "danger");
}

import * as api from "../api.js";
import * as ui from "../ui.js";
import state, { updateState } from "../state.js";

const POLL_MS = 1200;

let translationPollTimer = null;
let translationPollInFlight = false;
let translationPollFailureNotified = false;

export function initTranslation() {
  if (ui.els.selTargetLang) {
    ui.els.selTargetLang.addEventListener("change", (e) => {
      updateState("targetLanguage", e.target.value);
    });
  }

  if (ui.els.btnTranslate) {
    ui.els.btnTranslate.addEventListener("click", onTranslate);
  }
}

export function handleStageOneCompletion(jobId) {
  updateState("jobId", jobId);
  ui.showStageTwoPanel();
}

async function onTranslate() {
  if (state.busy || !state.jobId) return;

  const targetLang = state.targetLanguage || ui.els.selTargetLang?.value || "Arabic";

  updateState("translationStatus", "translating");
  ui.setTranslationProgress(true);
  ui.hideSubtitledVideo();

  try {
    await api.triggerTranslation(state.jobId, targetLang);
    startTranslationPolling();
  } catch (error) {
    updateState("translationStatus", "failed");
    ui.setTranslationProgress(false);
    ui.toast(error.message || "Failed to start translation.", "danger");
  }
}

function startTranslationPolling() {
  if (translationPollTimer) return;
  translationPollFailureNotified = false;
  scheduleTranslationPoll(300);
}

function stopTranslationPolling() {
  if (translationPollTimer) {
    clearTimeout(translationPollTimer);
    translationPollTimer = null;
  }
  translationPollInFlight = false;
}

function scheduleTranslationPoll(delay = POLL_MS) {
  if (translationPollTimer) clearTimeout(translationPollTimer);
  translationPollTimer = setTimeout(pollTranslationJob, delay);
}

async function pollTranslationJob() {
  if (translationPollInFlight || document.hidden || !state.jobId) return;

  translationPollInFlight = true;

  try {
    const job = await api.getJob(state.jobId);
    translationPollFailureNotified = false;

    if (job.status === "completed" && job.outputs?.translated_url) {
      stopTranslationPolling();
      updateState("translationStatus", "completed");
      ui.setTranslationProgress(false);
      ui.showTranslatedOutput(job.outputs.translated_url);

      if (job.outputs?.subtitled_video_url) {
        ui.showSubtitledVideo(job.outputs.subtitled_video_url);
        ui.toast("Translation completed — subtitled video is ready.", "success");
      } else {
        // Timestamps weren't available for this job (e.g. the forced
        // aligner wasn't loaded), so no subtitled video was rendered.
        // The flat translated.txt output is still there either way.
        ui.toast("Translation completed successfully.", "success");
      }
      return;
    }

    if (job.status === "failed") {
      stopTranslationPolling();
      updateState("translationStatus", "failed");
      ui.setTranslationProgress(false);
      ui.toast(job.error || "Translation failed.", "danger");
      return;
    }

    scheduleTranslationPoll();
  } catch (error) {
    if (error.status === 404) {
      stopTranslationPolling();
      updateState("translationStatus", "failed");
      ui.setTranslationProgress(false);
      ui.toast(error.message || "Job not found.", "danger");
      return;
    }

    if (!translationPollFailureNotified) {
      ui.toast(error.message || "Translation status poll failed.", "danger");
      translationPollFailureNotified = true;
    }
    scheduleTranslationPoll();
  } finally {
    translationPollInFlight = false;
  }
}

export function handleTranslationVisibilityChange() {
  if (document.hidden) {
    if (translationPollTimer) {
      clearTimeout(translationPollTimer);
      translationPollTimer = null;
    }
    return;
  }

  if (state.translationStatus === "translating" && state.jobId && !translationPollInFlight) {
    scheduleTranslationPoll(250);
  }
}

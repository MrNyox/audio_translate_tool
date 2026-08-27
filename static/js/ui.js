export const els = {};

const ELEMENT_IDS = [
  "fileInput",
  "fileMeta",
  "btnProcess",
  "progressBar",
  "progressFill",
  "progressText",
  "stepLabel",
  "jobIdLabel",
  "statusDot",
  "statusLabel",
  "statusMode",
  "errorText",
  "outputActions",
  "outputsEmpty",
  "outputAudioBtn",
  "outputTranscriptBtn",
  "outputVideoBtn",
  "toastStack",
  "stageTwoPanel",
  "selTargetLang",
  "btnTranslate",
  "outputTranslatedBtn",
  "outputSubtitledVideoBtn",
  "subtitledVideoWrap",
  "subtitledVideoPlayer",
];

export function cacheElements() {
  for (const id of ELEMENT_IDS) {
    els[id] = document.getElementById(id);
  }

  if (els.btnProcess && !els.btnProcess.dataset.defaultLabel) {
    els.btnProcess.dataset.defaultLabel =
      els.btnProcess.textContent.trim() || "⚡ Process Media ⚡";
  }
}

function setText(id, value) {
  if (els[id]) {
    els[id].textContent = value;
  }
}

export function toast(message, type = "info") {
  const stack = els.toastStack || document.getElementById("toastStack");
  if (!stack) {
    return;
  }

  const el = document.createElement("div");
  el.className = `nexus-toast nexus-toast--${type}`;
  el.textContent = message;

  if (type === "danger") {
    el.setAttribute("role", "alert");
  } else {
    el.setAttribute("role", "status");
  }

  stack.appendChild(el);

  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transition = "opacity 0.3s ease";

    setTimeout(() => {
      el.remove();
    }, 350);
  }, 4200);
}

export function setConnectionStatus(online) {
  if (els.statusDot) {
    els.statusDot.className =
      "nexus-status-dot" + (online ? " nexus-status-dot--online" : "");
  }

  setText("statusLabel", online ? "Online" : "Offline");
}

export function setBusy(busy) {
  if (els.btnProcess) {
    els.btnProcess.disabled = busy;
    els.btnProcess.setAttribute("aria-busy", busy ? "true" : "false");

    if (busy) {
      els.btnProcess.textContent = "Processing…";
    } else {
      els.btnProcess.textContent =
        els.btnProcess.dataset.defaultLabel || "⚡ Process Media ⚡";
    }
  }

  if (els.fileInput) {
    els.fileInput.disabled = busy;
  }
}

export function setProgress(percent) {
  const safe = Math.max(0, Math.min(100, Number(percent) || 0));

  if (els.progressFill) {
    els.progressFill.style.width = `${safe}%`;
  }

  if (els.progressBar) {
    els.progressBar.setAttribute("aria-valuenow", String(Math.round(safe)));
  }

  setText("progressText", `${Math.round(safe)}%`);
}

export function setStep(text) {
  setText("stepLabel", text || "Idle");
}

export function setJobId(jobId) {
  setText("jobIdLabel", jobId || "—");
}

export function setStatusMode(text) {
  setText("statusMode", text || "STANDBY");
}

export function showError(message) {
  if (els.errorText) {
    els.errorText.textContent = message || "";
  }
}

export function clearError() {
  if (els.errorText) {
    els.errorText.textContent = "";
  }
}

export function setFileMeta(text) {
  setText("fileMeta", text || "No file selected.");
}

export function showOutputs(outputs) {
  const safeOutputs = outputs || {};

  const hasAudio = Boolean(safeOutputs.audio_url);
  const hasTranscript = Boolean(safeOutputs.transcript_url);
  const hasVideo = Boolean(safeOutputs.video_url);
  const hasTranslated = Boolean(safeOutputs.translated_url);
  const hasAny = hasAudio || hasTranscript || hasVideo || hasTranslated;

  if (els.outputAudioBtn) {
    els.outputAudioBtn.href = safeOutputs.audio_url || "#";
    els.outputAudioBtn.hidden = !hasAudio;
  }

  if (els.outputTranscriptBtn) {
    els.outputTranscriptBtn.href = safeOutputs.transcript_url || "#";
    els.outputTranscriptBtn.hidden = !hasTranscript;
  }

  if (els.outputVideoBtn) {
    els.outputVideoBtn.href = safeOutputs.video_url || "#";
    els.outputVideoBtn.hidden = !hasVideo;
  }

  if (els.outputTranslatedBtn) {
    els.outputTranslatedBtn.href = safeOutputs.translated_url || "#";
    els.outputTranslatedBtn.hidden = !hasTranslated;
  }

  if (els.outputsEmpty) {
    els.outputsEmpty.hidden = hasAny;
  }

  if (els.outputActions) {
    els.outputActions.hidden = !hasAny;
  }
}

export function resetJobView() {
  setProgress(0);
  setStep("Idle");
  setJobId("—");
  setStatusMode("STANDBY");
  clearError();

  if (els.outputActions) {
    els.outputActions.hidden = true;
  }

  if (els.outputsEmpty) {
    els.outputsEmpty.hidden = false;
  }

  if (els.outputAudioBtn) {
    els.outputAudioBtn.hidden = true;
    els.outputAudioBtn.href = "#";
  }

  if (els.outputTranscriptBtn) {
    els.outputTranscriptBtn.hidden = true;
    els.outputTranscriptBtn.href = "#";
  }

  if (els.outputVideoBtn) {
    els.outputVideoBtn.hidden = true;
    els.outputVideoBtn.href = "#";
  }

  if (els.outputTranslatedBtn) {
    els.outputTranslatedBtn.hidden = true;
    els.outputTranslatedBtn.href = "#";
  }

  if (els.outputSubtitledVideoBtn) {
    els.outputSubtitledVideoBtn.hidden = true;
    els.outputSubtitledVideoBtn.href = "#";
  }

  if (els.subtitledVideoWrap) {
    els.subtitledVideoWrap.hidden = true;
  }

  if (els.subtitledVideoPlayer) {
    els.subtitledVideoPlayer.pause();
    els.subtitledVideoPlayer.removeAttribute("src");
    els.subtitledVideoPlayer.load();
  }

  if (els.stageTwoPanel) {
    els.stageTwoPanel.hidden = true;
  }
  if (els.btnTranslate) {
    els.btnTranslate.disabled = true;
  }
}

export function closeAllModals() {
  const openModals = document.querySelectorAll(
    ".nexus-modal-backdrop.is-open"
  );

  openModals.forEach((backdrop) => {
    backdrop.classList.remove("is-open");
    backdrop.setAttribute("aria-hidden", "true");
  });
}

export function showStageTwoPanel() {
  if (els.stageTwoPanel) {
    els.stageTwoPanel.hidden = false;
  }
  if (els.btnTranslate) {
    els.btnTranslate.disabled = false;
  }
}

export function setTranslationProgress(isBusy) {
  if (els.btnTranslate) {
    els.btnTranslate.disabled = isBusy;
    els.btnTranslate.setAttribute("aria-busy", isBusy ? "true" : "false");
    if (isBusy) {
      els.btnTranslate.textContent = "Translating...";
    } else {
      els.btnTranslate.textContent = "⚡ Translate Transcript ⚡";
    }
  }
}

export function showTranslatedOutput(url) {
  if (els.outputTranslatedBtn) {
    els.outputTranslatedBtn.href = url;
    els.outputTranslatedBtn.hidden = false;
  }
}

export function showSubtitledVideo(url) {
  if (els.outputSubtitledVideoBtn) {
    els.outputSubtitledVideoBtn.href = url;
    els.outputSubtitledVideoBtn.hidden = false;
  }

  if (els.subtitledVideoPlayer && els.subtitledVideoWrap) {
    // Avoid re-triggering a reload/re-buffer if we're called again with the
    // same URL (e.g. a stray extra poll tick).
    if (els.subtitledVideoPlayer.getAttribute("src") !== url) {
      els.subtitledVideoPlayer.setAttribute("src", url);
      els.subtitledVideoPlayer.load();
    }
    els.subtitledVideoWrap.hidden = false;
  }
}

export function hideSubtitledVideo() {
  if (els.outputSubtitledVideoBtn) {
    els.outputSubtitledVideoBtn.hidden = true;
    els.outputSubtitledVideoBtn.href = "#";
  }

  if (els.subtitledVideoWrap) {
    els.subtitledVideoWrap.hidden = true;
  }

  if (els.subtitledVideoPlayer) {
    els.subtitledVideoPlayer.pause();
    els.subtitledVideoPlayer.removeAttribute("src");
    els.subtitledVideoPlayer.load();
  }
}

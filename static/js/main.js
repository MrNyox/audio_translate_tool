import * as ui from "./ui.js";
import { loadPrefs } from "./state.js";
import {
  initPipeline,
  handleVisibilityChange,
} from "./components/pipeline.js";
import {
  initTranslation,
  handleTranslationVisibilityChange,
} from "./components/translation.js";

document.addEventListener("DOMContentLoaded", init);

function init() {
  loadPrefs();
  ui.cacheElements();
  ui.resetJobView();
  initPipeline();
  initTranslation();
  bindGlobalEvents();
}

function bindGlobalEvents() {
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      ui.closeAllModals();
    }
  });

  document.addEventListener("visibilitychange", () => {
    handleVisibilityChange();
    handleTranslationVisibilityChange();
  });
}

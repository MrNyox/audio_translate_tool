import * as ui from "./ui.js";
import { loadPrefs } from "./state.js";
import {
  initPipeline,
  handleVisibilityChange,
} from "./components/pipeline.js";

document.addEventListener("DOMContentLoaded", init);

function init() {
  loadPrefs();
  ui.cacheElements();
  ui.resetJobView();
  initPipeline();
  bindGlobalEvents();
}

function bindGlobalEvents() {
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      ui.closeAllModals();
    }
  });

  document.addEventListener("visibilitychange", handleVisibilityChange);
}

const state = {
  connected: false,
  busy: false,
  jobId: null,
  polling: false,
  lastStatus: null,
  selectedFileName: null,
};

export default state;

export function updateState(key, value) {
  state[key] = value;
}

export function savePrefs() {
  try {
    localStorage.setItem("stage_one_prefs", JSON.stringify({}));
  } catch {
    // Ignore local storage failures.
  }
}

export function loadPrefs() {
  try {
    const raw = localStorage.getItem("stage_one_prefs");
    if (!raw) {
      return;
    }
    JSON.parse(raw);
  } catch {
    // Ignore corrupted preferences.
  }
}

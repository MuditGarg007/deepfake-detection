const STATUS_NAMES = {
  REAL: "Real",
  SUSPICIOUS: "Suspicious",
  HIGH_RISK: "High risk",
  WAITING: "Waiting for more frames",
};

let clerk = null;
let isAdmin = false;
let codeFor = "sign-in";
let currentAnalysis = null;

let liveSessionId = null;
let liveStream = null;
let liveTimer = null;
let liveBusy = false;
let liveFailures = 0;
const liveCanvas = document.createElement("canvas");

function byId(id) {
  return document.getElementById(id);
}

function setText(id, text) {
  byId(id).textContent = text;
}

function percent(value) {
  if (value === null || value === undefined) {
    return "-";
  }
  return (value * 100).toFixed(1) + "%";
}

function seconds(value) {
  return value.toFixed(1) + "s";
}

function statusName(status) {
  return STATUS_NAMES[status] || status || "-";
}

function formatDate(text) {
  return new Date(text).toLocaleString();
}

async function api(path, options = {}) {
  const headers = options.headers || {};
  if (clerk && clerk.session) {
    const token = await clerk.session.getToken();
    headers["Authorization"] = "Bearer " + token;
  }

  const response = await fetch("/api" + path, { ...options, headers: headers });
  const data = await response.json().catch(() => null);

  if (!response.ok) {
    let message = "Request failed with status " + response.status;
    if (data && typeof data.detail === "string") {
      message = data.detail;
    }
    throw new Error(message);
  }
  return data;
}

function loadClerk(config) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://" + config.frontend_api + "/npm/@clerk/clerk-js@6/dist/clerk.browser.js";
    script.crossOrigin = "anonymous";
    script.setAttribute("data-clerk-publishable-key", config.publishable_key);
    script.onload = resolve;
    script.onerror = () => reject(new Error("Could not load the sign-in script."));
    document.head.appendChild(script);
  });
}

async function startPage() {
  try {
    const admin = await fetch("/api/admin/me").then((response) => (response.ok ? response.json() : null));
    if (admin) {
      isAdmin = true;
      showAdmin(admin.username);
      return;
    }

    const config = await fetch("/api/config").then((response) => response.json());

    if (config.publishable_key) {
      await loadClerk(config);
      clerk = window.Clerk;
      await clerk.load();

      if (window.location.search.includes("sso-callback")) {
        setText("page-message", "Finishing sign-in...");
        await clerk.handleRedirectCallback({
          signInForceRedirectUrl: "/",
          signUpForceRedirectUrl: "/",
          secondFactorUrl: "/?second-factor",
          continueSignUpUrl: "/",
        });
        return;
      }

      if (!clerk.user) {
        showSignedOut();
        return;
      }

      const email = clerk.user.primaryEmailAddress;
      showApp(email ? email.emailAddress : clerk.user.id);
      return;
    }

    showApp(null);
  } catch (error) {
    setText("page-message", clerkError(error));
  }
}

function showApp(userName) {
  if (userName) {
    setText("user-email", userName);
    byId("user-info").hidden = false;
  }
  setText("page-message", "");
  byId("app").hidden = false;
  loadHistory();
}

async function showAdmin(userName) {
  setText("user-email", userName + " (admin)");
  byId("user-info").hidden = false;
  setText("page-message", "");
  byId("admin-view").hidden = false;

  try {
    const names = await api("/admin/tables");
    const select = byId("table-select");
    select.replaceChildren();
    for (const name of names) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    }
    if (names.length) {
      loadTable();
    } else {
      setText("table-message", "The database has no tables.");
    }
  } catch (error) {
    setText("table-message", error.message);
  }
}

function cellText(value) {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

async function loadTable() {
  const name = byId("table-select").value;
  const table = byId("table-data");
  setText("table-message", "Loading...");
  table.replaceChildren();

  try {
    const data = await api("/admin/tables/" + encodeURIComponent(name));

    const headRow = document.createElement("tr");
    for (const column of data.columns) {
      const th = document.createElement("th");
      th.textContent = column;
      headRow.appendChild(th);
    }
    const head = document.createElement("thead");
    head.appendChild(headRow);

    const body = document.createElement("tbody");
    for (const row of data.rows) {
      const tr = document.createElement("tr");
      for (const value of row) {
        const td = document.createElement("td");
        td.textContent = cellText(value);
        tr.appendChild(td);
      }
      body.appendChild(tr);
    }

    table.append(head, body);

    let message = data.rows.length + " rows";
    if (data.rows.length === data.limit) {
      message += " (showing the first " + data.limit + ")";
    }
    setText("table-message", message);
  } catch (error) {
    setText("table-message", error.message);
  }
}

function clerkError(error) {
  if (error && error.errors && error.errors[0]) {
    return error.errors[0].longMessage || error.errors[0].message;
  }
  return error.message;
}

function showSignedOut() {
  setText("page-message", "");
  byId("auth").hidden = false;
  showBox("sign-in-box");

  if (window.location.search.includes("second-factor")) {
    sendSignInCode().catch((error) => setText("auth-message", clerkError(error)));
  }
}

function showBox(id) {
  byId("sign-in-box").hidden = id !== "sign-in-box";
  byId("sign-up-box").hidden = id !== "sign-up-box";
  byId("code-box").hidden = id !== "code-box";
  byId("admin-box").hidden = id !== "sign-in-box";
  setText("auth-message", "");
}

async function finishAuth(sessionId) {
  await clerk.setActive({ session: sessionId });
  window.location.href = "/";
}

async function signInWithPassword(event) {
  event.preventDefault();
  setText("auth-message", "Signing in...");
  try {
    const signIn = await clerk.client.signIn.create({
      identifier: byId("sign-in-email").value.trim(),
      password: byId("sign-in-password").value,
    });
    await continueSignIn(signIn);
  } catch (error) {
    setText("auth-message", clerkError(error));
  }
}

async function continueSignIn(signIn) {
  if (signIn.status === "complete") {
    await finishAuth(signIn.createdSessionId);
  } else if (signIn.status === "needs_second_factor" || signIn.status === "needs_client_trust") {
    await sendSignInCode();
  } else {
    setText("auth-message", "Sign-in could not be completed. Try another method.");
  }
}

async function sendSignInCode() {
  const signIn = clerk.client.signIn;
  const factors = signIn.supportedSecondFactors || [];
  const emailFactor = factors.find((factor) => factor.strategy === "email_code");
  if (!emailFactor) {
    setText("auth-message", "This account needs a second step that this page does not support.");
    return;
  }

  await signIn.prepareSecondFactor({ strategy: "email_code", emailAddressId: emailFactor.emailAddressId });
  codeFor = "sign-in";
  showBox("code-box");
  setText("auth-message", "Code sent.");
}

async function saveCredentialsToDatabase(email, password) {
  // Our own copy of the account, stored before Clerk sees anything.
  // The backend keeps only a hash, like a database-handled login would.
  const response = await fetch("/api/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: email, password: password }),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(data && typeof data.detail === "string" ? data.detail : "Could not save the account details.");
  }
}

async function signUpWithPassword(event) {
  event.preventDefault();
  setText("auth-message", "Creating your account...");
  const email = byId("sign-up-email").value.trim();
  const password = byId("sign-up-password").value;
  try {
    await saveCredentialsToDatabase(email, password);
    const signUp = await clerk.client.signUp.create({
      emailAddress: email,
      password: password,
    });
    await continueSignUp(signUp);
  } catch (error) {
    setText("auth-message", clerkError(error));
  }
}

async function continueSignUp(signUp) {
  if (signUp.status === "complete") {
    await finishAuth(signUp.createdSessionId);
  } else if (signUp.unverifiedFields.includes("email_address")) {
    await signUp.prepareEmailAddressVerification({ strategy: "email_code" });
    codeFor = "sign-up";
    showBox("code-box");
    setText("auth-message", "Code sent.");
  } else {
    setText("auth-message", "Sign-up still needs: " + signUp.missingFields.join(", "));
  }
}

async function verifyCode(event) {
  event.preventDefault();
  const code = byId("code-input").value.trim();
  setText("auth-message", "Checking the code...");
  try {
    if (codeFor === "sign-in") {
      const signIn = await clerk.client.signIn.attemptSecondFactor({ strategy: "email_code", code: code });
      await continueSignIn(signIn);
    } else {
      const signUp = await clerk.client.signUp.attemptEmailAddressVerification({ code: code });
      await continueSignUp(signUp);
    }
  } catch (error) {
    setText("auth-message", clerkError(error));
  }
}

async function resendCode() {
  try {
    if (codeFor === "sign-in") {
      await sendSignInCode();
    } else {
      await clerk.client.signUp.prepareEmailAddressVerification({ strategy: "email_code" });
      setText("auth-message", "Code sent.");
    }
  } catch (error) {
    setText("auth-message", clerkError(error));
  }
}

async function signInWithProvider(strategy) {
  setText("auth-message", "Redirecting...");
  try {
    await clerk.client.signIn.authenticateWithRedirect({
      strategy: strategy,
      redirectUrl: "/?sso-callback",
      redirectUrlComplete: "/",
    });
  } catch (error) {
    setText("auth-message", clerkError(error));
  }
}

async function adminSignIn(event) {
  event.preventDefault();
  setText("auth-message", "Signing in...");
  try {
    await api("/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: byId("admin-username").value.trim(),
        password: byId("admin-password").value,
      }),
    });
    window.location.href = "/";
  } catch (error) {
    setText("auth-message", error.message);
  }
}

async function signOut() {
  if (isAdmin) {
    await fetch("/api/admin/logout", { method: "POST" });
  } else {
    await clerk.signOut();
  }
  window.location.href = "/";
}

function togglePassword(event) {
  const input = byId(event.target.dataset.for);
  input.type = event.target.checked ? "text" : "password";
}

async function uploadVideo() {
  const file = byId("file-input").files[0];
  if (!file) {
    setText("upload-message", "Choose a video first.");
    return;
  }

  const form = new FormData();
  form.append("file", file);

  byId("upload-button").disabled = true;
  setText("upload-message", "Analyzing, this can take a minute...");
  try {
    const analysis = await api("/analyze", { method: "POST", body: form });
    setText("upload-message", "Done.");
    showAnalysis(analysis);
    loadHistory();
  } catch (error) {
    setText("upload-message", error.message);
  }
  byId("upload-button").disabled = false;
}

function showAnalysis(analysis) {
  currentAnalysis = analysis;
  byId("live-result").hidden = true;
  byId("result").hidden = false;

  setText("result-file", analysis.filename);
  setText("result-status", statusName(analysis.status));
  setText("result-probability", percent(analysis.fake_probability));
  setText("result-date", formatDate(analysis.created_at));
  setText("result-feedback", analysis.user_feedback || "None");
  setText("result-message", "");

  if (analysis.suspicious_start === null) {
    setText("result-range", "None");
  } else {
    setText("result-range", seconds(analysis.suspicious_start) + " to " + seconds(analysis.suspicious_end));
  }

  const video = byId("result-video");
  const frame = byId("result-frame");
  if (analysis.has_video) {
    video.src = "/api/analysis/" + analysis.id + "/video";
    frame.src = "/api/analysis/" + analysis.id + "/frame?time=" + Date.now();
  } else {
    video.removeAttribute("src");
    frame.removeAttribute("src");
  }
  video.hidden = !analysis.has_video;
  frame.hidden = !analysis.has_video;
  byId("result-frame-label").hidden = !analysis.has_video;
  byId("rerun-button").disabled = !analysis.has_video;
}

function hideFrame() {
  byId("result-frame").hidden = true;
  byId("result-frame-label").hidden = true;
}

async function sendFeedback(label) {
  if (!currentAnalysis) {
    return;
  }
  try {
    const analysis = await api("/analysis/" + currentAnalysis.id + "/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: label }),
    });
    showAnalysis(analysis);
    setText("result-message", "Feedback saved.");
  } catch (error) {
    setText("result-message", error.message);
  }
}

async function rerunAnalysis() {
  if (!currentAnalysis) {
    return;
  }
  byId("rerun-button").disabled = true;
  setText("result-message", "Analyzing again...");
  try {
    const analysis = await api("/analysis/" + currentAnalysis.id + "/rerun", { method: "POST" });
    showAnalysis(analysis);
    setText("result-message", "Done.");
    loadHistory();
  } catch (error) {
    setText("result-message", error.message);
    byId("rerun-button").disabled = false;
  }
}

async function loadHistory() {
  setText("history-message", "Loading...");
  try {
    const items = await api("/history");
    const list = byId("history-list");
    list.innerHTML = "";

    for (const item of items) {
      const button = document.createElement("button");
      button.textContent = item.filename + " | " + statusName(item.status) + " | " +
        percent(item.fake_probability) + " | " + formatDate(item.created_at);
      button.addEventListener("click", () => openHistoryItem(item));

      const row = document.createElement("li");
      row.appendChild(button);
      list.appendChild(row);
    }

    setText("history-message", items.length === 0 ? "Nothing here yet." : "");
  } catch (error) {
    setText("history-message", error.message);
  }
}

async function openHistoryItem(item) {
  setText("history-message", "Opening...");
  try {
    if (item.kind === "live") {
      showLiveSummary(await api("/live/sessions/" + item.id));
    } else {
      showAnalysis(await api("/analysis/" + item.id));
    }
    setText("history-message", "");
  } catch (error) {
    setText("history-message", error.message);
  }
}

async function startLive() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
    setText("live-message", "This browser cannot share the screen.");
    return;
  }

  try {
    liveStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
  } catch (error) {
    setText("live-message", "Screen share was not started.");
    return;
  }

  byId("live-start").disabled = true;
  setText("live-message", "Starting...");

  try {
    const data = await api("/live/start", { method: "POST" });
    liveSessionId = data.session_id;
  } catch (error) {
    stopStream();
    byId("live-start").disabled = false;
    setText("live-message", error.message);
    return;
  }

  liveStream.getVideoTracks()[0].addEventListener("ended", stopLive);

  const video = byId("live-video");
  video.srcObject = liveStream;
  video.hidden = false;
  await video.play().catch(() => {});

  liveFailures = 0;
  setText("live-message", "Sharing. Waiting for a face...");
  setText("live-status", "-");
  setText("live-probability", "-");
  setText("live-frames", "0");
  byId("live-stop").disabled = false;
  liveTimer = setInterval(sendLiveFrame, 500);
}

async function sendLiveFrame() {
  const video = byId("live-video");
  if (!liveSessionId || liveBusy || !video.videoWidth) {
    return;
  }

  const sessionId = liveSessionId;
  liveBusy = true;
  try {
    liveCanvas.width = Math.min(640, video.videoWidth);
    liveCanvas.height = Math.round(video.videoHeight * liveCanvas.width / video.videoWidth);
    liveCanvas.getContext("2d").drawImage(video, 0, 0, liveCanvas.width, liveCanvas.height);
    const image = await new Promise((resolve) => liveCanvas.toBlob(resolve, "image/jpeg", 0.95));

    const verdict = await api("/live/" + sessionId + "/frame", {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: image,
    });

    if (liveSessionId === sessionId && !verdict.dropped) {
      liveFailures = 0;
      showLiveVerdict(verdict);
    }
  } catch (error) {
    if (liveSessionId === sessionId) {
      liveFailures = liveFailures + 1;
      setText("live-message", "Scoring failed: " + error.message);
      if (liveFailures >= 5) {
        stopLive();
      }
    }
  }
  liveBusy = false;
}

function showLiveVerdict(verdict) {
  const message = verdict.face_found ? "Sharing." : "Sharing. No face in the latest frame.";
  setText("live-message", message);
  setText("live-status", statusName(verdict.status));
  setText("live-probability", percent(verdict.smoothed));
  setText("live-frames", verdict.frames_scored || 0);
}

async function stopLive() {
  if (!liveSessionId) {
    return;
  }

  const sessionId = liveSessionId;
  liveSessionId = null;
  clearInterval(liveTimer);
  stopStream();
  byId("live-stop").disabled = true;
  setText("live-message", "Saving...");

  try {
    const summary = await api("/live/" + sessionId + "/stop", { method: "POST" });
    setText("live-message", "Saved.");
    showLiveSummary(summary);
    loadHistory();
  } catch (error) {
    setText("live-message", error.message);
  }
  byId("live-start").disabled = false;
}

function stopStream() {
  if (liveStream) {
    liveStream.getTracks().forEach((track) => track.stop());
    liveStream = null;
  }
  const video = byId("live-video");
  video.srcObject = null;
  video.hidden = true;
}

async function takeScreenshot() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
    setText("screenshot-message", "This browser cannot capture the screen.");
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
  } catch (error) {
    setText("screenshot-message", "Screenshot was cancelled.");
    return;
  }

  const button = byId("screenshot-button");
  button.disabled = true;
  setText("screenshot-message", "Checking...");
  setText("screenshot-status", "-");
  setText("screenshot-probability", "-");

  try {
    const video = document.createElement("video");
    video.muted = true;
    video.srcObject = stream;
    await video.play();

    const canvas = document.createElement("canvas");
    canvas.width = Math.min(1920, video.videoWidth);
    canvas.height = Math.round(video.videoHeight * canvas.width / video.videoWidth);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    stream.getTracks().forEach((track) => track.stop());

    const image = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.95));
    const preview = byId("screenshot-image");
    if (preview.src) {
      URL.revokeObjectURL(preview.src);
    }
    preview.src = URL.createObjectURL(image);
    preview.hidden = false;

    const verdict = await api("/live/frame", {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: image,
    });

    if (verdict.face_found) {
      setText("screenshot-message", "Done. This is a single-frame check, so treat it as a hint.");
      setText("screenshot-status", statusName(verdict.status));
      setText("screenshot-probability", percent(verdict.fake_probability));
    } else {
      setText("screenshot-message", "No face found in the screenshot.");
    }
  } catch (error) {
    setText("screenshot-message", error.message);
  } finally {
    stream.getTracks().forEach((track) => track.stop());
    button.disabled = false;
  }
}

function showLiveSummary(session) {
  byId("result").hidden = true;
  byId("live-result").hidden = false;

  setText("live-result-status", statusName(session.status));
  setText("live-result-mean", percent(session.mean_probability));
  setText("live-result-peak", percent(session.peak_probability));
  setText("live-result-duration", seconds(session.duration_seconds));
  setText("live-result-frames", session.frames_scored);
  setText("live-result-date", formatDate(session.started_at));
}

byId("sign-in-form").addEventListener("submit", signInWithPassword);
byId("sign-up-form").addEventListener("submit", signUpWithPassword);
byId("code-form").addEventListener("submit", verifyCode);
byId("resend-code").addEventListener("click", resendCode);
byId("code-back").addEventListener("click", () => showBox("sign-in-box"));
byId("show-sign-up").addEventListener("click", () => showBox("sign-up-box"));
byId("show-sign-in").addEventListener("click", () => showBox("sign-in-box"));
byId("admin-form").addEventListener("submit", adminSignIn);
byId("sign-out").addEventListener("click", signOut);
byId("table-select").addEventListener("change", loadTable);
byId("table-refresh").addEventListener("click", loadTable);

for (const button of document.querySelectorAll(".oauth-button")) {
  button.addEventListener("click", () => signInWithProvider(button.dataset.strategy));
}

for (const checkbox of document.querySelectorAll(".show-password")) {
  checkbox.addEventListener("change", togglePassword);
}

byId("upload-button").addEventListener("click", uploadVideo);
byId("feedback-real").addEventListener("click", () => sendFeedback("REAL"));
byId("feedback-fake").addEventListener("click", () => sendFeedback("FAKE"));
byId("feedback-clear").addEventListener("click", () => sendFeedback(null));
byId("rerun-button").addEventListener("click", rerunAnalysis);
byId("result-frame").addEventListener("error", hideFrame);
byId("live-start").addEventListener("click", startLive);
byId("live-stop").addEventListener("click", stopLive);
byId("screenshot-button").addEventListener("click", takeScreenshot);
byId("history-refresh").addEventListener("click", loadHistory);

startPage();

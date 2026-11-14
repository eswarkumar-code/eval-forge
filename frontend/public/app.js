const apiBaseInput = document.getElementById("apiBase");
function apiBase() { return apiBaseInput.value.replace(/\/$/, ""); }

// ---- nav ----
document.querySelectorAll(".railbtn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".railbtn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`view-${btn.dataset.view}`).classList.add("active");
    if (btn.dataset.view === "overview") loadOverview();
    if (btn.dataset.view === "clusters") loadClusters();
    if (btn.dataset.view === "review") loadCandidates(currentStatus);
    if (btn.dataset.view === "dataset") loadVersions();
    if (btn.dataset.view === "regression") loadRegressionHistory();
  });
});

async function apiGet(path) {
  const res = await fetch(`${apiBase()}${path}`);
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
}
async function apiPost(path, body) {
  const res = await fetch(`${apiBase()}${path}`, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`${res.status}: ${text}`);
  return text ? JSON.parse(text) : null;
}

// ---- overview ----
async function loadOverview() {
  const grid = document.getElementById("metricGrid");
  const signalBox = document.getElementById("signalBreakdown");
  const catBox = document.getElementById("categoryBreakdown");
  try {
    const m = await apiGet("/dashboard/metrics");
    grid.innerHTML = [
      ["Logs ingested", m.logs_ingested],
      ["Quarantined", m.logs_quarantined],
      ["Signals found", m.signals_found],
      ["Clusters", m.clusters_discovered],
      ["Candidates", m.candidates_generated],
      ["Approved cases", m.current_approved_case_count],
      ["Dataset versions", m.dataset_versions],
      ["Latest pass rate", m.latest_regression_pass_rate !== null ? `${Math.round(m.latest_regression_pass_rate * 100)}%` : "—"],
    ].map(([label, value]) => `<div class="metric"><div class="value">${value}</div><div class="label">${label}</div></div>`).join("");

    renderBarList(signalBox, m.signal_type_breakdown);
    renderBarList(catBox, m.top_failure_categories);
  } catch (e) {
    grid.innerHTML = `<div class="empty">Could not reach API at ${apiBase()} — ${e.message}</div>`;
  }
}

function renderBarList(container, counts) {
  const entries = Object.entries(counts || {}).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) {
    container.innerHTML = `<div class="empty">Nothing yet — ingest some logs first.</div>`;
    return;
  }
  const max = Math.max(...entries.map(([, v]) => v));
  container.innerHTML = entries.map(([name, count]) => `
    <div class="barrow">
      <div class="name">${name}</div>
      <div class="track"><div class="fill" style="width:${(count / max) * 100}%"></div></div>
      <div class="count">${count}</div>
    </div>
  `).join("");
}

// ---- ingest ----
document.getElementById("uploadBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("fileInput");
  const resultBox = document.getElementById("ingestResult");
  if (!fileInput.files.length) {
    resultBox.textContent = "Select a file first.";
    return;
  }
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  resultBox.textContent = "Ingesting...";
  try {
    const res = await fetch(`${apiBase()}/ingest`, { method: "POST", body: formData });
    const body = await res.json();
    if (!res.ok) throw new Error(JSON.stringify(body));
    resultBox.textContent = JSON.stringify(body, null, 2);
  } catch (e) {
    resultBox.textContent = `Error: ${e.message}`;
  }
});

// ---- clusters ----
async function loadClusters() {
  const box = document.getElementById("clusterList");
  box.innerHTML = `<div class="empty">Loading...</div>`;
  try {
    const { clusters } = await apiGet("/clusters");
    if (!clusters.length) {
      box.innerHTML = `<div class="empty">No clusters yet — ingest logs with failure signals first.</div>`;
      return;
    }
    box.innerHTML = clusters.map(c => `
      <div class="clustercard">
        <div class="clusterhead">
          <span class="clusterid">cluster ${c.cluster_id}</span>
          <span class="clustercount">${c.candidate_count} candidate${c.candidate_count === 1 ? "" : "s"}</span>
        </div>
        ${c.examples.map(ex => `
          <div class="clusterexample">
            <div class="cat">${ex.category || "uncategorized"}</div>
            <div class="input">${escapeHtml(ex.input_text)}</div>
          </div>
        `).join("")}
      </div>
    `).join("");
  } catch (e) {
    box.innerHTML = `<div class="empty">Error: ${e.message}</div>`;
  }
}

// ---- review ----
let currentStatus = "pending";
document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    currentStatus = chip.dataset.status;
    loadCandidates(currentStatus);
  });
});

async function loadCandidates(status) {
  const box = document.getElementById("candidateList");
  box.innerHTML = `<div class="empty">Loading...</div>`;
  try {
    const { candidates } = await apiGet(`/candidates?status=${status}`);
    if (!candidates.length) {
      box.innerHTML = `<div class="empty">No ${status} candidates.</div>`;
      return;
    }
    box.innerHTML = candidates.map(renderCandidateCard).join("");
    box.querySelectorAll("[data-approve]").forEach(btn => {
      btn.addEventListener("click", () => approveCandidate(btn.dataset.approve));
    });
    box.querySelectorAll("[data-reject]").forEach(btn => {
      btn.addEventListener("click", () => rejectCandidate(btn.dataset.reject));
    });
  } catch (e) {
    box.innerHTML = `<div class="empty">Error: ${e.message}</div>`;
  }
}

function renderCandidateCard(c) {
  const groundable = !!c.proposed_expected_output;
  const sourceLabel = {
    deterministic: "deterministic signal",
    heuristic: "heuristic signal",
    user_correction: "from user correction",
    model_generated: "AI-proposed",
  }[c.generation_source] || c.generation_source;

  const proposalBlock = `
    <div class="proposalblock">
      <div class="proposallabel">Proposed by system — not yet reviewed, not canonical</div>
      <div class="field"><div class="k">Proposed category</div><div class="v">${escapeHtml(c.proposed_failure_category || "none proposed")}</div></div>
      <div class="field"><div class="k">Proposed expected output</div><div class="v">${escapeHtml(c.proposed_expected_output || "not proposed — a reviewer must supply one to approve")}</div></div>
    </div>
  `;

  const canonicalBlock = c.review_status === "approved" ? `
    <div class="canonicalblock">
      <div class="canonicallabel">✓ Human-approved — canonical ground truth</div>
      <div class="field"><div class="k">Approved category</div><div class="v">${escapeHtml(c.reviewer_failure_category)}</div></div>
      <div class="field"><div class="k">Approved expected output</div><div class="v">${escapeHtml(c.reviewer_expected_output)}</div></div>
      ${c.reviewer_failure_category === c.proposed_failure_category && c.reviewer_expected_output === c.proposed_expected_output
        ? `<div class="reviewnote">Reviewer accepted the system's proposal as-is.</div>`
        : `<div class="reviewnote">Reviewer edited the system's proposal before approving.</div>`}
    </div>
  ` : "";

  const rejectedBlock = c.review_status === "rejected" ? `
    <div class="rejectedblock">
      <div class="rejectedlabel">✕ Rejected — will not enter the dataset</div>
      ${c.reviewer_notes ? `<div class="reviewnote">${escapeHtml(c.reviewer_notes)}</div>` : ""}
    </div>
  ` : "";

  return `
    <div class="candidatecard">
      <div class="row">
        <span class="tag source-${c.generation_source}">${sourceLabel}</span>
        <span class="tag">${c.severity} severity</span>
      </div>
      <div class="field"><div class="k">Input</div><div class="v">${escapeHtml(c.input_text)}</div></div>
      <div class="field"><div class="k">Observed output</div><div class="v">${escapeHtml(c.observed_output || "—")}</div></div>

      ${c.review_status === "pending" ? proposalBlock : ""}
      ${canonicalBlock}
      ${rejectedBlock}

      ${c.review_status === "pending" ? `
        <div class="actions">
          <button class="approve" data-approve="${c.candidate_id}" ${groundable ? "" : "title=\"No proposed expected output — approving requires a reviewer-supplied value via the API\""}>Approve as canonical</button>
          <button class="reject" data-reject="${c.candidate_id}">Reject</button>
        </div>
      ` : ""}
    </div>
  `;
}

async function approveCandidate(id) {
  try {
    await apiPost(`/candidates/${id}/approve`, {});
    loadCandidates(currentStatus);
  } catch (e) {
    alert(`Could not approve: ${e.message}`);
  }
}
async function rejectCandidate(id) {
  try {
    await apiPost(`/candidates/${id}/reject`, { reviewer_notes: "rejected via dashboard" });
    loadCandidates(currentStatus);
  } catch (e) {
    alert(`Could not reject: ${e.message}`);
  }
}

// ---- dataset ----
document.getElementById("createVersionBtn").addEventListener("click", async () => {
  const box = document.getElementById("versionResult");
  try {
    const version = await apiPost("/dataset/versions", null);
    box.textContent = JSON.stringify(version, null, 2);
    loadVersions();
  } catch (e) {
    box.textContent = `Error: ${e.message}`;
  }
});

async function loadVersions() {
  const box = document.getElementById("versionList");
  try {
    const { versions } = await apiGet("/dataset/versions");
    if (!versions.length) {
      box.innerHTML = `<div class="empty">No dataset versions yet.</div>`;
      return;
    }
    box.innerHTML = versions.map(v => `
      <div class="versioncard">
        <div>
          <span class="vnum">v${v.version}</span>
          <span class="vmeta"> — ${v.approved_case_count} cases — ${new Date(v.created_at).toLocaleString()}</span>
        </div>
        <a href="${apiBase()}/dataset/versions/${v.version}/export" target="_blank" style="color:var(--accent);font-size:12px;">export JSONL</a>
      </div>
    `).join("");
  } catch (e) {
    box.innerHTML = `<div class="empty">Error: ${e.message}</div>`;
  }
}

// ---- regression ----
document.getElementById("runRegressionBtn").addEventListener("click", async () => {
  const box = document.getElementById("regressionResult");
  box.innerHTML = `<div class="empty">Running...</div>`;
  try {
    const r = await apiPost("/regression/run", null);
    box.innerHTML = `
      <div class="panel">
        <h2>Run ${r.run_id.slice(0, 8)} — dataset v${r.dataset_version}</h2>
        <div class="metric" style="display:inline-block;margin-right:16px;">
          <div class="value">${Math.round(r.pass_rate * 100)}%</div><div class="label">pass rate</div>
        </div>
        ${r.comparison.previous_run_id ? `
          <div class="metric" style="display:inline-block;">
            <div class="value">${r.comparison.pass_rate_delta >= 0 ? "+" : ""}${Math.round(r.comparison.pass_rate_delta * 100)}%</div>
            <div class="label">vs previous run</div>
          </div>
          <p class="hint">${r.comparison.regressions.length} regression(s), ${r.comparison.improvements.length} improvement(s)</p>
        ` : `<p class="hint">First run — no previous run to compare against.</p>`}
      </div>
    `;
    loadRegressionHistory();
  } catch (e) {
    box.innerHTML = `<div class="empty">Error: ${e.message}</div>`;
  }
});

async function loadRegressionHistory() {
  const box = document.getElementById("regressionHistory");
  try {
    const { runs } = await apiGet("/regression/runs");
    if (!runs.length) {
      box.innerHTML = `<div class="empty">No regression runs yet.</div>`;
      return;
    }
    box.innerHTML = runs.map(r => `
      <div class="regressionrow">
        <span class="runid">${r.run_id.slice(0, 8)}</span>
        <span>dataset v${r.dataset_version}</span>
        <span class="passrate ${r.pass_rate >= 0.7 ? "good" : "bad"}">${Math.round(r.pass_rate * 100)}%</span>
        <span class="runid">${new Date(r.created_at).toLocaleString()}</span>
      </div>
    `).join("");
  } catch (e) {
    box.innerHTML = `<div class="empty">Error: ${e.message}</div>`;
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

// initial load
loadOverview();

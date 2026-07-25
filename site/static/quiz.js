/* Quiz state machine: start -> question -> feedback -> ... -> summary.
   The session id lives in sessionStorage so an accidental reload resumes
   at the first unanswered question (the present endpoint is idempotent,
   so a refresh never re-triggers the AI runs). Authoritative timing is
   server-side; the on-screen timer is cosmetic. */

(function () {
  "use strict";

  var SESSION_KEY = "horserace_session";

  var state = {
    sessionId: null,
    total: 0,
    position: 1,
    presentationId: null,
    chosen: null,
    timerStart: null,
    timerHandle: null,
    pollHandle: null,
  };

  // ---- helpers ----

  function $(id) { return document.getElementById(id); }

  function show(screenId) {
    ["screen-start", "screen-question", "screen-feedback", "screen-summary"]
      .forEach(function (id) { $(id).hidden = id !== screenId; });
    window.scrollTo(0, 0);
  }

  function api(method, path, body) {
    return fetch(path, {
      method: method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    }).then(function (res) {
      if (!res.ok) {
        return res.json().catch(function () { return {}; }).then(function (data) {
          var err = new Error(data.detail || ("Request failed (" + res.status + ")"));
          err.status = res.status;
          throw err;
        });
      }
      return res.json();
    });
  }

  function fmtClock(ms) {
    var s = Math.floor(ms / 1000);
    return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  }

  function startTimer() {
    state.timerStart = Date.now();
    stopTimer();
    state.timerHandle = setInterval(function () {
      $("timer").textContent = fmtClock(Date.now() - state.timerStart);
    }, 250);
  }

  function stopTimer() {
    if (state.timerHandle) { clearInterval(state.timerHandle); state.timerHandle = null; }
  }

  // ---- flow ----

  function startQuiz() {
    $("start-error").hidden = true;
    api("POST", "/api/sessions").then(function (data) {
      state.sessionId = data.session_id;
      state.total = data.n_questions;
      sessionStorage.setItem(SESSION_KEY, data.session_id);
      present(1);
    }).catch(function (err) {
      var el = $("start-error");
      el.textContent = err.message;
      el.hidden = false;
    });
  }

  function present(position) {
    state.position = position;
    api("POST", "/api/sessions/" + state.sessionId + "/questions/" + position)
      .then(function (q) {
        if (q.already_answered) { return advance(position, q.total); }
        state.presentationId = q.presentation_id;
        state.total = q.total;
        state.chosen = null;
        renderQuestion(q);
        show("screen-question");
        startTimer();
      })
      .catch(function (err) { fail(err); });
  }

  function renderQuestion(q) {
    $("progress").textContent = "Q " + q.position + "/" + q.total;
    $("timer").textContent = "0:00";
    $("q-context").textContent = q.headline
      ? "From the Guardian · " + q.headline
      : "From this week's news";
    $("q-text").textContent = q.question;

    var letters = ["A", "B", "C", "D"];
    var container = $("options");
    container.innerHTML = "";
    q.options.forEach(function (text, i) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "option";
      btn.setAttribute("role", "radio");
      btn.setAttribute("aria-checked", "false");
      btn.dataset.letter = letters[i];
      btn.innerHTML =
        '<span class="option-letter">' + letters[i] + "</span><span></span>";
      btn.lastChild.textContent = text;
      btn.addEventListener("click", function () { choose(btn); });
      container.appendChild(btn);
    });

    $("confidence-slider").value = 50;
    $("confidence-value").textContent = "50";
    $("btn-submit").disabled = true;
    $("question-error").hidden = true;
  }

  function choose(btn) {
    document.querySelectorAll(".option").forEach(function (el) {
      el.setAttribute("aria-checked", "false");
    });
    btn.setAttribute("aria-checked", "true");
    state.chosen = btn.dataset.letter;
    $("btn-submit").disabled = false;
  }

  function submit() {
    if (!state.chosen) return;
    $("btn-submit").disabled = true;
    stopTimer();
    api("POST", "/api/presentations/" + state.presentationId + "/answer", {
      answer_letter: state.chosen,
      confidence: parseInt($("confidence-slider").value, 10),
    }).then(function (fb) {
      $("fb-progress").textContent = "Q " + fb.position + "/" + fb.total;
      $("fb-time").textContent = fmtClock(fb.elapsed_ms);
      var verdict = $("fb-verdict");
      verdict.textContent = fb.is_correct
        ? "Correct."
        : "Not this time — it was " + fb.correct_letter + ".";
      verdict.className = "verdict " + (fb.is_correct ? "correct" : "wrong");
      $("fb-explanation").textContent = fb.explanation || "";
      $("btn-next").textContent = fb.quiz_complete ? "See the photo finish" : "Next question";
      state.quizComplete = fb.quiz_complete;
      show("screen-feedback");
    }).catch(function (err) {
      if (err.status === 409) { return advance(state.position, state.total); }
      var el = $("question-error");
      el.textContent = err.message;
      el.hidden = false;
      $("btn-submit").disabled = false;
    });
  }

  function advance(position, total) {
    if (position >= total) { return showSummary(); }
    present(position + 1);
  }

  function next() {
    if (state.quizComplete) { showSummary(); } else { present(state.position + 1); }
  }

  // ---- summary / polling ----

  function showSummary() {
    show("screen-summary");
    pollSummary();
  }

  function pollSummary() {
    api("GET", "/api/sessions/" + state.sessionId + "/summary")
      .then(function (sum) {
        renderSummary(sum);
        var pending = sum.ai.some(function (m) { return m.n_pending > 0; });
        $("sum-pending-note").hidden = !pending;
        if (pending) {
          state.pollHandle = setTimeout(pollSummary, 2000);
        }
      })
      .catch(function () { state.pollHandle = setTimeout(pollSummary, 4000); });
  }

  function laneRow(name, isYou, rows, total, score, timeMs) {
    var lane = document.createElement("div");
    lane.className = "lane";

    var nameEl = document.createElement("span");
    nameEl.className = "lane-name";
    nameEl.textContent = name;
    if (isYou) nameEl.style.color = "var(--turf)";

    var ticks = document.createElement("span");
    ticks.className = "lane-ticks";
    for (var p = 1; p <= total; p++) {
      var tick = document.createElement("span");
      tick.className = "tick";
      var row = rows[p];
      if (row) {
        if (row.status === "pending") tick.classList.add("pending");
        else if (row.status === "error") tick.classList.add("error");
        else tick.classList.add(row.is_correct ? "correct" : "wrong");
      }
      ticks.appendChild(tick);
    }

    var scoreEl = document.createElement("span");
    scoreEl.className = "lane-score";
    scoreEl.textContent = score + (timeMs != null ? " · " + fmtClock(timeMs) : "");

    lane.appendChild(nameEl);
    lane.appendChild(ticks);
    lane.appendChild(scoreEl);
    return lane;
  }

  function renderSummary(sum) {
    var h = sum.human;
    $("sum-headline").textContent =
      "You finished " + h.n_correct + "/" + sum.total;
    $("sum-time").textContent =
      "Total time on the clock: " + fmtClock(h.total_elapsed_ms) +
      ". Green squares are correct answers; red were misses.";

    var lanes = $("sum-lanes");
    lanes.innerHTML = "";

    var humanRows = {};
    h.questions.forEach(function (r) {
      humanRows[r.position] = { status: "ok", is_correct: r.is_correct };
    });
    lanes.appendChild(laneRow(
      "You", true, humanRows, sum.total,
      h.n_correct + "/" + sum.total, h.total_elapsed_ms
    ));

    sum.ai.forEach(function (m) {
      var rows = {};
      m.questions.forEach(function (r) {
        rows[r.position] = { status: r.status, is_correct: r.is_correct };
      });
      var score = m.n_pending > 0
        ? m.n_correct + "/" + sum.total + " …"
        : m.n_correct + "/" + sum.total;
      lanes.appendChild(laneRow(
        m.display_name, false, rows, sum.total, score,
        m.n_answered > 0 ? m.total_elapsed_ms : null
      ));
    });
  }

  function fail(err) {
    // A broken session (e.g. DB reset) should not strand the visitor.
    sessionStorage.removeItem(SESSION_KEY);
    var el = $("start-error");
    el.textContent = err.message + " — start a fresh race.";
    el.hidden = false;
    show("screen-start");
  }

  function resume(sessionId) {
    state.sessionId = sessionId;
    api("GET", "/api/sessions/" + sessionId + "/resume")
      .then(function (data) {
        state.total = data.summary.total;
        if (data.summary.completed) { showSummary(); }
        else { present(data.position); }
      })
      .catch(function () {
        sessionStorage.removeItem(SESSION_KEY);
        show("screen-start");
      });
  }

  // ---- wire up ----

  $("btn-start").addEventListener("click", startQuiz);
  $("btn-submit").addEventListener("click", submit);
  $("btn-next").addEventListener("click", next);
  $("btn-again").addEventListener("click", function () {
    sessionStorage.removeItem(SESSION_KEY);
    if (state.pollHandle) clearTimeout(state.pollHandle);
    startQuiz();
  });
  $("confidence-slider").addEventListener("input", function (e) {
    $("confidence-value").textContent = e.target.value;
  });

  var saved = sessionStorage.getItem(SESSION_KEY);
  if (saved) { resume(saved); } else { show("screen-start"); }
})();
